"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import datetime
import math
import os
import random
import re

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import utils
import visualize
# from nms.nms_wrapper import nms
# # from torchvision.ops import nms, roi_pool, roi_align
# from roialign.roi_align.crop_and_resize import CropAndResizeFunction
# from roi_align.crop_and_resize import CropAndResizeFunction
from torchvision.ops import nms
from torchvision.ops import roi_align
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib.patches as patches
import wandb
from datetime import datetime as odatetime
import torchvision.models as models
from torchvision import transforms as pth_transforms
import PIL
import cv2
import vision_transformer as vits
# from roi_align.crop_and_resize import CropAndResizeFunction
cats = ['BG','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter',
        'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
        'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
        'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

RES18 = False
VITS = True

# RES18 = True
# VITS = False

# RES18 = False
# VITS = False

NORM_B = True

# NORM_B = False


# import warnings
# warnings.filterwarnings('ignore')
def norm_image(img):
    return (img - img.min()) / (img.max() - img.min())
############################################################
# This is inspired by https://github.com/multimodallearning/pytorch-mask-rcnn

############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]

def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2

############################################################
#  Resnet Implementation 
#  Modified from torchvision.models.resnet 
#  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
############################################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, architecture, stage5=False):
        super(ResNet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x


    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

############################################################
#  Feature Pyramid Network
############################################################


class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        # Resnet backbone
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )        
        self.P4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )      
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)   
        
    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)
        # p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        # p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        # p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
        p4_out = self.P4_conv1(c4_out) + F.interpolate(p5_out, scale_factor=2)  # F.upsample(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.interpolate(p4_out, scale_factor=2)  # F.upsample(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.interpolate(p3_out, scale_factor=2)  # F.upsample(p3_out, scale_factor=2)
        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out]





############################################################
#  Proposal Generator
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0].detach() + 0.5 * height
    center_x = boxes[:, 1].detach() + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0].detach() * height
    center_x += deltas[:, 1].detach() * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def proposal_generator(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.data, :] 
    anchors = anchors[order.data, :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy
    # for small objects, so we're skipping it.

    # Non-max suppression
    keep = nms(boxes, scores, nms_threshold)#nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    if NORM_B:
        # Normalize dimensions to range of 0 to 1.
        norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
        if config.GPU_COUNT:
            norm = norm.cuda()
        normalized_boxes = boxes / norm

        # Add back batch dimension
        normalized_boxes = normalized_boxes.unsqueeze(0)

        return normalized_boxes
    else:
        return boxes


############################################################
#  RO Align 
############################################################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    # Feature Maps. List of feature maps from different level of the
    # feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    if len(boxes.shape) != 2:
        boxes = boxes.unsqueeze(0)
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper. Account for
    # the fact that our coordinates are normalized here.
    # e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
    roi_level = roi_level.round().int()

    pooled = []
    box_to_level = []
    # if VITS or RES18:
    if RES18:
        roi_level = roi_level.clamp(2,2)
        nb = range(2, 3)
    else:
        # Loop through levels and apply ROI pooling to each. P2 to P5.
        roi_level = roi_level.clamp(2,5)
        nb = range(2, 6)

    for i, level in enumerate(nb):
    # for i, level in enumerate(range(2, 6)):
        ix = roi_level==level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:,0]
        level_boxes = boxes[ix.data, :]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]),requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        feature_maps[i] = feature_maps[i].unsqueeze(0)  #CropAndResizeFunction needs batch dimension
        # pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled_features = roi_align(feature_maps[i], [level_boxes], output_size=(pool_size, pool_size))
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


############################################################
#  Detection Target Layer
############################################################
def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.

    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps

def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    # remove batch dimension
    proposals = proposals.squeeze(0) #prrob nan 1 0 -> so prob rpn_rois
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    if torch.nonzero(gt_class_ids < 0).size()[0]:
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
        crowd_boxes = gt_boxes[crowd_ix.data, :]
        crowd_masks = gt_masks[crowd_ix.data, :, :]
        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[non_crowd_ix.data, :]
        gt_masks = gt_masks[non_crowd_ix.data, :]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0]*[True]), requires_grad=False)
        if config.GPU_COUNT:
            no_crowd_bool = no_crowd_bool.cuda()

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    if torch.nonzero(positive_roi_bool).size()[0]:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data,:]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data,:] # <- iou values of prop. box et gt: (number positive count, number of gt boxes )
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data,:]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        # Compute bbox refinement for positive ROIs
        deltas = Variable(utils.box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data,:,:]

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
        if config.GPU_COUNT:
            box_ids = box_ids.cuda()
        # ori
        # masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)

        # masks = Variable(
        #     CropAndResizeFunction.apply(roi_masks.unsqueeze(1), boxes, box_ids, config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0))
        if NORM_B:
            masks = roi_align(roi_masks.unsqueeze(1).detach(), [boxes*gt_masks.shape[-1]], output_size=(config.MASK_SHAPE[0], config.MASK_SHAPE[1])).squeeze(1)

            # masks = roi_align(roi_masks.unsqueeze(1).detach(), [boxes], output_size=(config.MASK_SHAPE[0], config.MASK_SHAPE[1]), spatial_scale=1024)
            # y1, x1, y2, x2 = boxes.chunk(4, dim=1)
            # boxes = torch.cat([x1, y1, x2, y2], dim=1)
            # masks = roi_align(roi_masks.unsqueeze(1).detach(), [boxes*gt_masks.shape[-1]], output_size=(config.MASK_SHAPE[0], config.MASK_SHAPE[1]))

            # cust masks
            # masks = []
            for idx, box in enumerate(torch.round(boxes * 1024).cpu().detach().numpy()):
                y1, x1, y2, x2 = box.astype(int)
                # a = cv2.resize(roi_masks[idx].cpu().detach().numpy()[y1:y2, x1:x2],
                #                (config.MASK_SHAPE[0], config.MASK_SHAPE[1]))
                a = cv2.resize(roi_masks[idx].cpu().detach().numpy()[y1:y2, x1:x2],
                               (config.MASK_SHAPE[0], config.MASK_SHAPE[1]), interpolation=cv2.INTER_LINEAR)
                masks[idx] = Variable(torch.from_numpy(a).float(), requires_grad=False)
                # masks.append(a)
                # plt.imshow(cv2.resize(a, (config.MASK_SHAPE[0], config.MASK_SHAPE[1])))
                # plt.imshow(a)
                # plt.show()

            # masks = Variable(torch.from_numpy(np.array(masks)), requires_grad=False)
        else:
            # All zero due to boxes in norm. coordinate
            masks = roi_align(roi_masks.unsqueeze(1).detach(), [boxes], output_size=(config.MASK_SHAPE[0], config.MASK_SHAPE[1])).squeeze(1)

            # Variable(torch.from_numpy(a).float(), requires_grad=False)
            # torch.from_numpy(scale)

        # np.where(roi_align(roi_masks.unsqueeze(1).detach(), [boxes*gt_masks.shape[-1]], output_size=(config.MASK_SHAPE[0], config.MASK_SHAPE[1])).squeeze(1).cpu().detach().numpy()>0)
        # np.where(roi_masks[0].cpu().detach().numpy()>0)
        # np.where(masks[0].cpu().detach().numpy()>0) #TODO
        #TODO proob boxes
        # fig, axs = plt.subplots(roi_masks.shape[0], 2, figsize=(10, 20))
        # for o in range(roi_masks.shape[0]):
        #     axs[o, 0].imshow(roi_masks[o].cpu().detach().numpy())
        #     b = boxes[o].cpu().detach().numpy() * 1024
        #     rect = patches.Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], linewidth=2, edgecolor='r',
        #                              facecolor='none')
        #     rx, ry = rect.get_xy()
        #     plt.text(rx, ry + 30, b, fontsize=13, color='r', weight='bold')
        #     axs[o, 0].add_patch(rect)
        #     axs[o, 1].imshow(masks[o].cpu().detach().numpy())
        # plt.show()

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)
    else:
        positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if torch.nonzero(negative_roi_bool).size()[0] and positive_count>0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count,4), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count,4), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()

    return rois, roi_gt_class_ids, deltas, masks


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    return boxes

def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
    class_scores = probs[idx, class_ids.data]
    deltas_specific = deltas[idx, class_ids.data]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

    # Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois *= scale

    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    # Filter out background boxes
    keep_bool = class_ids>0

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:,0]

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep.data]
    pre_nms_scores = class_scores[keep.data]
    pre_nms_rois = refined_rois[keep.data]

    # TODO pre_nms_class_ids is emtpy, because class_ids is all zero
    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]

        # Sort
        ix_rois = pre_nms_rois[ixs.data]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.data,:]

        # class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
        class_keep = nms(ix_rois, ix_scores, config.DETECTION_NMS_THRESHOLD)

        # Map indicies
        class_keep = keep[ixs[order[class_keep].data].data]

        if i==0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.data],
                        class_ids[keep.data].unsqueeze(1).float(),
                        class_scores[keep.data].unsqueeze(1)), dim=1)

    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    rois = rois.squeeze(0)

    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)

    return detections


############################################################
#  Region Proposal Network
############################################################

class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        # Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        # Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        # Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        # Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        # Bounding box refinement. [batch, H, W, anchors per location, depth]
        # where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        # Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0,2,3,1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Feature Pyramid Network Heads
############################################################

class Classifier(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois):
        co = x
        x = pyramid_roi_align([rois]+x, self.pool_size, self.image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1,1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        if rois.shape[0] != mrcnn_class_logits.shape[0]:
            x = pyramid_roi_align([rois] + co, self.pool_size, self.image_shape)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = x.view(-1, 1024)
            mrcnn_class_logits = self.linear_class(x)
            mrcnn_probs = self.softmax(mrcnn_class_logits)

            mrcnn_bbox = self.linear_bbox(x)
            mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]

class Mask(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:,0],indices.data[:,1],:]
    anchor_class = anchor_class[indices.data[:,0],indices.data[:,1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss

def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:,0],indices.data[:,1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0,:rpn_bbox.size()[0],:]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if target_class_ids.size()[0]:
        loss = F.cross_entropy(pred_class_logits,target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if target_class_ids.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0].data,:]
        pred_bbox = pred_bbox[indices[:,0].data,indices[:,1].data,:]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size()[0]:
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:,0].data,:,:]
        y_pred = pred_masks[indices[:,0].data,indices[:,1].data,:,:]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss

def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits) #7,81
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    image1=image
    mask, class_ids = dataset.load_mask(image_id)
    shape = image.shape
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)

    mask = utils.resize_mask(mask, scale, padding)
    # ok
    # fig = plt.figure(figsize=(15, 7))
    # for idx in range(mask.shape[-1] + 1):
    #     if idx == 0:
    #         ax = fig.add_subplot(151)
    #         ax.imshow(image)
    #     else:
    #         idx -= 1
    #         m = mask[:, :, idx]
    #         ax = fig.add_subplot(idx // 5 + 5, 5, idx + 1)
    #         ax.imshow(m)
    #         ax.title.set_text(class_ids[idx])
    # plt.show()

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)
    # ok
    # fig = plt.figure(figsize=(15, 7))
    # for idx in range(mask.shape[-1] + 1):
    #     if idx == 0:
    #         ax = fig.add_subplot(151)
    #         ax.imshow(image)
    #     else:
    #         idx -= 1
    #         m = mask[:, :, idx]
    #         b = bbox[idx]
    #         ax = fig.add_subplot(idx // 5 + 5, 5, idx + 1)
    #         ax.imshow(m)
    #         rect = patches.Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], linewidth=2, edgecolor='r',
    #                                  facecolor='none')
    #         rx, ry = rect.get_xy()
    #         plt.text(rx, ry + 30, class_ids[idx], fontsize=13, color='r', weight='bold')
    #         ax.add_patch(rect)
    #         ax.title.set_text(class_ids[idx])
    # plt.show()

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask: # set to False, otherwise mask will be resized to 56x56 -> deforme
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)

    return image, image_meta, class_ids, bbox, mask

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, config, augment=False):
        """A generator that returns images and corresponding target class ids,
            bounding box deltas, and masks.

            dataset: The Dataset object to pick data from
            config: The model config object
            shuffle: If True, shuffles the samples before every epoch
            augment: If True, applies image augmentation to images (currently only
                     horizontal flips are supported)

            Returns a Python generator. Upon calling next() on it, the
            generator returns two lists, inputs and outputs. The containtes
            of the lists differs depending on the received arguments:
            inputs list:
            - images: [batch, H, W, C]
            - image_metas: [batch, size of image meta]
            - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                        are those of the image unless use_mini_mask is True, in which
                        case they are defined in MINI_MASK_SHAPE.

            outputs list: Usually empty in regular training. But if detection_targets
                is True then the outputs list contains target class_ids, bbox deltas,
                and masks.
            """
        self.b = 0  # batch item index
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.dataset = dataset
        self.config = config
        self.augment = augment

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        if VITS:
            # 12288
            # self.anchors = utils.generate_pyramid_anchors([128],
            #                                               config.RPN_ANCHOR_RATIOS,
            #                                               [[64,64]],
            #                                               [16],
            #                                               config.RPN_ANCHOR_STRIDE)
            self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                    config.RPN_ANCHOR_RATIOS,
                                                                                    config.BACKBONE_SHAPES,
                                                                                    config.BACKBONE_STRIDES,
                                                                                    config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
        elif RES18:
            self.anchors = utils.generate_pyramid_anchors([128],
                                                          config.RPN_ANCHOR_RATIOS,
                                                          [[64, 64]],
                                                          [16],
                                                          config.RPN_ANCHOR_STRIDE)

            # vgg16
            # self.anchors = utils.generate_pyramid_anchors([128],
            #                                               config.RPN_ANCHOR_RATIOS,
            #                                               [[32,32]],
            #                                               [16],
            #                                               config.RPN_ANCHOR_STRIDE)

        else:
            self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                     config.RPN_ANCHOR_RATIOS,
                                                     config.BACKBONE_SHAPES,
                                                     config.BACKBONE_STRIDES,
                                                     config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, image_index):
        # Get GT bounding boxes and masks for image.
        image_id = self.image_ids[image_index]
        image, image_metas, gt_class_ids, gt_boxes, gt_masks = \
            load_image_gt(self.dataset, self.config, image_id, augment=self.augment,
                          use_mini_mask=self.config.USE_MINI_MASK)
        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            return None

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]


        # ok
        # fig = plt.figure(figsize=(15, 7))
        # for idx in range(gt_masks.shape[-1] + 1):
        #     if idx == 0:
        #         ax = fig.add_subplot(151)
        #         ax.imshow(image)
        #     else:
        #         idx -= 1
        #         mask = gt_masks[:, :, idx]
        #         b = gt_boxes[idx]
        #         ax = fig.add_subplot(idx // 5 + 5, 5, idx + 1)
        #         ax.imshow(mask)
        #         rect = patches.Rectangle((b[1], b[0]), b[3]-b[1], b[2]-b[0], linewidth=2, edgecolor='r', facecolor='none')
        #         rx, ry = rect.get_xy()
        #         plt.text(rx, ry + 30, cats[idx], fontsize=13, color='r', weight='bold')
        #         ax.add_patch(rect)
        #         #ax.title.set_text(gt_class_ids[idx])
        # plt.show()

        if VITS or RES18:
            images = image
            train_transform = pth_transforms.Compose([
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.407),(1,1,1)), #(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ]) # [i/255 for i in [123.7, 116.8, 103.9]]
            images = train_transform(images)
        else:
            images = mold_image(image.astype(np.float32), self.config)
            # Convert
            images = torch.from_numpy(images.transpose(2, 0, 1)).float()

        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()

        return images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks

    def __len__(self):
        return self.image_ids.shape[0]


############################################################
#  MaskRCNN Class
############################################################

class Pre_Resnet(nn.Module):
    def __init__(self):
        super(Pre_Resnet, self).__init__()
        # self.pretrained_resnet = models.resnet18(pretrained=True, progress=True)
        # self.pretrained_resnet = models.resnet18()
        self.pretrained_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # self.pretrained_resnet.eval()
        # self.pretrained_resnet = models.vgg16(pretrained=True, progress=True)


    def forward(self, inputs): # bs,3,1024,1024
        # return self.pretrained_resnet.features(inputs)

        x = self.pretrained_resnet.conv1(inputs)
        x = self.pretrained_resnet.bn1(x)
        x = self.pretrained_resnet.relu(x)
        x = self.pretrained_resnet.maxpool(x) # bs,64,256,256

        x = self.pretrained_resnet.layer1(x) # bs,64,256,256
        x = self.pretrained_resnet.layer2(x) # bs,256,128,128
        x = self.pretrained_resnet.layer3(x) # bs,256,64,64
        # x = self.pretrained_resnet.layer4(x) # bs,512,32,32

        return x  # self.pretrained_resnet.avgpool(x) # bs,512,1,1


class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
        """

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        if RES18:
            self.fpn = Pre_Resnet()
            # 12288
            self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors([128],
                                                                                  config.RPN_ANCHOR_RATIOS,
                                                                                  [[64, 64]],
                                                                                  [16],
                                                                                  config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
            # vgg16
            # self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors([128],
            #                                                                       config.RPN_ANCHOR_RATIOS,
            #                                                                       [[32, 32]],
            #                                                                       [16],
            #                                                                       config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)

        elif VITS:
            self.fpn = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
            print("Loading dino_deitsmall16_pretrain.pth")
            self.fpn.load_state_dict(torch.load('./data/dino_deitsmall16_pretrain.pth'))
            # self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors([128],
            #                                                                       config.RPN_ANCHOR_RATIOS,
            #                                                                       [[64, 64]],
            #                                                                       [16],
            #                                                                       config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
            self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                    config.RPN_ANCHOR_RATIOS,
                                                                                    config.BACKBONE_SHAPES,
                                                                                    config.BACKBONE_STRIDES,
                                                                                    config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)

        else:
            # Build the shared convolutional layers.
            # Bottom-up Layers
            # Returns a list of the last layers of each stage, 5 in total.
            # Don't create the thead (stage 5), so we pick the 4th item in the list.

            resnet = ResNet("resnet101", stage5=True)
            C1, C2, C3, C4, C5 = resnet.stages()  # C4
            # FPN network
            self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)
            # Generate Anchors
            self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                    config.RPN_ANCHOR_RATIOS,
                                                                                    config.BACKBONE_SHAPES,
                                                                                    config.BACKBONE_STRIDES,
                                                                                    config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)

        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()


        if VITS:
            depth = 384
        else:
            depth = 256

        # VGG16
        # depth =128

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, depth=depth)

        # FPN Classifier
        self.classifier = Classifier(depth, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # FPN Mask
        self.mask = Mask(depth, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def load_pre_weights(self, filepath):
        """load pre-trained weights from coco or imagenet
        """
        if os.path.exists(filepath):

            pretrained_dict = torch.load(filepath)
            model_dict = self.state_dict()
            # remove the useless layers
            del pretrained_dict['classifier.linear_class.bias']
            del pretrained_dict['classifier.linear_class.weight']
            del pretrained_dict['classifier.linear_bbox.bias']
            del pretrained_dict['classifier.linear_bbox.weight']
            del pretrained_dict['mask.conv5.bias']
            del pretrained_dict['mask.conv5.weight']
            #  overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            #  load the new state dict
            self.load_state_dict(model_dict)
        else:
            print("Weight file not found ...")

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images, device):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images)# torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        # To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.to(device)

        # Wrap in variable
        with torch.no_grad():
            molded_images = Variable(molded_images)

        # Run object detection
        detections, mrcnn_mask = self.predict([molded_images, image_metas], mode='inference')

        # Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def collate_custom(self, batch):
        """ Convert the input tensors into lists, to enable multi-image batch training
        """
        images = [item[0] for item in batch]
        image_metas = [item[1] for item in batch]
        rpn_match = [item[2] for item in batch]
        rpn_bbox = [item[3] for item in batch]
        gt_class_ids = [item[4] for item in batch]
        gt_boxes = [item[5] for item in batch]
        gt_masks = [item[6] for item in batch]

        return [images, image_metas,rpn_match,rpn_bbox,gt_class_ids,gt_boxes,gt_masks]
    

    def train_model(self, train_dataset, learning_rate, epochs,BatchSize,steps, layers,use_wandb=False):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(fpn.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # From a specific Resnet stage and up
            "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # All layers
            "all": ".*",
        } # "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_set = Dataset(train_dataset, self.config, augment=False)
        # train_set, val_set = torch.utils.data.random_split(train_set, [4000, len(train_set)-4000])
        if use_wandb:
            train_generator = torch.utils.data.DataLoader(train_set, batch_size=BatchSize,
                                                          collate_fn=self.collate_custom, shuffle=True, num_workers=4)
        else:
            train_generator = torch.utils.data.DataLoader(train_set, batch_size=BatchSize,
                                                          collate_fn=self.collate_custom, shuffle=True, num_workers=0)
        nb_batches = int(len(train_generator))
        print("Number of batches:", nb_batches)
        print("Number of images:", nb_batches*BatchSize)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)

        # training mode
        self.train()
        # Set batchnorm always in eval mode during training
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(set_bn_eval)
        
        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)
        # optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        # sum(p.numel() for p in self.parameters() if p.requires_grad) # 21'488'120
        # sum(p.numel() for p in self.fpn.parameters() if p.requires_grad) # 3'344'384
        # sum(p.numel() for p in self.classifier.parameters() if p.requires_grad) # 14'310'805
        # sum(p.numel() for p in self.mask.parameters() if p.requires_grad) # 2'643'537
        # [p.requires_grad for n,p in self.classifier.named_parameters()]

        # Training
        #laa
        for epoch in range(self.epoch+1, epochs+1):
            log("Epoch {}/{}.".format(epoch,epochs))
            epoch_loss = 0
            step = 0

            temp_param = list(self.classifier.parameters())[10]
            temp_param = [v for v in list(temp_param.cpu().detach().numpy())]

            start_time = time.time()
            for inputs in train_generator:
                optimizer.zero_grad()

                # torch.autograd.set_detect_anomaly(True)
                loss=0
                images,image_metas_batch,rpn_match_batch,rpn_bbox_batch,gt_class_ids_batch,gt_boxes_batch,gt_masks_batch = inputs

                # ok
                # i = 3
                # fig = plt.figure(figsize=(20, 10))
                # for o in range(gt_boxes_batch[i].shape[0]):
                #     if o == 0:
                #         ax = fig.add_subplot(151)
                #         ax.imshow(norm_image(rearrange(images[i].cpu().detach().numpy(), 'c b p -> b p c', c=3)))
                #     else:
                #         o -= 1
                #         b = gt_boxes_batch[i][o]
                #         m = gt_masks_batch[i][o]  # bbox array [num_instances, (y1, x1, y2, x2)]
                #         ids = gt_class_ids_batch[i][o]
                #         ax = fig.add_subplot(o // 5 + 5, 5, o + 1)
                #         ax.imshow(m)
                #         # x,y,w,h
                #         rect = patches.Rectangle((int(b[1].numpy()), int(b[0].numpy())),
                #                                  int(b[3].numpy()) - int(b[1].numpy()),
                #                                  int(b[2].numpy()) - int(b[0].numpy()),
                #                                  linewidth=4, edgecolor='r', facecolor='none')
                #         ax.add_patch(rect)
                #         # ax.title.set_text(f"{cats[ids]}, {b}")
                #         ax.title.set_text(cats[ids])
                # plt.tight_layout(pad=3)
                # plt.show()

                # from list to tensor
                images = torch.stack(images).cuda()

                # fee
                if VITS:
                    # vits
                    with torch.no_grad():
                        # n,4,4097,384
                        intermediate_output = self.fpn.get_intermediate_layers(images, n=4)
                        # n,4,384,64,64
                        intermediate_output = [rearrange(x[:, 1:, :], 'b (m n) c -> b c m n', n=64) for x in intermediate_output] # [torch.cat([x[:, 1:, :] for x in intermediate_output], dim=-1)]
                        # depth384, p2_out:n3, p3_out:n2, p4_out:n1, p5_out:n1+conv kernel2, p6_out:n1+conv kernel4
                        p2_out = F.interpolate(intermediate_output[0], scale_factor=4)
                        p3_out = F.interpolate(intermediate_output[1], scale_factor=2)
                        # p4_out = intermediate_output[3]
                        # p5_out = F.interpolate(intermediate_output[2], scale_factor=0.5)
                        # p6_out = F.interpolate(intermediate_output[2], scale_factor=0.25)
                        p4_out = intermediate_output[2]
			 p5_out = F.interpolate(intermediate_output[3], scale_factor=0.5)
			 p6_out = F.interpolate(intermediate_output[3], scale_factor=0.25)

                        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]#p4_out
                        # output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)  # bs, 64x64+1, 384
                elif RES18:
                    #np.where(rpn_feature_maps[0][0][0][2].cpu().detach().numpy()!=0)
                    self.fpn.eval()
                    # # bs, 512,32,32 if fpn is resnet18
                    rpn_feature_maps = [self.fpn(images)]

                    # vgg16 bs,128,64,64
                    # rpn_feature_maps = [rearrange(self.fpn(images), 'b (c e m) d f -> b m (d c) (f e)', m=128,c=2,e=2)]
                else:
                    # Feature extraction
                    # bs,256: 256,256 -> 128,128 -> 64,64 -> 32,32 -> 16,16
                    [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(images)
                    # Note that P6 is used in RPN, but not in the classifier heads.
                    rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]


                # Loop through pyramid layers
                layer_outputs = []  # list of lists
                for p in rpn_feature_maps:
                    layer_outputs.append(self.rpn(p))

                # Concatenate layer outputs
                # Convert from list of lists of level outputs to list of lists
                # of outputs across levels.
                # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
                outputs = list(zip(*layer_outputs))
                outputs = [torch.cat(list(o), dim=1) for o in outputs]
                # size: [batch, None, None];    For example
                # rpn_class_logits: torch.Size([2, 409200, 2]),
                # rpn_class: torch.Size([2, 409200, 2]),
                # rpn_bbox: torch.Size([2, 409200, 4])
                rpn_class_logits_batch, rpn_class_batch, rpn_pred_bbox_batch = outputs

                # Generate proposals
                # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
                # and zero padded.
                proposal_count = self.config.POST_NMS_ROIS_TRAINING

                # compute the losses for each batch
                for i in range(BatchSize): # loop over each image in the batch
                        # from list to tensor
                        image_metas=image_metas_batch[i]
                        rpn_class_logits=rpn_class_logits_batch[i]
                        rpn_class=rpn_class_batch[i]
                        rpn_match=rpn_match_batch[i]
                        rpn_bbox=rpn_bbox_batch[i]
                        rpn_pred_bbox=rpn_pred_bbox_batch[i]
                        gt_class_ids=gt_class_ids_batch[i]
                        gt_boxes=gt_boxes_batch[i]
                        gt_masks=gt_masks_batch[i]
                        
                        # add the batch dimension
                        image_metas=image_metas.unsqueeze(0)
                        rpn_class_logits=rpn_class_logits.unsqueeze(0)
                        rpn_class=rpn_class.unsqueeze(0)
                        rpn_match=rpn_match.unsqueeze(0)
                        rpn_bbox=rpn_bbox.unsqueeze(0)
                        rpn_pred_bbox=rpn_pred_bbox.unsqueeze(0)
                        gt_class_ids=gt_class_ids.unsqueeze(0)
                        gt_boxes=gt_boxes.unsqueeze(0)
                        gt_masks=gt_masks.unsqueeze(0)

                        
                        # image_metas as numpy array
                        image_metas = image_metas.numpy()
                        
                        # GPU or CPU
                        if self.config.GPU_COUNT:
                            images = images.cuda()
                            rpn_match = rpn_match.cuda()
                            rpn_bbox = rpn_bbox.cuda()
                            gt_class_ids = gt_class_ids.cuda()
                            gt_boxes = gt_boxes.cuda()
                            gt_masks = gt_masks.cuda()
                        
                        # compute rpn_rois
                        rpn_rois = proposal_generator([rpn_class, rpn_pred_bbox],
                                 proposal_count=proposal_count,
                                 nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                 anchors=self.anchors,
                                 config=self.config)

                        if NORM_B:
                            # Normalize coordinates
                            h, w = self.config.IMAGE_SHAPE[:2]
                            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
                            if self.config.GPU_COUNT:
                                scale = scale.cuda()
                            gt_boxes = gt_boxes / scale

                        # Generate detection targets
                        # Subsamples proposals and generates target outputs for training
                        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
                        # padded. Equally, returned rois and targets are zero padded.
                        rois, target_class_ids, target_deltas, target_mask = \
                            detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)
                        # may have 27 gt_box and only 6 roi when training -> mrcnn_class later will have 6 also, depends on roi

                        if not rois.size()[0]:
                            # print("1", rois.size()[0])
                            mrcnn_class_logits = Variable(torch.FloatTensor())
                            mrcnn_class = Variable(torch.IntTensor())
                            mrcnn_bbox = Variable(torch.FloatTensor())
                            mrcnn_mask = Variable(torch.FloatTensor())
                            if self.config.GPU_COUNT:
                                mrcnn_class_logits = mrcnn_class_logits.cuda()
                                mrcnn_class = mrcnn_class.cuda()
                                mrcnn_bbox = mrcnn_bbox.cuda()
                                mrcnn_mask = mrcnn_mask.cuda()
                        else: #prrob roi tjs: tensor([[0., 0., 1., 1.], [1., 0., 1., 0.], [1., 0., 1., 0.]], device='cuda:0', grad_fn=<CatBackward0>)
                            # check prediction masks
                            # nb_gt_ms = gt_masks.shape[1]
                            # nb_pred = target_class_ids.shape[0]
                            # nb_nonzero = np.sum(target_class_ids.cpu().detach().numpy() > 0)
                            # fig, axs = plt.subplots(int((nb_gt_ms + nb_nonzero) / 5) + 1, 5, figsize=(10, 20))
                            # gt_ms = gt_masks.cpu().detach().numpy()[0]
                            # gt_ids = gt_class_ids.cpu().detach().numpy()[0]
                            # half = int(gt_boxes.shape[0] / 2)
                            # gt_b = gt_boxes.cpu().detach().numpy()[0] * 1024
                            # for idx, nb_gt_m in enumerate(range(nb_gt_ms)):
                            #     axs[idx // 5, idx % 5].imshow(
                            #         rearrange(norm_image(images[i].cpu().detach().numpy()), 'c b p -> b p c', c=3))
                            #     # ax.imshow(gt_ms[idx])
                            #     rect = patches.Rectangle((gt_b[idx][1], gt_b[idx][0]), gt_b[idx][3] - gt_b[idx][1],
                            #                              gt_b[idx][2] - gt_b[idx][0], linewidth=2, edgecolor='r',
                            #                              facecolor='none')
                            #     rx, ry = rect.get_xy()
                            #     axs[idx // 5, idx % 5].add_patch(rect)
                            #     axs[idx // 5, idx % 5].title.set_text(cats[gt_ids[idx]])
                            # pred_ms = target_mask.cpu().detach().numpy()
                            # pred_ids = target_class_ids.cpu().detach().numpy()
                            # pred_bs = rois.cpu().detach().numpy() * 1024
                            # k = idx + 1
                            # for idx, pr in enumerate(range(nb_nonzero)):
                            #     axs[(idx + k) // 5, (idx + k) % 5].imshow(pred_ms[idx])
                            #     # ax.imshow(np.zeros((1024,1024)))
                            #     pred_b = pred_bs[idx]
                            #     rect = patches.Rectangle((pred_b[1], pred_b[0]), pred_b[3] - pred_b[1],
                            #                              pred_b[2] - pred_b[0], linewidth=2, edgecolor='r',
                            #                              facecolor='none')
                            #     axs[(idx + k) // 5, (idx + k) % 5].add_patch(rect)
                            #     axs[(idx + k) // 5, (idx + k) % 5].title.set_text(
                            #         f"{cats[pred_ids[idx]]}&{nb_pred - nb_nonzero}")
                            # plt.tight_layout(pad=2)
                            # plt.show()

                            # Network Heads
                            # Proposal classifier and BBox regressor heads
                            # print("2", rois.size()[0])

                            # if VITS or RES18:
                            if RES18:
                                mrcnn_feature_maps = [rpn_feature_maps[0][i].unsqueeze(0)]
                            else:
                                mrcnn_feature_maps = [p2_out[i].unsqueeze(0),p3_out[i].unsqueeze(0),p4_out[i].unsqueeze(0),p5_out[i].unsqueeze(0)]
                            # mrcnn_class_logits et mrcnn_class (nb_ob, 81); mrcnn_bbox (nb_ob, 81,4)
                            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)

                            # if rois 15x4 -> mrcnn_bbox 15x81x4; mrcnn_class_logits, mrcnn_class: 15x81
                            # Create masks for detections
                            mrcnn_mask = self.mask(mrcnn_feature_maps, rois) # 15,81,28,28
                            # m_ = np.where(mrcnn_mask.cpu().detach().numpy()>0)
                            # print("pred mask ", m_)

                            # see whether is there any nonzero prediction during training
                            # preds = [np.where(row>=0.5) for row in mrcnn_class.cpu().detach().numpy()]
                            # # print([p[0] for p in preds if len(p[0])!=0]) # multiple boxes (with multiple? id in box)
                            # obs = [p[0] for p in preds if len(p[0]) != 0]
                            # if obs:
                            #     print(obs)
                            # max_ = mrcnn_class.cpu().detach().numpy().max(axis=1)
                            # pre = [np.where(val == max_[idx])[0] for idx, val in enumerate(mrcnn_class.cpu().detach().numpy())]
                            # preds = list(set([np.argmax(v) for v in mrcnn_class.cpu().detach().numpy()]))
                            # if preds:
                            #     a = f"pred: {preds}   rpn:{list(set(target_class_ids.cpu().numpy()))}   gt:{list(set(gt_class_ids.cpu().numpy()[0]))}"
                            #     # print("---")
                            #     # print("pre raw",pre)
                            #     # print(a)
                            #     with open('./logs/mrcnn_log.txt', 'a') as f:
                            #         f.write(f"---\npre raw:{pre}\n")
                            #         f.write(f"{a}\n")
                            #         if np.where(mrcnn_mask.cpu().detach().numpy() > 0):
                            #             f.write("mrcnn_mask: not all zero\n")
                        # Compute losses
                        rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = \
                            compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                                           mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
                        img_loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss
                        loss=loss+img_loss
                loss=loss/BatchSize
                loss.backward()
                # TODO: need to find an appropriate number for 'clip' 
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
                # if rois.size()[0]:
                #     # grads = [p.grad for n, p in self.classifier.named_parameters() if p.requires_grad][6]
                #     val = list(self.classifier.parameters())[10]
                #     # True if grad is not all 0
                #     print("when rois non-empty, that param updated:", np.any([v_t!=v for v_t, v in zip(temp_param, list(val.cpu().detach().numpy()))]))
                #     # assert np.any([v_t!=v for v_t, v in zip(temp_param, list(val.cpu().detach().numpy()))])

                # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, step, steps, loss.item()))
                step=step+1
                if step%steps==0:
                    avg_loss = epoch_loss / step
                    s = "===> Epoch[{}]({}/{}): Avg Loss so far: {:.4f}".format(epoch, step, nb_batches, avg_loss)
                    # print(s)
                    with open('./logs/mrcnn_log.txt', 'a') as f:
                        f.write(s + "\n")
                    # log_dict = dict()
                    # log_dict['tr_loss'] = tr_loss
                    # log_dict['te_loss'] = te_loss
                    if use_wandb:
                        wandb.log({'loss': avg_loss})

                    if not os.path.exists(self.log_dir):
                        os.makedirs(self.log_dir)
                    torch.save(self.state_dict(), self.checkpoint_path.format(epoch))
                    #laa
            # save model
            # End of each epoch
            # torch.save(self.state_dict(), self.checkpoint_path.format(epoch))
            # break
            print("This training epoch took %.2f minutes" % ((time.time() - start_time) / 60.0))

        self.epoch = epochs


    def predict(self, input, mode):
        molded_images = input[0]
        image_metas = input[1]

        if mode == 'inference':
            self.eval()
        elif mode == 'training':
            self.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        # if VITS or RES18:
        if RES18:
            rpn_feature_maps = [self.fpn(molded_images)]
            mrcnn_feature_maps = [rpn_feature_maps[0]]
        else:
            # Feature extraction
            [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)
            # Note that P6 is used in RPN, but not in the classifier heads.
            rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
            mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_generator([rpn_class, rpn_bbox],
                                 proposal_count=proposal_count,
                                 nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                 anchors=self.anchors,
                                 config=self.config)

        if mode == 'inference':
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :4] / scale

            # Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            # Create masks for detections
            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)

            # Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)

            return [detections, mrcnn_mask]

        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]

            # Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if not rois.size()[0]:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
            else:
                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)

                # Create masks for detections
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois)

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask]
            

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            if VITS or RES18:
                train_transform = pth_transforms.Compose([
                    pth_transforms.ToTensor(),
                    pth_transforms.Normalize((0.485, 0.456, 0.407), (1, 1, 1)),
                    # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ])  # [i/255 for i in [123.7, 116.8, 103.9]]
                molded_image = train_transform(molded_image)
            else:
                molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
