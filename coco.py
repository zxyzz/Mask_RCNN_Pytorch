"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

from config import Config
import utils
import model as modellib

import torch
import wandb
import sys
import matplotlib.patches as patches

from datetime import datetime as odatetime

cats = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter',
        'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
        'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
        'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Configurations
############################################################

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, device, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = np.random.permutation(image_ids)[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        # r = model.detect([image])[0]
        r = model.detect([image], device)[0]
        t_prediction += (time.time() - t)

        fig = plt.figure(figsize=(15, 7))
        for idx in range(r['rois'].shape[0]):
            if idx == 0:
                ax = fig.add_subplot(151)
                ax.imshow(image)
            else:
                idx -= 1
                mask = r["masks"][:, :, idx]
                b = r['rois'][idx]
                ax = fig.add_subplot(idx // 5 + 5, 5, idx + 1)
                ax.imshow(mask)
                rect = patches.Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], linewidth=2, edgecolor='r',
                                         facecolor='none')
                rx, ry = rect.get_xy()
                plt.text(rx, ry + 30, cats[r['class_ids'][idx]], fontsize=13, color='r', weight='bold')
                # rect = patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=4, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.savefig(f'/home/lts5/Pictures/pred_{image_id}_{coco_image_ids[i]}.png')
        # a=1

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("--command",
                        default="train",
                        help="'train' or 'eval' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        default="./data",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        type=int,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False,
                        default="./logs/coco20211127T1331/mask_rcnn_coco_0005.pth",
                        help="Path to weights .pth file or 'coco'") # --model ./logs/coco20211201T1924_res18/mask_rcnn_coco_0015
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=2,
                        type=int,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    parser.add_argument('--lr', required=False,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument('--batchsize', required=False,
                        default=4,
                        type=int,
                        help='Batch size')
    parser.add_argument('--steps', required=False,
                        default=10,
                        type=int,
                        help='steps per epoch')    
    parser.add_argument('--device', required=False,
                        default="gpu",
                        help='gpu or cpu')
    parser.add_argument('--epochs', required=False,
                        default=10,
                        type=int,
                        help='Number of training epochs')
    parser.add_argument('--epochs2', required=False,
                        default=15,
                        type=int,
                        help='Number of Stage 2 epochs')
    parser.add_argument('--epochs3', required=False,
                        default=20,
                        type=int,
                        help='Number of Stage 3 epochs')
    parser.add_argument('--use_wandb', default=False, action='store_true', help='Use wandb.')
    parser.add_argument('--load', default=False, action='store_true', help='Load model.')


    args = parser.parse_args()                        

    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()

    # config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)

    # Select Device
    if args.device == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    model = model.to(device)

    # Select weights file to load
    # if args.model:
    #     if args.model.lower() == "coco":
    #         model_path = COCO_MODEL_PATH
    #     elif args.model.lower() == "last":
    #         # Find last trained weights
    #         model_path = model.find_last()[1]
    #     elif args.model.lower() == "imagenet":
    #         # Start from ImageNet trained weights
    #         model_path = config.IMAGENET_MODEL_PATH
    #     else:
    #         model_path = args.model
    # else:
    #     model_path = ""

    # Load weights
    if args.command == "eval" or args.load:
        assert os.path.isfile(args.model+'.pth')
        print("Loading local weights ", args.model)
        model.load_weights(args.model)
    else:
        print("No local weights were loadedlaa")

    # input parameters
    lr=float(args.lr)
    batchsize=int(args.batchsize)
    steps=int(args.steps)

    # output_filename = 'mrcnn_log'
    # with open(f'./logs/{output_filename}.txt', 'a') as f:
    with open('./logs/mrcnn_log.txt', 'a') as f:
        dt_string = odatetime.now().strftime("%d/%m/%Y %H:%M:%S")
        f.write(f"######## {dt_string} ######## {model.checkpoint_path}\n")
        f.write(' '.join(sys.argv[1:]) + "\n")
    print(dt_string)

    # create dictionaries of categories and id
    # with open('cat80.txt') as file:
    #     cats = file.readlines()
    # dict_id2cat = dict()
    # for id_, cat in enumerate(cats):
    #     dict_id2cat[id_ + 1] = cat.replace('\n', '')
    # dict_cat2id = {v: k for k, v in dict_id2cat.items()}

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.

        if args.use_wandb:
            wandb.init(project='mrcnn', entity='zxyzz')
            w_config = wandb.config
            w_config.learning_rate = lr
            wandb.watch(model, log='all', log_freq=1)

        #laa
        dataset_train = CocoDataset()
        coco = dataset_train.load_coco(args.dataset, 'train', year=2017, auto_download=False,
                                       class_ids=None, class_map=None, return_coco=True)
        # dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        # dataset_val = CocoDataset()
        # coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True,
        #                              auto_download=True)
        # dataset_val.prepare()

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train,
                          learning_rate=lr,
                          epochs=args.epochs,
                          BatchSize=batchsize,
                          steps=steps,
                          layers='heads',
                          use_wandb=args.use_wandb)

        # print("Stage 1: Running COCO evaluation on {} images.".format(args.limit))
        # evaluate_coco(model, dataset_train, coco, device, "bbox", limit=2)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train,
                    learning_rate=lr,
                    epochs=args.epochs2,
                    BatchSize=batchsize,
                    steps=steps,
                    layers='4+',
                          use_wandb=args.use_wandb)

        # print("Stage 2: Running COCO evaluation on {} images.".format(args.limit))
        # evaluate_coco(model, dataset_train, coco, device, "bbox", limit=int(args.limit))

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train,
                    learning_rate=lr / 10,
                    epochs=args.epochs3,
                    BatchSize=batchsize,
                    steps=steps,
                    layers='all',
                          use_wandb=args.use_wandb)

        # print("Stage 3: Running COCO evaluation on {} images.".format(args.limit))
        # evaluate_coco(model, dataset_train, coco, device, "bbox", limit=int(args.limit))


    elif args.command == "eval":
        # Validation dataset
        # dataset_val = CocoDataset()
        # coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
        # dataset_val.prepare()

        dataset_val = CocoDataset()
        coco = dataset_val.load_coco(args.dataset, 'val', year=2017, auto_download=False,
                                       class_ids=None, class_map=None, return_coco=True)
        dataset_val.prepare()

        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, device, "bbox", limit=args.limit)

        # evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
        # evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))


    print("Done")