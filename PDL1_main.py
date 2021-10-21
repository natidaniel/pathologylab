"""
Mask R-CNN
Train on the toy PDL1 dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 pdl1_playground.py train --dataset=/path/to/PDL1/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 pdl1_playground.py train --dataset=/path/to/PDL1/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 pdl1_playground.py train --dataset=/path/to/PDL1/dataset --weights=imagenet

    # Apply color splash to an image
    python3 pdl1_playground.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 pdl1_playground.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

# Root directory of the project
ROOT_DIR = os.getcwd()
ALGO_ROOT = os.path.join(ROOT_DIR, 'algo')

# Import Mask RCNN
sys.path.append(ROOT_DIR)
sys.path.append(ALGO_ROOT)

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ALGO_ROOT, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import algo.PDL1Net.PDL1NetTrainer as PDL1NetTrainer
import algo.PDL1Net.PDL1NetTester as PDL1NetTester
# import datautils.PDL1Net_DataLoader
import params.PDL1NetConfig as PDL1NetConfig

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect PDL1.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/PDL1/dataset/",
                        help='Directory of the PDL1 dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--augment', required=False, action="store_true",
                        help='if the flag is used then train will run with augmenter')
    parser.add_argument('--result_dir', required=False,
                        metavar="/path/to/result/",
                        help='Path to output folder when running on test dataset')
    parser.add_argument('--synthetic', required=False,
                        action="store_true",
                        help='True if the data is synthetic')
    parser.add_argument('--real', required=False,
                        action="store_true",
                        help='True if the data is real cropped slide')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train" or args.command == "test":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
    elif args.command == "splash_mask":
        assert args.image, "Provide --image to apply splash_mask"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PDL1NetConfig.PDL1NetConfig()
    else:
        class InferenceConfig(PDL1NetConfig.PDL1NetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    print("create model")
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        Trainer = PDL1NetTrainer.PDL1NetTrainer(model, config, args)
        Trainer.train(augment=args.augment)
    elif args.command == "splash":
        Tester = PDL1NetTester.PDL1NetTester(model, args)
        Tester.detect_and_color_splash(args.image)
    elif args.command == "splash_mask":
        Tester = PDL1NetTester.PDL1NetTester(model, args)
        Tester.detect_and_show_mask(image_path=args.image)
    elif args.command == "test":
        Tester = PDL1NetTester.PDL1NetTester(model, args)
        Tester.test_sequence(result_dir_name=args.result_dir, real_slide=args.real)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
