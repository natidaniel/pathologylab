#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import numpy as np
import math
import os
import cv2
import copy
import json

from skimage.morphology import extrema
from skimage.morphology import watershed as skwater


# # rotating the images
imgs = [f for f in os.listdir(img_dist + "rotated")]

for img_name in imgs:
    orig_img = cv2.imread(img_dist + "rotated\\" + img_name)
    rotated = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(img_dist + "rotated\\" + img_name[:-4] + "_rotated" + img_name[-4:], rotated) 

imgs2 = [f for f in os.listdir(img_dist + "rotated")]

# rotated again
for img_name in imgs2:
    if "rotated" in img_name:
        orig_img = cv2.imread(img_dist + "rotated\\" + img_name)
        rotated = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(img_dist + "rotated\\" + img_name[:-4] + "2" + img_name[-4:], rotated) 

imgs3 = [f for f in os.listdir(img_dist + "rotated")]

# rotated again
for img_name in imgs3:
    if "rotated2" in img_name:
        orig_img = cv2.imread(img_dist + "rotated\\" + img_name)
        rotated = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(img_dist + "rotated\\" + img_name[:-5] + "3" + img_name[-4:], rotated) 


# ## rotating the annotations and creating the new json file

with open("C:\\Users\\dekelmeirom\\OneDrive - Technion\\Documents\\university\\PDL_project\\CCD_Project_15MAR21_215.json") as json_file:
    json_data = json.load(json_file)

images = json_data["_via_img_metadata"]
images_names = list(images.keys())
working_images = images_names[0:215]

with open(img_dist + "rotated_train_image_names.txt", 'r') as rotated_train_image_names_file:
    rotated_train_image_names = rotated_train_image_names_file.read()
rotated_train_image_names = rotated_train_image_names.split("\n")

with open(img_dist + "PDL_Project_train_215.json", 'r') as train_file:
    train_data = json.load(train_file)

new_train_images = copy.deepcopy(train_data)
for img_name in rotated_train_image_names:
    orig_img = cv2.imread(img_dist + "rotated\\" + img_name.split("png")[0][:-1] + ".png")
    index = images_names.index(img_name)
    regions = new_train_images[images_names[index]]['regions']
    new_regions = copy.deepcopy(regions)
    for region in new_regions:
        old_x = region['shape_attributes']['all_points_x']
        old_y = region['shape_attributes']['all_points_y']
        new_x = []
        new_y = []
        for i in range(len(old_x)):
            new_x.append(orig_img.shape[0] - old_y[i])
            new_y.append(old_x[i])
        region['shape_attributes']['all_points_x'] = new_x
        region['shape_attributes']['all_points_y'] = new_y
    new_image = copy.deepcopy(images[images_names[index]])
    new_image['regions'] = new_regions
    new_image['filename'] = new_image['filename'][:-4] + "_rotated" + new_image['filename'][-4:]
    new_train_images[img_name + "_rotated"] = new_image

# new_train = copy.deepcopy(json_data)
# new_train["_via_img_metadata"] = new_images
train_path = img_dist + "PDL_Project_aug_train_215.json"
with open(train_path, 'w') as train_file:
    json.dump(new_train_images, train_file)

with open(img_dist + "PDL_Project_aug_train_215.json", 'r') as train_file:
    train_data = json.load(train_file)

new_train_images = copy.deepcopy(train_data)
for img_name in rotated_train_image_names:
    orig_img = cv2.imread(img_dist + "rotated\\" + img_name.split("png")[0][:-1] + "_rotated.png")
    index = images_names.index(img_name)
    regions = new_train_images[images_names[index] + "_rotated"]['regions']
    new_regions = copy.deepcopy(regions)
    for region in new_regions:
        old_x = region['shape_attributes']['all_points_x']
        old_y = region['shape_attributes']['all_points_y']
        new_x = []
        new_y = []
        for i in range(len(old_x)):
            new_x.append(orig_img.shape[0] - old_y[i])
            new_y.append(old_x[i])
        region['shape_attributes']['all_points_x'] = new_x
        region['shape_attributes']['all_points_y'] = new_y
    new_image = copy.deepcopy(images[images_names[index]])
    new_image['regions'] = new_regions
    new_image['filename'] = new_image['filename'][:-4] + "_rotated2" + new_image['filename'][-4:]
    new_train_images[img_name + "_rotated2"] = new_image

train_path = img_dist + "PDL_Project_aug2_train_215.json"
with open(train_path, 'w') as train_file:
    json.dump(new_train_images, train_file)
    
new_train2 = copy.deepcopy(json_data)
new_train2["_via_img_metadata"] = new_train_images
train_path2 = img_dist + "PDL_Project_aug2_train_215_for_via.json"
with open(train_path2, 'w') as train_file2:
    json.dump(new_train2, train_file2)


# ## modify also some test images

with open(img_dist + "rotated_test_image_names.txt", 'r') as rotated_test_image_names_file:
    rotated_test_image_names = rotated_test_image_names_file.read()
rotated_test_image_names = rotated_test_image_names.split("\n")

with open(img_dist + "PDL_Project_test_215.json", 'r') as test_file:
    test_data = json.load(test_file)

new_test_images = copy.deepcopy(test_data)
for img_name in rotated_test_image_names:
    orig_img = cv2.imread(img_dist + "rotated\\" + img_name.split("png")[0][:-1] + ".png")
    index = images_names.index(img_name)
    regions = new_test_images[images_names[index]]['regions']
    new_regions = copy.deepcopy(regions)
    for region in new_regions:
        old_x = region['shape_attributes']['all_points_x']
        old_y = region['shape_attributes']['all_points_y']
        new_x = []
        new_y = []
        for i in range(len(old_x)):
            new_x.append(orig_img.shape[0] - old_y[i])
            new_y.append(old_x[i])
        region['shape_attributes']['all_points_x'] = new_x
        region['shape_attributes']['all_points_y'] = new_y
    new_image = copy.deepcopy(images[images_names[index]])
    new_image['regions'] = new_regions
    new_image['filename'] = new_image['filename'][:-4] + "_rotated" + new_image['filename'][-4:]
    new_test_images[img_name + "_rotated"] = new_image

test_path = img_dist + "PDL_Project_aug_test_215.json"
with open(test_path, 'w') as test_file:
    json.dump(new_test_images, test_file)
