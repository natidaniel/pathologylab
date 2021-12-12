#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import copy

def poly_area(x, y, absoluteValue = True):

    result = 0.5 * np.abs(np.dot(x[:-1], y[1:]) + x[-1]*y[0] - np.dot(y[:-1], x[1:]) - y[-1]*x[0])
    if absoluteValue:
        return abs(result)
    else:
        return result


with open("C:\\Users\\dekelmeirom\\OneDrive - Technion\\Documents\\university\\PDL_project\\CCD_Project_28MAR21_all_images.json") as json_file:
    json_data = json.load(json_file)

images = json_data["_via_img_metadata"]

images_names = list(images.keys())
working_images = images_names

for image_name in working_images:
    image = images[image_name]
    for region in image['regions']:
        area = poly_area(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"])
        region["area"] = area

# ### calculate the total region area for each class in each image

for image_name in working_images:
    positive_area = 0
    negative_area = 0
    image = images[image_name]
    for region in image['regions']:
        if region['region_attributes']['type'] == '2':
            negative_area += region["area"]
        elif region['region_attributes']['type'] == '3':
            positive_area += region["area"]
    image['positive_area'] = positive_area
    image['negative_area'] = negative_area  

print("image_name, positive, negative")
for image_name in working_images:
    image = images[image_name]
    print(image_name + ", " + str(image['positive_area']) + ", " + str(image['negative_area']))

# ### split train and test

test_image_names_path = "C:\\Users\\dekelmeirom\\OneDrive - Technion\\Documents\\university\\PDL_project\\test_image_names_all.txt"
with open(test_image_names_path, 'r') as test_image_names_file:
    test_image_names = test_image_names_file.read()
test_image_names = test_image_names.split("\n")
# test_image_names = test_image_names[:-1] # remove the last empty element

# remove bad labeled and duplicated images
ban_list = [test_image_names[5], test_image_names[8], test_image_names[9], test_image_names[-2], test_image_names[-1]]
print(ban_list)

save_path = "C:\\Users\\dekelmeirom\\OneDrive - Technion\\Documents\\university\\PDL_project\\"
test_path = save_path + "PDL_Project_test_256.json"
train_path = save_path + "PDL_Project_train_256.json"
working_images = images_names #[0:215] #images_names[59:161]
test_images = {}
train_images = {}
for image_name in working_images:
    if image_name in test_image_names:
        if image_name not in ban_list:
            test_images[image_name] = images[image_name]
    else:
        train_images[image_name] = images[image_name]
        
test = copy.deepcopy(json_data)
train = copy.deepcopy(json_data)
test["_via_img_metadata"] = test_images
train["_via_img_metadata"] = train_images

with open(test_path, 'w') as test_file:
    json.dump(test_images, test_file)
with open(train_path, 'w') as train_file:
    json.dump(train_images, train_file)
