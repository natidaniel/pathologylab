#!/usr/bin/env python

import copy
import os
import argparse
import json

def split_train_and_test(json_path, output_path, test_images_names_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    images = json_data["_via_img_metadata"]
    images_names = list(images.keys())
    
    with open(test_images_names_path, 'r') as test_image_names_file:
        test_image_names = test_image_names_file.read()
    test_image_names = test_image_names.split("\n")
    
    test_path = os.path.join(output_path, "PDL_Project_test.json")
    train_path = os.path.join(output_path, "PDL_Project_train.json")
    
    test_images = {}
    train_images = {}
    for image_name in images_names:
        if image_name in test_image_names:
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
        
def main():
    parser = argparse.ArgumentParser(description='This script is ...'
                                    , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", "-i", default="./CCD_Project.json",
                        help="Path to where the json file is saved. default='./CCD_Project.json'")
    parser.add_argument("--output", "-o", default="./output",
                        help="Directory name where the result josn files will be saved. default='./output'")
    parser.add_argument("--test_names", "-t", default="./test_names.txt",
                        help="Path to where the text file containing the test images names is located. default='./test_names.txt'")

    args = parser.parse_args()
    split_train_and_test(args.input, args.output, args.test_names)
    
    return


if __name__ == "__main__":
    main()

