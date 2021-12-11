#!/usr/bin/env python

import json
import numpy as np
import os
import csv
import argparse

def poly_area(x, y, absoluteValue = True):

    result = 0.5 * np.abs(np.dot(x[:-1], y[1:]) + x[-1]*y[0] - np.dot(y[:-1], x[1:]) - y[-1]*x[0])
    if absoluteValue:
        return abs(result)
    else:
        return result
    
def calculate_areas_from_json(json_path, output_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    images = json_data["_via_img_metadata"]
    images_names = list(images.keys())
    with open(os.path.join(output_path, "PDL_areas.csv"), mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["image_name", "positive area", "negative area"])
        for image_name in images_names:
            image = images[image_name]
            positive_area = 0
            negative_area = 0
            for region in image['regions']:
                area = poly_area(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"])
                if region['region_attributes']['type'] == '2':
                    negative_area += area
                elif region['region_attributes']['type'] == '3':
                    positive_area += area
            csv_writer.writerow([image_name, positive_area, negative_area])
        
def main():
    parser = argparse.ArgumentParser(description='This script is ...'
                                    , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", "-i", default="./CCD_Project.json",
                        help="Path to where the json file is saved. default='./CCD_Project.json'")
    parser.add_argument("--output", "-o", default="./output",
                        help="Directory name where the result csv file will be saved. default='./output'")

    args = parser.parse_args()
    calculate_areas_from_json(args.input, args.output)
    
    return


if __name__ == "__main__":
    main()
