#!/usr/bin/env python

import cv2
import numpy as np
import os

def count_areas_from_image_labels(labels_path, output_path):
    labels_image_names = os.listdir(labels_path)
    # legend = {0:"other", 1:"inflammation", 2:"negative", 3:"positive", 4:"black-pad", 5:"air", 6:"cell", 7:"noise"}
    
    with open(os.path.join(output_path, "PDL_areas.csv"), mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["image_name", "positive area", "negative area", "other area"])
        
        total_positive = 0
        total_negative = 0
        total_other = 0
        for label_name in labels_image_names:
            positive = 0
            negative = 0
            other = 0
            label_image = cv2.imread(os.path.join(labels_path, label_name))
            for i in range(label_image.shape[0]):
                for j in range(label_image.shape[1]):
                    if label_image[i, j, 0] == 2:
                        negative += 1
                    elif label_image[i, j, 0] == 3:
                        positive += 1
                    elif label_image[i, j, 0] != 4: # do not include black-pad
                        other += 1
            total_positive += positive
            total_negative += negative
            total_other += other
            
            csv_writer.writerow([label_name, positive, negative, other])
            
        csv_writer.writerow(["total", total_positive, total_negative, total_other])
        
def main():
    parser = argparse.ArgumentParser(description='This script is ...'
                                    , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", "-i", default="./input",
                        help="Path to the directory of the images labels. default='./input'")
    parser.add_argument("--output", "-o", default="./output",
                        help="Directory name where the result csv file will be saved. default='./output'")

    args = parser.parse_args()
    count_areas_from_image_labels(args.input, args.output)
    
    return


if __name__ == "__main__":
    main()

