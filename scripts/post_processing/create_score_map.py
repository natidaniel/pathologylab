#!/usr/bin/env python

import numpy as np
import argparse
import pickle
import csv
import os
import matplotlib.pyplot as plt

def extract_row_and_col(img_name):
    row = img_name.split("_")[2]
    col = img_name.split("_")[3].split(".")[0]
    return row, col

def create_score_map(input_path, output_path):
    with open(input_path, "rb") as results_file:
        results_data = pickle.load(results_file)

    with open(os.path.join(output_path, "real_slide_scores_locations.csv"), mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        max_row = 0
        max_col = 0
        for img_name in results_data['areas_per_image_air_filt']:
            row, col = extract_row_and_col(img_name)
            max_row = max(int(row), max_row)
            max_col = max(int(col), max_col)
        slide_matrix = np.zeros((max_row, max_col))
        slide_matrix[:] = np.nan
        for img_name in results_data['areas_per_image_air_filt']:
            row, col = extract_row_and_col(img_name)
            img_areas = results_data['areas_per_image_air_filt'][img_name]
            if img_areas[0] + img_areas[1] == 0:
                img_pred_score = 0
            else:
                img_pred_score = img_areas[0] / (img_areas[0] + img_areas[1])
            slide_matrix[int(row)-1, int(col)-1] = img_pred_score
        for i in range(slide_matrix.shape[0]):
            new_row = []
            for j in range(slide_matrix.shape[1]):
                new_row.append(slide_matrix[i, j])
            csv_writer.writerow(new_row)
            
    plt.figure()
    plt.imshow(slide_matrix, cmap=plt.get_cmap('jet') )
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "real_slide_location_map.png"))
        
def main():
    parser = argparse.ArgumentParser(description='This script is ...'
                                    , formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", "-i", default="./input",
                        help="Path to the directory of the result pickle file. default='./input'")
    parser.add_argument("--output", "-o", default="./output",
                        help="Directory name where the result csv file and image will be saved. default='./output'")

    args = parser.parse_args()
    create_score_map(args.input, args.output)
    
    return


if __name__ == "__main__":
    main()
