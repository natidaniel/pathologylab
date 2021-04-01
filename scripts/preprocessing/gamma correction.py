#!/usr/bin/env python
# coding: utf-8

from skimage import exposure
from matplotlib import pyplot as plt
from matplotlib import colors as colors
import numpy as np
import matplotlib.image as mpimg
import colorsys
import math
import os

PATH = "C:\\Users\\dekelmeirom\\OneDrive - Technion\\Documents\\university\\pdl_project_res\\"
SAVE_PATH = "C:\\Users\\dekelmeirom\\OneDrive - Technion\\Documents\\university\\pdl_project_res\\"

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gamma_correction(img):
    gray = rgb2gray(img)

    # compute gamma = log(mid*255)/log(mean)
    mid = 0.5
    mean = np.mean(gray)
    gamma = math.log(mid)/math.log(mean)

    img_gamma = exposure.adjust_gamma(img, gamma)
    return img_gamma

for filename in os.listdir(PATH):
    with open(PATH + filename, "rb") as img_file:
        img = plt.imread(img_file)
    img_gamma = gamma_correction(img)
    plt.imsave(SAVE_PATH + filename[:-4] + "gamma.png", img_gamma)
