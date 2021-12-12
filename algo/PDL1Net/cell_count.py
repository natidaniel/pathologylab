import cv2
import numpy as np
import copy
import algo.mrcnn.visualize_pdl1 as vis_pdl1

class_names = {"INFLAMMATION": 1, "NEGATIVE": 2, "POSITIVE": 3, "OTHER": 4}


def gamma_correction(img, gammas):
    """
    apply gamma correction on the given image.
    allow different gamma for each color channel
    :param img: image in BGR color format
    :param gammas: array of gamma to use for each channel (in RGB order)
    :return: corrected image
    """
    # assume the format of the image is BGR, but the gammas are in RGB
    img[:, :, 0] = (((img[:, :, 0] / 255) ** gammas[2]) * 255)
    img[:, :, 1] = (((img[:, :, 1] / 255) ** gammas[1]) * 255)
    img[:, :, 2] = (((img[:, :, 2] / 255) ** gammas[0]) * 255)
    return img


def hue_nuclei_masking(img, min_hue, max_hue):
    """
    mask the image's nuclei by hue limits
    :param img: the image to apply thr masking to
    :param min_hue: the minimum hue to consider as nuclei
    :param max_hue: the maximum hue to consider as nuclei
    :return: mask of the filtered image
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_RGB2HSV)

    hue_mask = np.logical_and(min_hue < hsv_img[:, :, 0], hsv_img[:, :, 0] < max_hue)
    return hue_mask


def morphological_correction(mask, kernel_size=4):
    # create the negative of the mask
    negative_mask = np.ones(mask.shape)
    negative_mask = negative_mask - mask
    # close operation
    kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
    negative_mask = cv2.morphologyEx(negative_mask.astype('uint8'), cv2.MORPH_CLOSE, kernel_close)
    # return to the non-negative mask
    mask = np.ones(negative_mask.shape)
    mask = mask - negative_mask
    return mask


def morphological_correction_big(mask, kernel_size=5):
    # close operation
    kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_CLOSE, kernel_close)

    return mask


def find_contours(mask):
    # uncomment this line and comment the line after if using later version of openCV
    #_, cnts, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    single_pixel = []
    for cnt in cnts:
        if cnt.shape[0] <= 1:
            single_pixel.append(cnt)
    return cnts, single_pixel


def count_nucleus(img, img_class, img_masks, img_class_ids):
    working_img = copy.deepcopy(img)
    gammas = []
    if img_class == class_names["POSITIVE"]:
        gammas = [2, 1.6, 1.3] # RGB
        mask = vis_pdl1.get_class_mask(3, img_masks, img_class_ids)
    elif img_class == class_names["NEGATIVE"]:
        gammas = [2.2, 1.6, 1.3] # RGB
        mask = vis_pdl1.get_class_mask(2, img_masks, img_class_ids)
    else:
        mask = np.zeros((1024, 1024))
    # create the 3d mask
    temp_mask = np.broadcast_to(mask, (3, mask.shape[0], mask.shape[1]))
    mask_3d = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(3):
        mask_3d[:, :, i] = temp_mask[i, :, :]
    # apply the mask
    working_img = working_img * mask_3d
    working_img = working_img.astype('uint8')

    working_img = gamma_correction(working_img, gammas)

    if img_class == class_names["POSITIVE"]:
        hue_min = 70
        hue_max = 175
    elif img_class == class_names["NEGATIVE"]:
        hue_min = 50
        hue_max = 175
    else:
        hue_min = 70
        hue_max = 175
    mask = hue_nuclei_masking(working_img, hue_min, hue_max)

    if img_class == class_names["POSITIVE"]:
        kernel_size = 4
    elif img_class == class_names["NEGATIVE"]:
        kernel_size = 4
    mask_after_morph = morphological_correction(mask, kernel_size)
    cnts, single_pixel = find_contours(mask_after_morph)
    if len(cnts) > 40: # on high number of cells - do not use morphological operation
        if img_class == class_names["POSITIVE"]:
            kernel_size = 5
        elif img_class == class_names["NEGATIVE"]:
            kernel_size = 5
        mask_after_morph = morphological_correction_big(mask, kernel_size)
        cnts, single_pixel = find_contours(mask_after_morph)
    return len(cnts) - len(single_pixel), mask_after_morph
