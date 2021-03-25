
import math
import numpy as np
import mrcnn.utils as utils
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import mrcnn.visualize as vis
from cv2 import imwrite, cvtColor, COLOR_GRAY2RGB, COLOR_BGR2RGB, resize
import cv2


result_dir = os.path.join("output", "out_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now()))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# TODO: create function in Tester that compute the inputs for this function
def compute_batch_iou(num_classes, list_pred_masks, list_gt_masks, matched_classes):
    """
    This Function takes a list of masks (preds and gt) coming from several images and computes the average IoU,
    given the IoU of each image as an array with dim = num_classes+1.
    note: the average is taken among images, not among instances.

    :param num_classes: ndarray (1,num_images): how many classes exist in each image? (background doesn't count)
    :param list_pred_masks: list of the masks which were predicted during evaluation
    :param list_gt_masks: list of the ground truth masks
    :param matched_classes: list of ndarray contain the class per mask. The indices are corresponding between
                        the prediction to the GT.
    :return IoU_mean: a numpy.ndarray of shape [num_classes+1,1], with the average IoU per class.

    :assumption: the callee needs to sort the elements in the lists according to the output_IoU0_C1_BG1 of utils.compute_matches
    """
    IoU_mean = np.zeros((num_classes + 1 ,1))
    cnt_instances = np.zeros((num_classes + 1, 1))
    N = len(list_pred_masks)
    for i in np.arange(N):
        IoU_curr, existing_classes_curr = compute_mean_iou_per_image(num_classes, list_pred_masks[i], list_gt_masks[i], matched_classes[i])
        IoU_mean += IoU_curr
        cnt_instances += existing_classes_curr
    select_nonzero = cnt_instances != 0
    IoU_mean[select_nonzero] = IoU_mean[select_nonzero] / cnt_instances[select_nonzero]
    return IoU_mean

def compute_mean_iou_per_image(num_classes, pred_masks, gt_masks, matched_classes):
    '''
    This function computes the average IoU per image.
    each image can contain different amount and kind of mask's classes,
    the average is taken among each class separately.
    :param num_classes: ndarray (1,num_images): how many classes exist in each image? (background doesn't count)
    :param pred_masks: the image's predicted masks
    :param gt_masks: the image's ground truth masks
    :param matched_classes: ndarray it's indices has correspondence between the prediction to the GT.
    :return IoU_per_class: mean IoU over all instances from the same class. ndarray with shape[num_classes + 1, 1]
    :return class_exist: ndarray with shape[num_classes + 1, 1], which contain 1 at indces where the class has at least
                        one instance in the image
    '''
    # IoU: mat. IoU score of each possible pair of masks gt Vs preds
    IoU = utils.compute_overlaps_masks(pred_masks, gt_masks)
    diag_indices = np.arange(IoU.shape[0])  # ndarray. IoU of matched masks
    IoU_diag = IoU[diag_indices, diag_indices]
    IoU_per_class = np.zeros((num_classes+1, 1))
    cnt_classes = np.zeros((num_classes+1, 1))
    for i in np.arange(IoU_diag.shape[0]):
        IoU_per_class[matched_classes[i]] += IoU_diag[i]
        cnt_classes[matched_classes[i]] += 1
    select_nonzero = cnt_classes != 0
    IoU_per_class[select_nonzero] = IoU_per_class[select_nonzero] / cnt_classes[select_nonzero]
    class_exist = cnt_classes > 0
    return IoU_per_class, class_exist.astype(np.uint8)


import itertools
def get_confusion_matrix(num_classes, gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1))

    for i, j in itertools.product(range(overlaps.shape[0]), range(overlaps.shape[1])):
        if overlaps[i, j] > threshold:
            confusion_matrix[pred_class_ids[i], gt_class_ids[j]] += 1

    thresh = threshold
    if overlaps.shape[0] != 0 and overlaps.shape[1] != 0:
        max_iou_pred = np.max(overlaps, axis=1)
        select_smaller_thresh =  max_iou_pred < thresh
        class_pred = pred_class_ids[select_smaller_thresh]
        confusion_matrix[class_pred, np.zeros_like(class_pred)] += 1

        max_iou_gt = np.max(overlaps, axis=0)
        select_smaller_thresh = max_iou_gt < thresh
        class_gt = gt_class_ids[select_smaller_thresh]
        confusion_matrix[np.zeros_like(class_gt), class_gt] += 1

    return confusion_matrix


def get_class_mask(target_class_id, masks_lists, class_ids):
    """
    return the mask of specific class for the whole image from the list of all the
    roi masks
    :param target_class_id: the id of the class to compute the while mask for
    :param masks_lists: the list of all the masks to compute the image masks from
    :param class_ids: the class id for each roi in the masks list
    :return: mask (binary image) of the given class
    """
    mask = np.zeros((1024, 1024), dtype=bool)
    for ind, class_id in enumerate(class_ids):
        if class_id == target_class_id:
            mask = np.logical_or(mask, masks_lists[:, :, ind])
    return mask


def get_confusion_matrix_pixel_level(gt_masks, pred_masks, gt_class_ids, pred_class_ids):
    pdl_positive_mask_gt = get_class_mask(3, gt_masks, gt_class_ids)
    pdl_negative_mask_gt = get_class_mask(2, gt_masks, gt_class_ids)
    pdl_positive_mask_pred = get_class_mask(3, pred_masks, pred_class_ids)
    pdl_negative_mask_pred = get_class_mask(2, pred_masks, pred_class_ids)
    other_mask_gt = np.logical_not(np.logical_or(pdl_positive_mask_gt, pdl_negative_mask_gt))
    other_mask_pred = np.logical_not(np.logical_or(pdl_positive_mask_pred, pdl_negative_mask_pred))
    # order in matrix: other, negative, positive
    confusion_matrix = np.zeros((3, 3))
    gt = [other_mask_gt, pdl_negative_mask_gt, pdl_positive_mask_gt]
    pred = [other_mask_pred, pdl_negative_mask_pred, pdl_positive_mask_pred]
    gt_areas = np.zeros(3)
    for i, gt_mask in enumerate(gt):
        gt_areas[i] = np.sum(gt_mask)
        for j, pred_mask in enumerate(pred):
            confusion_matrix[j, i] = np.sum(np.logical_and(gt_mask, pred_mask))

    return confusion_matrix, gt_areas


def plot_confusion_matrix(confusion_matrix, class_names, threshold=0.5, savename=None):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    confusion_matrix: [pred_boxes, gt_boxes] IoU confusion_matrix of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    fontsize = 40
    fig = plt.figure(figsize=(14, 14))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(class_names)), class_names, rotation=45, fontsize=int(fontsize * 0.8))
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, fontsize=int(fontsize * 0.8))

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]),
                                  range(confusion_matrix.shape[1])):
        color = ("white" if confusion_matrix[i, j] > thresh
                 else "black" if confusion_matrix[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:4.2f}%".format(100 * confusion_matrix[i, j]),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=fontsize, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth", fontsize=fontsize)
    plt.ylabel("Predictions", fontsize=fontsize)
    plt.title("Confusion Matrix", fontsize=int(fontsize * 1.4))
    plt.tight_layout()
    if savename is not None:
        name = os.path.join(result_dir, savename+".png")
        plt.savefig(name)
    # plt.show()
    return fig


def get_IoU_from_matches(match_pred2gt, matched_classes, ovelaps):
    """
    if given an image, claculate the IoU of the segments in the image
    :param match_pred2gt: maps index of predicted segment to index of ground truth segment
    :param matched_classes: maps index of predicted segment to class number
    :param ovelaps: maps [predicted segment index, gt segment index] to the IoU value of the segments
    :return:
        1. IoUs - IoU for all segments
        2. IoUs_classes - mean IoU per class
    """
    IoUs = [ [] for _ in range(5) ]
    match_pred2gt = match_pred2gt.astype(np.int32)
    for pred, gt in enumerate(match_pred2gt):
        if gt < 0:
            continue
        IoUs[matched_classes[pred]].append(ovelaps[pred, gt])

    # mean segments's IoU according to classes
    IoUs_classes = np.zeros((5, 1))
    for class_idx, lst in enumerate(IoUs):
        if not lst:
            continue
        arr = np.array(lst)
        IoUs_classes[class_idx] = (np.mean(arr))

    return IoUs, IoUs_classes

def score_area(masks_positive, masks_negative):
    """
    calculate the area of the pdl1 positive out of all the cancerous cells,
    score = num` pixels pdl1 *pos* / (num` pixels pdl1 *pos* + num` pixels pdl1 *neg*)
    :param masks_positive: ndarray [N, H, W] N number of positive segments
    :param masks_negative: ndarray [M, H, W] M number of negative segments
    :return: score = num` pixels pdl1 *pos* / (num` pixels pdl1 *pos* + num` pixels pdl1 *neg*)
    """
    positive_area = np.sum((masks_positive > 0).ravel())
    negative_area = np.sum((masks_negative > 0).ravel())
    if positive_area == 0 and negative_area == 0:
        return math.nan
    score = positive_area / (positive_area + negative_area)
    return score

def score_almost_metric(gt_masks, gt_classes, pred_masks, pred_classes):
    """
    Calculate the score of the ground-truth image and the prediction image using `score_area` function.
    The function then compute the difference between their scores and returns it.
    :param gt_masks: masks as resulted from model.load_image_gt function
    :param gt_classes: classes as resulted from model.load_image_gt function
    :param pred_masks: masks as resulted from PDL1NetTester.test function
    :param pred_classes: classes as resulted from PDL1NetTester.test function
    :return: difference between ground-truth score and prediction score
    """
    gt_positive_masks = gt_masks[..., gt_classes == 3]
    gt_negative_masks = gt_masks[..., gt_classes == 2]
    score_gt = score_area(gt_positive_masks, gt_negative_masks)
    pred_positive_masks = pred_masks[..., pred_classes == 3]
    pred_negative_masks = pred_masks[..., pred_classes == 2]
    score_pred = score_area(pred_positive_masks, pred_negative_masks)
    if math.isnan(score_gt):
        diff = score_pred
    elif math.isnan(score_pred):
        diff = -score_gt
    elif math.isnan(score_gt) and math.isnan(score_pred):
        diff = math.nan
    else:
        diff = score_pred - score_gt
    return diff

def remove_black_frame(image):
    """
    crops black boundaries from the image if there are any
    using the smallest BBox that contains all the none black pixels
    :param image: the image to remove the boundaries from
    :return: cropped image
    """
    if len(image.shape) > 2:
        image_gray = cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    _, mask = cv2.threshold(image_gray, 1.0, 255.0, cv2.THRESH_BINARY)

    # findContours destroys input
    if len(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)) == 2:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours by largest first (if there are more than one)
    contours = sorted(contours, key=lambda contour: len(contour), reverse=True)
    roi = cv2.boundingRect(contours[0])

    # use the roi to select into the original 'stitched' image
    if len(image.shape) > 2:
        return image[roi[1]:roi[3], roi[0]:roi[2], :]
    else:
        return image[roi[1]:roi[3], roi[0]:roi[2]]


def imwrite_mask(image, masks, classes, savename, remove_inflamation=False,  saveoriginal=False):
    """
    saves an image with colored segments s.t blue is pdl1+ red is pdl1- and green is inflammation.
    :param image: the image to be colored
    :param masks: masks as resulted from PDL1NetTester.test function
    :param classes: classes as resulted from PDL1NetTester.test function
    :param savename: save the image in the output folder using the given savename str as file name
    :param remove_inflamation: if true do not color inflammation segments.
    :param saveoriginal: if True save also the image itself unmodified to the output folder
    """
    if len(image.shape) < 3:
        image = cvtColor(image, COLOR_GRAY2RGB)
    if any(classes):  # if classes in not empty list
        if remove_inflamation:
            inflamation_num = 1
            classes[classes == inflamation_num] = 0
        classes = classes.reshape(1, 1, -1)
        # for i in range(len(classes)):
        #     mask[masks[:, :, i] is True] = (masks[:, :, i] * classes[i])[masks[:,:,i] is True]
        masks = masks * classes
        mask = np.max(masks, axis=2)
        class_to_color = {1: (0, 1., 0), 2: (1., 0, 0), 3: (0, 0, 1.)}
        edited_image = image.copy()
        for class_ in np.unique(classes.ravel()):
            edited_image = vis.apply_mask(edited_image, mask, class_to_color[class_], label=class_, alpha=0.5)
        edited_image = remove_black_frame(edited_image)
        if savename is not None:
            file_name = os.path.join(result_dir, "mask_"+savename+".png")
            edited_image = resize(edited_image, masks.shape[:2])
            edited_image = cvtColor(edited_image, COLOR_BGR2RGB)
            imwrite(file_name, edited_image)
    else:
        if savename is not None:
            file_name = os.path.join(result_dir, "mask_"+savename+".png")
            image_org = remove_black_frame(image)
            image_org = resize(image_org, masks.shape[:2])
            image_org = cvtColor(image_org, COLOR_BGR2RGB)
            imwrite(file_name, image_org)
    if saveoriginal:
        if savename is not None:
            file_name = os.path.join(result_dir, "org_" + savename + ".png")
            image_org = remove_black_frame(image)
            image_org = resize(image_org, masks.shape[:2])
            image_org = cvtColor(image_org, COLOR_BGR2RGB)
            imwrite(file_name, image_org)

def imwrite_boxes(image, boxes, masks, class_ids, class_names,
                      scores, savename):
    fig, ax = plt.subplots(1, figsize=(16,16))
    vis.display_instances(image, boxes, masks, class_ids, class_names, scores, ax=ax)
    if savename is not None:
        file_name = os.path.join(result_dir, "box_" + savename + ".png")
        fig.savefig(file_name)

def imwrite_class(image, masks, classes, target_class, savename):
    mask = get_class_mask(target_class, masks, classes)
    class_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(3):
        class_mask[...,i] = mask
    edited_image = image.copy()
    if savename is not None:
        file_name = os.path.join(result_dir, "class_" + savename + ".png")
        # edited_image = remove_black_frame(edited_image)
        edited_image = resize(edited_image, masks.shape[:2])
        edited_image = cvtColor(edited_image, COLOR_BGR2RGB)
        edited_image = edited_image * class_mask
        imwrite(file_name, edited_image)

def plot_hist(data, savename=None):
    """
    plot the data using histogram plot
    :param data: the data to be plotted
    :param savename: if not None saves the figure to the output folder with the given name
    """
    fig, axe = plt.subplots()
    axe.hist(data)
    if savename is not None:
        name = os.path.join(result_dir, savename+".png")
        plt.savefig(name)

    # plt.show()


def inspect_backbone_activation(model, image, savename=None, args=None):
    """
    insert the image through the model pipe and insect (using plt.imshow) its plot after each layer
    TODO: add a variable to choose the layer to inspect from
    :param model: the MRCNN model
    :param image: the image to inspect
    :param savename: if None does nothing, else save the image to the output folder with the given name
    :param args: if None does nothing, else expect object with 'backbone' attribute to choose the right pipe to follow
    """
    if args is not None:
        if args.backbone.lower() == "resnet50":
            activation_resnet50 = [
                ("input_image", tf.identity(model.keras_model.get_layer("input_image").output)),
                ("res2c_out", model.keras_model.get_layer("res2c_out").output),
                ("res3c_out", model.keras_model.get_layer("res3c_out").output),
                # ("res4w_out", model.keras_model.get_layer("res4w_out").output),  # for resnet100
                ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
                ("roi", model.keras_model.get_layer("ROI").output),
            ]
            activations = model.run_graph([image], activation_resnet50)
        elif args.backbone.lower() == "resnet101":
            activation_resnet101 = [
                ("input_image", tf.identity(model.keras_model.get_layer("input_image").output)),
                ("res2c_out", model.keras_model.get_layer("res2c_out").output),
                ("res3c_out", model.keras_model.get_layer("res3c_out").output),
                ("res4w_out", model.keras_model.get_layer("res4w_out").output),  # for resnet100
                ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
                ("roi", model.keras_model.get_layer("ROI").output),
            ]
            activations = model.run_graph([image], activation_resnet101)
        else:
            raise(NotImplemented, "{} is not implemented for inspect_backbone_activation".format(args.backbone))
    else:
        activation_resnet50 = [
            ("input_image", tf.identity(model.keras_model.get_layer("input_image").output)),
            ("res2c_out", model.keras_model.get_layer("res2c_out").output),
            ("res3c_out", model.keras_model.get_layer("res3c_out").output),
            # ("res4w_out", model.keras_model.get_layer("res4w_out").output),  # for resnet100
            ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
            ("roi", model.keras_model.get_layer("ROI").output),
        ]
        activations = model.run_graph([image], activation_resnet50)
    if savename is not None:
        savename = os.path.join(result_dir, savename + ".png")
    try:
        vis.display_images(np.transpose(activations["res2c_out"][0, :, :, :4], [2, 0, 1]),
                       cols=4, out_path=savename, show=False)
    except ...:
        pass
