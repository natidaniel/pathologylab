import numpy as np
import skimage
import datetime
import matplotlib.pyplot as plt
import os
import pickle
import gc

a = os.getcwd()
import algo.mrcnn.visualize_pdl1 as vis_pdl1
import algo.mrcnn.utils as utils
import algo.mrcnn.model as modellib
from datautils.PDL1NetDataLoader import PDL1NetDataset
import params.PDL1NetConfig as config
from algo.PDL1Net.cell_count import count_nucleus
import math

class PDL1NetTester:
    """
    class represents a PDL1 net Tester
    """

    def __init__(self, model, args):
        self.model = model
        self.args = args

    def test(self, images):
        if images is None:
            raise NameError("None was sent, but list of images is expected")
        # if only one image was sent wrap it in list
        if not isinstance(images, list):
            images = [images]
        return self.model.detect(images, verbose=1)[0]

    def color_splash(self, image, mask):
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]

        Returns result image.
        """
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        # Copy color pixels from the original color image where mask is set
        if mask.shape[0] > 0:
            splash = np.where(mask, image, gray).astype(np.uint8)
        else:
            splash = gray
        return splash


    def detect_and_show_mask(self, image_path):
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(self.args.image)
        # Detect objects
        r = self.model.detect([image], verbose=1)[0]
        # Color splash
        vis_pdl1.imwrite_mask(image, r['masks'], r['class_ids'], savename="splash_{}".format(os.path.split(self.args.image)[1]))
        # Save output_IoU0_C1_BG1
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # skimage.io.imsave(file_name, splash)

        # print("Saved to ", file_name)

    def detect_and_color_splash(self, image_path=None):
        """
        Detects the segments in the image and plot the image using color_splash
        """
        assert image_path, "image path is missing"
        # Run model detection and generate the color splash effect
        print("Running on {}".format(self.args.image))
        # Read image
        image = skimage.io.imread(self.args.image)
        # Detect objects
        r = self.test(image)
        # Color splash
        splash = self.color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        print("Saved to ", file_name)

    def test_sequence(self, show_image=True, sample=1, result_dir_name=None, real_slide=False):
        """
        Tests the model on the test dataset
        calculate and plots the Confusion matrix
        also calculate and plots the score of the area of the pdl1+ / (pdl1+ + pdl1-)
        note: class ids:
        0 - background (not ROI), 1 - inflammation, 2 - negative, 3 - positive, 4 - other
        :param show_image: if true plots the masked images from train
        :param sample: the frequency of images to show ( 1 each image, 2 every second image, etc. )
        saves the data into output folder
        """

        metric_data = {}
        if result_dir_name is not None:
            if os.path.exists(vis_pdl1.result_dir):
                # os.remove(vis_pdl1.result_dir)
                pass
            path, dir_name = os.path.split(vis_pdl1.result_dir)
            new_path = os.path.join(path, result_dir_name)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            vis_pdl1.result_dir = new_path
            metric_data = pickle.load(open(os.path.join(new_path, "metric_data.pickle"), "rb"))
        output_file = os.path.join(vis_pdl1.result_dir, "metric_data.pickle")

        # configure the Config Object given to the model
        class InferenceConfig(config.PDL1NetConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        # load test dataset using args.dataset as to the main as argument
        dataset_val = PDL1NetDataset()
        dataset_val.load_pdl1net_dataset(self.args.dataset, "val", real=self.args.real)

        dataset_val.prepare()

        print("start test")
        if hasattr(self.args, "config"):
            inference_config = self.args.config
        else:
            inference_config = InferenceConfig()
        matched_classes = []

        if result_dir_name is None:
            confusstion_matrix = np.zeros((dataset_val.num_classes, dataset_val.num_classes))
            confusion_matrix_by_pixel = np.zeros((3, 3))
            confusion_matrix_by_pixel_air_filt = np.zeros((3, 3))
            areas = np.array([0., 0., 0., 0.])
            areas_air_filt = np.array([0., 0., 0., 0.])
            gt_areas = np.array([0., 0., 0.])
            gt_areas_air_filt = np.array([0., 0., 0.])
            score_accuracy = []
            IoUs, IoU_classes = ([[] for _ in range(5)], [])
            IoU_per_image = {"NEGATIVE": {}, "POSITIVE": {}, "OTHER": {}, "ALL": {}}
            IoU_per_image_air_filt = {"NEGATIVE": {}, "POSITIVE": {}, "OTHER": {}, "ALL": {}}
            total_intersections = {"NEGATIVE": 0, "POSITIVE": 0, "OTHER": 0, "ALL": 0}
            total_unions = {"NEGATIVE": 0, "POSITIVE": 0, "OTHER": 0, "ALL": 0}
            total_intersections_air_filt = {"NEGATIVE": 0, "POSITIVE": 0, "OTHER": 0, "ALL": 0}
            total_unions_air_filt = {"NEGATIVE": 0, "POSITIVE": 0, "OTHER": 0, "ALL": 0}
            gt_tumor_area_per_image = {}
            gt_tumor_area_per_image_air_filt = {}
            accuracy_per_image = {}
            areas_per_image = {}
            accuracy_per_image_air_filt = {}
            areas_per_image_air_filt = {}
            cell_count_per_image = {}
            cell_count_per_image_gt = {}
            evaluated_images = []
        else:
            evaluated_images = metric_data["evaluated_images"]
            accuracy_per_image = metric_data["accuracy_per_image"]
            areas_per_image = metric_data["areas_per_image"]
            gt_tumor_area_per_image = metric_data["gt_tumor_area_per_image"]
            accuracy_per_image_air_filt = metric_data["accuracy_per_image_air_filt"]
            areas_per_image_air_filt = metric_data["areas_per_image_air_filt"]
            gt_tumor_area_per_image_air_filt = metric_data["gt_tumor_area_per_image_air_filt"]
            confusstion_matrix = metric_data["confusstion_matrix"]
            confusion_matrix_by_pixel = metric_data["confusion_matrix_by_pixel"]
            confusion_matrix_by_pixel_air_filt = metric_data["confusion_matrix_by_pixel_air_filt"]
            areas = metric_data["areas"]
            areas_air_filt = metric_data["areas_air_filt"]
            gt_areas = metric_data["gt_areas"]
            gt_areas_air_filt = metric_data["gt_areas_air_filt"]
            score_accuracy = metric_data["score_accuracy"]
            cell_count_per_image = metric_data["cell_count_per_image"]
            cell_count_per_image_gt = metric_data["cell_count_per_image_gt"]
            IoU_classes = metric_data["IoU_classes"]
            IoUs = metric_data["IoUs"]
            IoU_per_image = metric_data["IoU_per_image"]
            IoU_per_image_air_filt = metric_data["IoU_per_image_air_filt"]
            total_intersections = metric_data["total_intersections"]
            total_unions = metric_data["total_unions"]
            total_intersections_air_filt = metric_data["total_intersections_air_filt"]
            total_unions_air_filt = metric_data["total_unions_air_filt"]

        # iterate over all the data and
        for image_id in np.arange(dataset_val.num_images):
            image_name = dataset_val.image_info[image_id]["id"]
            if image_name in evaluated_images:
                print("skipping image " + str(image_name))
                continue
            # Load image and ground truth data
            image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
                modellib.load_image_gt(dataset_val, inference_config,
                                       image_id, use_mini_mask=False)
            print(image_name)

            # plot the backbone activation layer as performed on the current image
            # if hasattr(self.args, "backbone"):
            #     vis_pdl1.inspect_backbone_activation(self.model, image,
            #                                          savename="{}_backbone".format(image_name), args=self.args)
            # else:
            #     vis_pdl1.inspect_backbone_activation(self.model, image,
            #                                          savename="{}_backbone".format(image_name))
            plt.close('all')

            # Run object detection
            try:
                results = self.model.detect([image], verbose=0)
            except:
                print("got error, try again")
                results = self.model.detect([image], verbose=0)
            r = results[0]
            pred_masks_air_filt = utils.clean_air(image, r['masks'], r['class_ids'])
            gt_masks_air_filt = utils.clean_air(image, gt_masks, gt_class_ids)

            if show_image is True and image_id % sample == 0:
                # save prediction images
                # vis_pdl1.imwrite_mask(image, r['masks'], r['class_ids'],
                #                       savename="{}".format(image_name), saveoriginal=False)
                vis_pdl1.imwrite_mask(image, pred_masks_air_filt, r['class_ids'],
                                      savename="{}_air_filt".format(image_name), saveoriginal=False)
                # save ground truth images
                # vis_pdl1.imwrite_mask(image, gt_masks, gt_class_ids,
                #                       savename="{}_gt".format(image_name))
                if not real_slide:
                    vis_pdl1.imwrite_mask(image, gt_masks_air_filt, gt_class_ids,
                                          savename="{}_gt_air_filt".format(image_name))
                    # print("saving rois boxes")
                    vis_pdl1.imwrite_boxes(image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
                                           r['scores'], savename="{}".format(image_name))
                # save image of the gt of each class separately
                # vis_pdl1.imwrite_class(image, gt_masks, gt_class_ids, 2,
                #                        savename="{}_negative".format(image_name))
                # vis_pdl1.imwrite_class(image, gt_masks, gt_class_ids, 3,
                #                        savename="{}_positive".format(image_name))

            gt_match, pred_match, overlaps = utils.compute_matches(gt_bboxes, gt_class_ids, gt_masks,
                                                                   r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                                   iou_threshold=0.5, score_threshold=0.0)

            # calculate our accuracy metric
            img_accuracy, img_gt_area = utils.compute_detection_accuracy(gt_class_ids, gt_masks, r["class_ids"],
                                                                         r["scores"], r['masks'], threshold=0)
            accuracy_per_image[image_name] = img_accuracy
            gt_tumor_area_per_image[image_name] = img_gt_area

            img_accuracy_air_filt, img_gt_area_air_filt = \
                utils.compute_detection_accuracy(gt_class_ids, gt_masks_air_filt, r["class_ids"], r["scores"],
                                                 pred_masks_air_filt, threshold=0)
            accuracy_per_image_air_filt[image_name] = img_accuracy_air_filt
            gt_tumor_area_per_image_air_filt[image_name] = img_gt_area_air_filt

            # calculates the IoU over the images and segments
            IoUs_image, IoU_classes_image = vis_pdl1.get_IoU_from_matches(pred_match, r["class_ids"], overlaps)
            for i in range(len(IoUs_image)):
                IoUs[i] += IoUs_image[i]
            IoU_classes += [IoU_classes_image]

            # calculate IoU in pixel level (for classes - negative (2), positive (3), other (0))
            IoU_per_image["NEGATIVE"][image_name], inter_neg, union_neg =\
                utils.compute_image_iou(r['masks'], r["class_ids"], gt_masks, gt_class_ids, [2])
            IoU_per_image["POSITIVE"][image_name], inter_pos, union_pos =\
                utils.compute_image_iou(r['masks'], r["class_ids"], gt_masks, gt_class_ids, [3])
            IoU_per_image["OTHER"][image_name], inter_other, union_other = \
                utils.compute_image_iou(r['masks'], r["class_ids"], gt_masks, gt_class_ids, [0])
            IoU_per_image["ALL"][image_name], inter_all, union_all = \
                utils.compute_image_iou(r['masks'], r["class_ids"], gt_masks, gt_class_ids, [2, 3, 0])
            total_intersections["NEGATIVE"] += inter_neg
            total_unions["NEGATIVE"] += union_neg
            total_intersections["POSITIVE"] += inter_pos
            total_unions["POSITIVE"] += union_pos
            total_intersections["OTHER"] += inter_other
            total_unions["OTHER"] += union_other
            total_intersections["ALL"] += inter_all
            total_unions["ALL"] += union_all
            IoU_per_image_air_filt["NEGATIVE"][image_name], inter_neg_air_filt, union_neg_air_filt = \
                utils.compute_image_iou(pred_masks_air_filt, r["class_ids"], gt_masks_air_filt, gt_class_ids, [2])
            IoU_per_image_air_filt["POSITIVE"][image_name], inter_pos_air_filt, union_pos_air_filt = \
                utils.compute_image_iou(pred_masks_air_filt, r["class_ids"], gt_masks_air_filt, gt_class_ids, [3])
            IoU_per_image_air_filt["OTHER"][image_name], inter_other_air_filt, union_other_air_filt = \
                utils.compute_image_iou(pred_masks_air_filt, r["class_ids"], gt_masks_air_filt, gt_class_ids, [0])
            IoU_per_image_air_filt["ALL"][image_name], inter_all_air_filt, union_all_air_filt = \
                utils.compute_image_iou(pred_masks_air_filt, r["class_ids"], gt_masks_air_filt, gt_class_ids, [2, 3, 0])
            total_intersections_air_filt["NEGATIVE"] += inter_neg_air_filt
            total_unions_air_filt["NEGATIVE"] += union_neg_air_filt
            total_intersections_air_filt["POSITIVE"] += inter_pos_air_filt
            total_unions_air_filt["POSITIVE"] += union_pos_air_filt
            total_intersections_air_filt["OTHER"] += inter_other_air_filt
            total_unions_air_filt["OTHER"] += union_other_air_filt
            total_intersections_air_filt["ALL"] += inter_all_air_filt
            total_unions_air_filt["ALL"] += union_all_air_filt

            # confusion matrix
            confusstion_matrix += vis_pdl1.get_confusion_matrix(4, gt_class_ids, r["class_ids"], r["scores"],
                                                                      overlaps, [], threshold=0.5)
            confusion_matrix_by_pixel_temp, gt_areas_temp = \
                vis_pdl1.get_confusion_matrix_pixel_level(gt_masks, r['masks'], gt_class_ids, r['class_ids'])
            confusion_matrix_by_pixel_air_filt_temp, gt_areas_air_filt_temp = \
                vis_pdl1.get_confusion_matrix_pixel_level(gt_masks_air_filt, pred_masks_air_filt, gt_class_ids,
                                                          r['class_ids'])
            confusion_matrix_by_pixel += confusion_matrix_by_pixel_temp
            confusion_matrix_by_pixel_air_filt += confusion_matrix_by_pixel_air_filt_temp
            gt_areas += gt_areas_temp
            gt_areas_air_filt += gt_areas_air_filt_temp

            # calculate the areas of each class for prediction and gt
            temp_areas = vis_pdl1.get_image_areas(gt_masks, gt_class_ids, r['masks'], r['class_ids'])
            temp_areas_air_filt = \
                vis_pdl1.get_image_areas(gt_masks_air_filt, gt_class_ids, pred_masks_air_filt, r['class_ids'])
            areas += temp_areas
            areas_air_filt += temp_areas_air_filt
            areas_per_image[image_name] = temp_areas
            areas_per_image_air_filt[image_name] = temp_areas_air_filt

            # calculate the number of cells for each class for prediction (negative (2), positive (3), other (4))
            cell_count_positive, _ = count_nucleus(image, 3, r['masks'], r['class_ids'])
            cell_count_negative, _ = count_nucleus(image, 2, r['masks'], r['class_ids'])
            cell_count_per_image[image_name] = {"POSITIVE": cell_count_positive,
                                                "NEGATIVE": cell_count_negative}
            cell_count_positive_gt, _ = count_nucleus(image, 3, gt_masks, gt_class_ids)
            cell_count_negative_gt, _ = count_nucleus(image, 2, gt_masks, gt_class_ids)
            cell_count_per_image_gt[image_name] = {"POSITIVE": cell_count_positive_gt,
                                                   "NEGATIVE": cell_count_negative_gt}

            #  obtain all the elemnts in pred which have corresponding GT elemnt
            pred_match_exist = pred_match > -1
            #  retrieve the index of the GT element at the position of the correlated element in prediction
            # sort_gt_as_pred = pred_match[pred_match_exist].astype(int)
            matched_classes.append(r["class_ids"][pred_match_exist])

            score = vis_pdl1.score_almost_metric(gt_masks, gt_class_ids, r['masks'], r['class_ids'])
            if not math.isnan(score):
                score_accuracy += [score]
            evaluated_images.append(image_name)
            metric_data["evaluated_images"] = evaluated_images
            metric_data["accuracy_per_image"] = accuracy_per_image
            metric_data["areas_per_image"] = areas_per_image
            metric_data["gt_tumor_area_per_image"] = gt_tumor_area_per_image
            metric_data["accuracy_per_image_air_filt"] = accuracy_per_image_air_filt
            metric_data["areas_per_image_air_filt"] = areas_per_image_air_filt
            metric_data["gt_tumor_area_per_image_air_filt"] = gt_tumor_area_per_image_air_filt
            metric_data["confusstion_matrix"] = confusstion_matrix
            metric_data["confusion_matrix_by_pixel"] = confusion_matrix_by_pixel
            metric_data["confusion_matrix_by_pixel_air_filt"] = confusion_matrix_by_pixel_air_filt
            metric_data["areas"] = areas
            metric_data["areas_air_filt"] = areas_air_filt
            metric_data["gt_areas"] = gt_areas
            metric_data["gt_areas_air_filt"] = gt_areas_air_filt
            metric_data["score_accuracy"] = score_accuracy
            metric_data["cell_count_per_image"] = cell_count_per_image
            metric_data["cell_count_per_image_gt"] = cell_count_per_image_gt
            metric_data["IoU_classes"] = IoU_classes
            metric_data["IoUs"] = IoUs
            metric_data["IoU_per_image"] = IoU_per_image
            metric_data["IoU_per_image_air_filt"] = IoU_per_image_air_filt
            metric_data["total_intersections"] = total_intersections
            metric_data["total_unions"] = total_unions
            metric_data["total_intersections_air_filt"] = total_intersections_air_filt
            metric_data["total_unions_air_filt"] = total_unions_air_filt
            with open(output_file, 'wb') as out_file:
                pickle.dump(metric_data, out_file)

        # release memory
        del image
        gc.collect()

        # save all the test results to a file
        print("result dir: " + str(vis_pdl1.result_dir))
        file_path = os.path.join(vis_pdl1.result_dir, "out.txt")
        with open(file_path, "w") as file:
            mean_IoU_per_image_per_class = np.zeros((5, 1))
            IoU_classes = np.stack(IoU_classes).reshape(-1, 5)
            for i in range(IoU_classes.shape[1]):
                if not any(IoU_classes[:, i] != 0):
                    continue
                mean_IoU_per_image_per_class[i] = IoU_classes[IoU_classes[:, i] != 0, i].mean()
            # mean_IoU_per_image_per_seg = np.mean(IoU_classes, axis=0)
            mean_IoUs_per_seg = np.zeros((len(IoUs), 1))
            for i in range(len(IoUs)):
                if not IoUs[i]:
                    continue
                IoUs[i] = np.array(IoUs[i])
                mean_IoUs_per_seg[i] = np.mean(IoUs[i])
            file.write('accuracy:\n{}\n'.format(score_accuracy))
            file.write("IoU over segments is \n{}\n".format(mean_IoUs_per_seg))
            file.write("IoU over images is \n{}\n".format(mean_IoU_per_image_per_class))

            file.write("the confusion matrix is:\n {}\n".format(confusstion_matrix))
            file.write("the confusion matrix by pixel is:\n {}\n".format(confusion_matrix_by_pixel))
            file.write("the confusion matrix by pixel air filtered is:\n {}\n".format(confusion_matrix_by_pixel_air_filt))

            file.write("\ncustom accuracy data:\n")
            custom_accuracy = 0
            total_gt_area = 0
            custom_accuracy_air_filt = 0
            total_gt_area_air_filt = 0
            for key in accuracy_per_image.keys():
                custom_accuracy += accuracy_per_image[key] * gt_tumor_area_per_image[key]
                total_gt_area += gt_tumor_area_per_image[key]
                file.write("custom accuracy for image {} is {} with gt area {}.  "
                           "IoU: {}, negative IoU: {}, positive IoU: {}, other IoU: {}\n".
                           format(key, accuracy_per_image[key], gt_tumor_area_per_image[key],
                                  IoU_per_image["ALL"][key], IoU_per_image["NEGATIVE"][key],
                                  IoU_per_image["POSITIVE"][key], IoU_per_image["OTHER"][key]))
            file.write("air filtered accuracies:\n")
            for key in accuracy_per_image_air_filt.keys():
                custom_accuracy_air_filt += accuracy_per_image_air_filt[key] * gt_tumor_area_per_image_air_filt[key]
                total_gt_area_air_filt += gt_tumor_area_per_image_air_filt[key]
                file.write("custom accuracy for image {} is {} with gt area {}.  "
                           "IoU: {}, negative IoU: {}, positive IoU: {}, other IoU: {}\n".
                           format(key, accuracy_per_image_air_filt[key], gt_tumor_area_per_image_air_filt[key],
                                  IoU_per_image_air_filt["ALL"][key], IoU_per_image_air_filt["NEGATIVE"][key],
                                  IoU_per_image_air_filt["POSITIVE"][key], IoU_per_image_air_filt["OTHER"][key]))

            weighted_avg_accuracy = custom_accuracy / total_gt_area
            file.write("total accuracy (average was weighted by gt area): {}\n".format(weighted_avg_accuracy))
            weighted_avg_accuracy_air_filt = custom_accuracy_air_filt / total_gt_area_air_filt
            file.write("total accuracy for air filtered (average was weighted by gt area): {}\n".format(weighted_avg_accuracy_air_filt))
            file.write("average IoU per image: All classes: {}, Negative: {}, Positive: {}, Other: {}\n".format(
                np.mean(list(IoU_per_image["ALL"].values())), np.mean(list(IoU_per_image["NEGATIVE"].values())),
                np.mean(list(IoU_per_image["POSITIVE"].values())), np.mean(list(IoU_per_image["OTHER"].values())
                                                                           )))
            file.write("average IoU per image air filtered: All classes: {}, Negative: {}, Positive: {}, Other: {}\n".format(
                np.mean(list(IoU_per_image_air_filt["ALL"].values())), np.mean(list(IoU_per_image_air_filt["NEGATIVE"].values())),
                np.mean(list(IoU_per_image_air_filt["POSITIVE"].values())), np.mean(list(IoU_per_image_air_filt["OTHER"].values())
                                                                                    )))
            file.write("total IoU: All classes: {}, Negative: {}, Positive: {}, Other: {}\n".format(
                total_intersections["ALL"] / total_unions["ALL"],
                total_intersections["NEGATIVE"] / total_unions["NEGATIVE"],
                total_intersections["POSITIVE"] / total_unions["POSITIVE"],
                total_intersections["OTHER"] / total_unions["OTHER"]))
            file.write("total IoU air filtered: All classes: {}, Negative: {}, Positive: {}, Other: {}\n".format(
                total_intersections_air_filt["ALL"] / total_unions_air_filt["ALL"],
                total_intersections_air_filt["NEGATIVE"] / total_unions_air_filt["NEGATIVE"],
                total_intersections_air_filt["POSITIVE"] / total_unions_air_filt["POSITIVE"],
                total_intersections_air_filt["OTHER"] / total_unions_air_filt["OTHER"]))

            total_correct_categories = 0
            categories_confusion_matrix_area = np.zeros((3, 3))
            category_to_int = {"LOW": 0, "MID": 1, "HIGH": 2}
            file.write("\n\nscore by area per image:\n")
            for key in areas_per_image.keys():
                pred_score, pred_category, gt_score, gt_category, correct_category = \
                    utils.compute_category_accuracy_by_area(areas_per_image[key][0], areas_per_image[key][1],
                                                            areas_per_image[key][2], areas_per_image[key][3])
                if correct_category:
                    total_correct_categories += 1
                categories_confusion_matrix_area[category_to_int[gt_category], category_to_int[pred_category]] += 1
                file.write("image {}: prediction score: {}, prediction category: {}, "
                           "gt score: {}, gt category: {}\n".format(key, pred_score, pred_category, gt_score, gt_category))
            file.write("correct categories: {}% ({} correct patches)\n".format(
                total_correct_categories / len(areas_per_image.keys()) * 100, total_correct_categories))

            total_correct_categories_air_filt = 0
            file.write("\n\nscore by area per image (air filtered):\n")
            for key in areas_per_image_air_filt.keys():
                pred_score, pred_category, gt_score, gt_category, correct_category = \
                    utils.compute_category_accuracy_by_area(areas_per_image_air_filt[key][0], areas_per_image_air_filt[key][1],
                                                            areas_per_image_air_filt[key][2], areas_per_image_air_filt[key][3])
                if correct_category:
                    total_correct_categories_air_filt += 1
                file.write("image {}: prediction score: {}, prediction category: {}, "
                           "gt score: {}, gt category: {}\n".format(key, pred_score, pred_category, gt_score,
                                                                    gt_category))
            file.write("correct categories air filtered: {}% ({} correct patches)\n".format(
                total_correct_categories_air_filt / len(areas_per_image.keys()) * 100, total_correct_categories_air_filt))

            # categories and score by cell count
            total_positive_cell_count_pred = 0
            total_negative_cell_count_pred = 0
            total_positive_cell_count_gt = 0
            total_negative_cell_count_gt = 0
            total_correct_categories_cell_count = 0
            categories_confusion_matrix_cell_count = np.zeros((3, 3))
            file.write("\n\nscore by cell count per image:\n")
            for key in cell_count_per_image.keys():
                pred_score, pred_category, gt_score, gt_category, correct_category = \
                    utils.compute_category_accuracy_by_cells(cell_count_per_image[key]["POSITIVE"],
                                                             cell_count_per_image[key]["NEGATIVE"],
                                                             cell_count_per_image_gt[key]["POSITIVE"],
                                                             cell_count_per_image_gt[key]["NEGATIVE"],
                                                             areas_per_image[key][2],
                                                             areas_per_image[key][3])
                if correct_category:
                    total_correct_categories_cell_count += 1
                categories_confusion_matrix_cell_count[category_to_int[gt_category], category_to_int[pred_category]] += 1

                if cell_count_per_image_gt[key]["POSITIVE"] == 0:
                    positive_error_percentage = 0
                else:
                    positive_error_percentage = (np.abs(cell_count_per_image[key]["POSITIVE"] - cell_count_per_image_gt[
                        key]["POSITIVE"]) / cell_count_per_image_gt[key]["POSITIVE"]) * 100
                if cell_count_per_image_gt[key]["NEGATIVE"] == 0:
                    negative_error_percentage = 0
                else:
                    negative_error_percentage = (np.abs(cell_count_per_image[key]["NEGATIVE"] - cell_count_per_image_gt[
                        key]["NEGATIVE"]) / cell_count_per_image_gt[key]["NEGATIVE"]) * 100
                file.write("image {}: prediction score: {}, prediction category: {}, "
                           "gt score: {}, gt category: {}, "
                           "prediction positive cell count: {}, prediction negative cell count: {},"
                           "gt (algorithm run based on gt masking) positive cell count: {},"
                           "gt (algorithm run based on gt masking) negative cell count: {},"
                           "positive error percentage: {}%, negative error percentage: {}%"
                           "\n".format(key, pred_score, pred_category, gt_score, gt_category,
                                       cell_count_per_image[key]["POSITIVE"],
                                       cell_count_per_image[key]["NEGATIVE"],
                                       cell_count_per_image_gt[key]["POSITIVE"],
                                       cell_count_per_image_gt[key]["NEGATIVE"],
                                       positive_error_percentage,
                                       negative_error_percentage
                                       ))
                total_positive_cell_count_pred += cell_count_per_image[key]["POSITIVE"]
                total_negative_cell_count_pred += cell_count_per_image[key]["NEGATIVE"]
                total_positive_cell_count_gt += cell_count_per_image_gt[key]["POSITIVE"]
                total_negative_cell_count_gt += cell_count_per_image_gt[key]["NEGATIVE"]
            file.write("correct categories: {}% ({} correct patches)\n".format(
                total_correct_categories_cell_count / len(cell_count_per_image.keys()) * 100,
                total_correct_categories_cell_count))

            if areas[0] + areas[1] == 0:
                pred_score = 0
            else:
                pred_score = areas[0] / (areas[0] + areas[1])
            if areas[2] + areas[3] == 0:
                gt_score = 0
            else:
                gt_score = areas[2] / (areas[2] + areas[3])
            if areas_air_filt[0] + areas_air_filt[1] == 0:
                pred_score_air_filt = 0
            else:
                pred_score_air_filt = areas_air_filt[0] / (areas_air_filt[0] + areas_air_filt[1])
            if areas_air_filt[2] + areas_air_filt[3] == 0:
                gt_score_air_filt = 0
            else:
                gt_score_air_filt = areas_air_filt[2] / (areas_air_filt[2] + areas_air_filt[3])
            if total_positive_cell_count_pred + total_negative_cell_count_pred == 0:
                pred_score_cell_count = 0
            else:
                pred_score_cell_count = total_positive_cell_count_pred / (total_positive_cell_count_pred + total_negative_cell_count_pred)
            if total_positive_cell_count_gt + total_negative_cell_count_gt == 0:
                gt_score_cell_count = 0
            else:
                gt_score_cell_count = total_positive_cell_count_gt / (total_positive_cell_count_gt + total_negative_cell_count_gt)
            file.write("\ntotal predicted wsi score of all patches (by area): {}\n".format(pred_score))
            file.write("total ground truth wsi score of all patches (by area): {}\n".format(gt_score))
            file.write("total predicted wsi score of all patches (by area, air filtered): {}\n".format(pred_score_air_filt))
            file.write("total ground truth wsi score of all patches (by area, air filtered): {}\n".format(gt_score_air_filt))
            file.write("total predicted wsi score of all patches (by cell count): {}\n".format(pred_score_cell_count))
            file.write("total ground truth wsi score of all patches (by cell count): {}\n".format(gt_score_cell_count))

            if not real_slide:  # plot all the metrics only on test data (which has ground truth)
                vis_pdl1.plot_hist(score_accuracy, savename="area_diff_hist")
                # create new class list to replace the 'BG' with 'other'
                right_indices = [4, 2, 3]
                copy_class_names = [dataset_val.class_names[i] for i in right_indices]
                indices_no_inf = [0] + list(range(2, 4))
                confusstion_matrix = confusstion_matrix[indices_no_inf, :][:, indices_no_inf]
                confusstion_matrix = confusstion_matrix / np.sum(confusstion_matrix)
                vis_pdl1.plot_confusion_matrix(confusstion_matrix, copy_class_names, savename="confussion_matrix")
                # normalize confusion matrix
                # for col in range(3):
                #     confusion_matrix_by_pixel[:, col] = confusion_matrix_by_pixel[:, col] / gt_areas[col]
                confusion_matrix_by_pixel = confusion_matrix_by_pixel / np.sum(confusion_matrix_by_pixel, 0)
                vis_pdl1.plot_confusion_matrix(confusion_matrix_by_pixel,
                                               ["other", "pdl-negative", "pdl-positive"], savename="confusion_matrix")
                # for col in range(3):
                #     confusion_matrix_by_pixel_air_filt[:, col] = confusion_matrix_by_pixel_air_filt[:, col] / gt_areas_air_filt[col]
                confusion_matrix_by_pixel_air_filt = confusion_matrix_by_pixel_air_filt / np.sum(confusion_matrix_by_pixel_air_filt, 0)
                vis_pdl1.plot_confusion_matrix(confusion_matrix_by_pixel_air_filt,
                                               ["other", "pdl-negative", "pdl-positive"], savename="confusion_matrix_air_filtered")

                categories_confusion_matrix_area = categories_confusion_matrix_area.transpose()
                categories_confusion_matrix_area = categories_confusion_matrix_area / np.sum(
                    categories_confusion_matrix_area, 0)
                vis_pdl1.plot_confusion_matrix(categories_confusion_matrix_area,
                                               ["LOW", "MID", "HIGH"], savename="confusion_matrix_categories_area")

                categories_confusion_matrix_cell_count = categories_confusion_matrix_cell_count.transpose()
                categories_confusion_matrix_cell_count = categories_confusion_matrix_cell_count / np.sum(
                    categories_confusion_matrix_cell_count, 0)
                vis_pdl1.plot_confusion_matrix(categories_confusion_matrix_cell_count,
                                               ["LOW", "MID", "HIGH"], savename="confusion_matrix_categories_cell_count")
