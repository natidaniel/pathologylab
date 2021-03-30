import numpy as np
import skimage
import datetime
import matplotlib.pyplot as plt
import os
import pickle

a = os.getcwd()
import algo.mrcnn.visualize_pdl1 as vis_pdl1
import algo.mrcnn.utils as utils
import algo.mrcnn.model as modellib
from datautils.PDL1NetDataLoader import PDL1NetDataset
import params.PDL1NetConfig as config
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

    def test_sequence(self, show_image=True, sample=1, result_dir_name=None):
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
        dataset_val.load_pdl1net_dataset(self.args.dataset, "val")
        dataset_val.prepare()

        print("start test")
        if hasattr(self.args, "config"):
            inference_config = self.args.config
        else:
            inference_config = InferenceConfig()
        matched_classes = []

        if result_dir_name is None:
            confusstion_matrix = np.zeros((dataset_val.num_classes, dataset_val.num_classes))
            confusion_matrix_by_pixel = np.zeros((4, 4))
            confusion_matrix_by_pixel_air_filt = np.zeros((4, 4))
            areas = np.array([0., 0., 0., 0.])
            areas_air_filt = np.array([0., 0., 0., 0.])
            gt_areas = np.array([0., 0., 0., 0.])
            gt_areas_air_filt = np.array([0., 0., 0., 0.])
            score_accuracy = []
            IoUs, IoU_classes = ([[] for _ in range(5)], [])
            accuracy_per_image = {}
            gt_area_per_image = {}
            accuracy_per_image_air_filt = {}
            gt_area_per_image_air_filt = {}
            evaluated_images = []
        else:
            evaluated_images = metric_data["evaluated_images"]
            accuracy_per_image = metric_data["accuracy_per_image"]
            gt_area_per_image = metric_data["gt_area_per_image"]
            accuracy_per_image_air_filt = metric_data["accuracy_per_image_air_filt"]
            gt_area_per_image_air_filt = metric_data["gt_area_per_image_air_filt"]
            confusstion_matrix = metric_data["confusstion_matrix"]
            confusion_matrix_by_pixel = metric_data["confusion_matrix_by_pixel"]
            confusion_matrix_by_pixel_air_filt = metric_data["confusion_matrix_by_pixel_air_filt"]
            areas = metric_data["areas"]
            areas_air_filt = metric_data["areas_air_filt"]
            gt_areas = metric_data["gt_areas"]
            gt_areas_air_filt = metric_data["gt_areas_air_filt"]
            score_accuracy = metric_data["score_accuracy"]
            IoU_classes = metric_data["IoU_classes"]
            IoUs = metric_data["IoUs"]

        # iterate over all the data and
        for image_id in np.arange(dataset_val.num_images):
            if dataset_val.image_info[image_id]["id"] in evaluated_images:
                print("skipping image " + str(dataset_val.image_info[image_id]["id"]))
                continue
            # Load image and ground truth data
            image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
                modellib.load_image_gt(dataset_val, inference_config,
                                       image_id, use_mini_mask=False)
            print(dataset_val.image_info[image_id]["id"])

            # plot the backbone activation layer as performed on the current image
            # if hasattr(self.args, "backbone"):
            #     vis_pdl1.inspect_backbone_activation(self.model, image,
            #                                          savename="{}_backbone".format(dataset_val.image_info[image_id]["id"]), args=self.args)
            # else:
            #     vis_pdl1.inspect_backbone_activation(self.model, image,
            #                                          savename="{}_backbone".format(dataset_val.image_info[image_id]["id"]))
            plt.close('all')

            # Run object detection
            results = self.model.detect([image], verbose=0)
            r = results[0]
            pred_masks_air_filt = utils.clean_air(image, r['masks'], r['class_ids'])
            gt_masks_air_filt = utils.clean_air(image, gt_masks, gt_class_ids)

            if show_image is True and image_id % sample == 0:
                # save prediction images
                # vis_pdl1.imwrite_mask(image, r['masks'], r['class_ids'],
                #                       savename="{}".format(dataset_val.image_info[image_id]["id"]), saveoriginal=False)
                vis_pdl1.imwrite_mask(image, pred_masks_air_filt, r['class_ids'],
                                      savename="{}_air_filt".format(dataset_val.image_info[image_id]["id"]), saveoriginal=False)
                # save ground truth images
                # vis_pdl1.imwrite_mask(image, gt_masks, gt_class_ids,
                #                       savename="{}_gt".format(dataset_val.image_info[image_id]["id"]))
                vis_pdl1.imwrite_mask(image, gt_masks_air_filt, gt_class_ids,
                                      savename="{}_gt_air_filt".format(dataset_val.image_info[image_id]["id"]))
                # print("saving rois boxes")
                vis_pdl1.imwrite_boxes(image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
                                       r['scores'], savename="{}".format(dataset_val.image_info[image_id]["id"]))
                # save image of the gt of each class separately
                # vis_pdl1.imwrite_class(image, gt_masks, gt_class_ids, 2,
                #                        savename="{}_negative".format(dataset_val.image_info[image_id]["id"]))
                # vis_pdl1.imwrite_class(image, gt_masks, gt_class_ids, 3,
                #                        savename="{}_positive".format(dataset_val.image_info[image_id]["id"]))

            gt_match, pred_match, overlaps = utils.compute_matches(gt_bboxes, gt_class_ids, gt_masks,
                                                                   r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                                   iou_threshold=0.5, score_threshold=0.0)

            # calculate our accuracy metric
            img_accuracy, img_gt_area = utils.compute_detection_accuracy(gt_class_ids, gt_masks, r["class_ids"],
                                                                         r["scores"], r['masks'], threshold=0)
            accuracy_per_image[dataset_val.image_info[image_id]["id"]] = img_accuracy
            gt_area_per_image[dataset_val.image_info[image_id]["id"]] = img_gt_area

            img_accuracy_air_filt, img_gt_area_air_filt = \
                utils.compute_detection_accuracy(gt_class_ids, gt_masks_air_filt, r["class_ids"], r["scores"],
                                                 pred_masks_air_filt, threshold=0)
            accuracy_per_image_air_filt[dataset_val.image_info[image_id]["id"]] = img_accuracy_air_filt
            gt_area_per_image_air_filt[dataset_val.image_info[image_id]["id"]] = img_gt_area_air_filt

            # calculates the IoU over the images and segments
            IoUs_image, IoU_classes_image = vis_pdl1.get_IoU_from_matches(pred_match, r["class_ids"], overlaps)
            # for i in range(len(IoUs_image)):
            #     IoUs[i] += IoUs_image[i]
            # IoU_classes += [IoU_classes_image]

            confusstion_matrix += vis_pdl1.get_confusion_matrix(5, gt_class_ids, r["class_ids"], r["scores"],
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

            #  obtain all the elemnts in pred which have corresponding GT elemnt
            pred_match_exist = pred_match > -1
            #  retrieve the index of the GT element at the position of the correlated element in prediction
            # sort_gt_as_pred = pred_match[pred_match_exist].astype(int)
            matched_classes.append(r["class_ids"][pred_match_exist])

            score = vis_pdl1.score_almost_metric(gt_masks, gt_class_ids, r['masks'], r['class_ids'])
            if not math.isnan(score):
                score_accuracy += [score]
            evaluated_images.append(dataset_val.image_info[image_id]["id"])
            metric_data["evaluated_images"] = evaluated_images
            metric_data["accuracy_per_image"] = accuracy_per_image
            metric_data["gt_area_per_image"] = gt_area_per_image
            metric_data["accuracy_per_image_air_filt"] = accuracy_per_image_air_filt
            metric_data["gt_area_per_image_air_filt"] = gt_area_per_image_air_filt
            metric_data["confusstion_matrix"] = confusstion_matrix
            metric_data["confusion_matrix_by_pixel"] = confusion_matrix_by_pixel
            metric_data["confusion_matrix_by_pixel_air_filt"] = confusion_matrix_by_pixel_air_filt
            metric_data["areas"] = areas
            metric_data["areas_air_filt"] = areas_air_filt
            metric_data["gt_areas"] = gt_areas
            metric_data["gt_areas_air_filt"] = gt_areas_air_filt
            metric_data["score_accuracy"] = score_accuracy
            metric_data["IoU_classes"] = IoU_classes
            metric_data["IoUs"] = IoUs
            with open(output_file, 'wb') as out_file:
                pickle.dump(metric_data, out_file)

        # save all the test results to a file
        print("result dir: " + str(vis_pdl1.result_dir))
        file_path = os.path.join(vis_pdl1.result_dir, "out.txt")
        with open(file_path, "w") as file:
        #     mean_IoU_per_image_per_class = np.zeros((5, 1))
        #     IoU_classes = np.stack(IoU_classes).reshape(-1, 5)
        #     for i in range(IoU_classes.shape[1]):
        #         if not any(IoU_classes[:, i] != 0):
        #             continue
        #         mean_IoU_per_image_per_class[i] = IoU_classes[IoU_classes[:, i] != 0, i].mean()
        #     # mean_IoU_per_image_per_seg = np.mean(IoU_classes, axis=0)
        #     mean_IoUs_per_seg = np.zeros((len(IoUs), 1))
        #     for i in range(len(IoUs)):
        #         if not IoUs[i]:
        #             continue
        #         IoUs[i] = np.array(IoUs[i])
        #         mean_IoUs_per_seg[i] = np.mean(IoUs[i])
        #     file.write('accuracy:\n{}\n'.format(score_accuracy))
        #     file.write("IoU over segments is \n{}\n".format(mean_IoUs_per_seg))
        #     file.write("IoU over images is \n{}\n".format(mean_IoU_per_image_per_class))

            file.write("the confusion matrix is:\n {}\n".format(confusstion_matrix))
            file.write("the confusion matrix by pixel is:\n {}\n".format(confusion_matrix_by_pixel))
            file.write("the confusion matrix by pixel air filtered is:\n {}\n".format(confusion_matrix_by_pixel_air_filt))

            file.write("\ncustom accuracy data:\n")
            custom_accuracy = 0
            total_gt_area = 0
            custom_accuracy_air_filt = 0
            total_gt_area_air_filt = 0
            for key in accuracy_per_image.keys():
                custom_accuracy += accuracy_per_image[key] * gt_area_per_image[key]
                total_gt_area += gt_area_per_image[key]
                file.write("custom accuracy for image {} is {} with gt area {}:\n".format(key, accuracy_per_image[key],
                                                                              gt_area_per_image[key]))
            file.write("air filtered accuracies:\n")
            for key in accuracy_per_image_air_filt.keys():
                custom_accuracy_air_filt += accuracy_per_image_air_filt[key] * gt_area_per_image_air_filt[key]
                total_gt_area_air_filt += gt_area_per_image_air_filt[key]
                file.write("custom accuracy for image {} is {} with gt area {}:\n".format(key, accuracy_per_image_air_filt[key],
                                                                                          gt_area_per_image_air_filt[key]))
            weighted_avg_accuracy = custom_accuracy / total_gt_area
            file.write("total accuracy (average was weighted by gt area): {}\n".format(weighted_avg_accuracy))
            weighted_avg_accuracy_air_filt = custom_accuracy_air_filt / total_gt_area_air_filt
            file.write("total accuracy for air filtered (average was weighted by gt area): {}".format(weighted_avg_accuracy_air_filt))

            pred_score = areas[0] / (areas[0] + areas[1])
            gt_score = areas[2] / (areas[2] + areas[3])
            pred_score_air_filt = areas_air_filt[0] / (areas_air_filt[0] + areas_air_filt[1])
            gt_score_air_filt = areas_air_filt[2] / (areas_air_filt[2] + areas_air_filt[3])
            file.write("\nprediction score (by area): {}\n".format(pred_score))
            file.write("ground truth score (by area): {}\n".format(gt_score))
            file.write("prediction score (by area, air filtered): {}\n".format(pred_score_air_filt))
            file.write("ground truth score (by area, air filtered): {}\n".format(gt_score_air_filt))

            vis_pdl1.plot_hist(score_accuracy, savename="area_diff_hist")
            # create new class list to replace the 'BG' with 'other'
            right_indices = [4, 5, 2, 3]
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
                                           ["other", "air", "pdl-negative", "pdl-positive"], savename="confusion_matrix")
            # for col in range(3):
            #     confusion_matrix_by_pixel_air_filt[:, col] = confusion_matrix_by_pixel_air_filt[:, col] / gt_areas_air_filt[col]
            confusion_matrix_by_pixel_air_filt = confusion_matrix_by_pixel_air_filt / np.sum(confusion_matrix_by_pixel_air_filt, 0)
            vis_pdl1.plot_confusion_matrix(confusion_matrix_by_pixel_air_filt,
                                           ["other", "air", "pdl-negative", "pdl-positive"], savename="confusion_matrix_air_filtered")
