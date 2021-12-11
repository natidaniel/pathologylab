"""
creator of synthetic data.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Updated by Itamar Gruber & Shai NG for PDL1Net project - April 2020
"""
import os
import sys
import math
import numpy.random as random
import numpy as np
import cv2
import colorsys
from PIL import Image
import utils

#  set a color palette with equidistant color
CONST_COLOR_PALETTE = [colorsys.hsv_to_rgb(i / 4, 1.0, 1.0) for i in range(4)]
for i, color in enumerate(CONST_COLOR_PALETTE):
    CONST_COLOR_PALETTE[i] = tuple([int(color[j] * 255) for j in range(3)])  # convert color to 255 integer dim'


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles, ellipse) placed randomly on a cells image(IHC WSI patch) surface.
    The images are generated on the fly. The Dataset object supports at most 4 classes (must have at least 1 class).
    """
    def __init__(self, config):
        self.config = config
        super(ShapesDataset, self).__init__()

    def load_shapes(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        for i, c in enumerate(self.config.CLASSES):
            # Add classes
            self.add_class(self.config.DATA_NAME, i + 1, c)

        self.COLOR_PALETTE = []
        if self.config.IS_CONST_COLOR_SHAPES:
            # for _ in range(self.config.NUM_CLASSES):
            #     # color = colorsys.hsv_to_rgb(random.rand(), 1.0, 1.0)  # ensures no brown color is chosen
            #
            #     color = tuple([int(color[i] * 255) for i in range(3)])  # convert color to 255 integer dim'
            #     self.COLOR_PALLET.append(color)

            # match classes tags to colors
            self.COLOR_PALETTE = dict(zip(self.config.CLASSES, CONST_COLOR_PALETTE))

        self.PATH_TO_IMAGE_DIR = self.config.PATH_TO_IMAGE_DIR
        self.PATH_SAVE_IMAGES = self.config.IMAGE_DIR
        self.PATH_SAVE_ANNOTATIONS = self.config.ANNOTATION_DIR
        if self.config.IS_BG_UNIFIED == False:
            ls_dir_images = os.listdir(self.PATH_TO_IMAGE_DIR)

        height = self.config.IMAGE_HEIGHT
        width = self.config.IMAGE_WIDTH

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for _ in range(self.config.NUM_OF_IMAGES_TO_GENERATE):
            path_to_image = None
            if self.config.IS_BG_UNIFIED == False:
                i = random.randint(len(ls_dir_images))
                while os.path.isdir(ls_dir_images[i]) and os.path.splitext(ls_dir_images[i])[1] != '.json':
                    i = (i + 1) % len(ls_dir_images)
                path_to_image = os.path.join(self.PATH_TO_IMAGE_DIR, ls_dir_images[i])
            shapes = self.random_image(height, width)
            self.add_image(self.config.DATA_NAME, image_id=i, path=path_to_image,
                           width=width, height=height,
                           shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file,
        in this case it also generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        if self.config.IS_BG_UNIFIED == False:
            image = Image.open(info['path'])
            image = image.copy()  # ?
            image = np.array(image.resize([info['height'], info['width']]))
        else:  # creates images with white background
            image = np.ones([info['height'], info['width'], 3], dtype=np.uint8) * 255
        # sorting the shape list due to the priority of each class
        # that way we ensure that the stronger classes overdraw the weaker ones
        copy_info = info['shapes']
        sort_func = lambda elem : self.config.CLASS_TO_ID[elem[0]]
        copy_info.sort(key=sort_func)
        # draw the shapes
        for shape, color, dims in copy_info:
            image = self.draw_shape(image, shape, dims, color)

        image_path = os.path.join(self.PATH_SAVE_IMAGES, '{:05d}'.format(image_id))
        image_path += ".jpeg"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, image)

        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.config.DATA_NAME:
            return info['shapes']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 255)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])

        for i, (shape, _, dims) in enumerate(info['shapes']):
            path = os.path.join(self.PATH_SAVE_ANNOTATIONS, '{:05d}'.format(image_id))
            path += "_" + shape + "_" + '{:05d}'.format(i) + ".png"
            cv2.imwrite(path, mask[:, :, i])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims

        if shape == self.config.CLASSES[0]:
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
        elif len(self.config.CLASSES) >= 2 and shape == self.config.CLASSES[1]:
            image = cv2.circle(image, (x, y), s, color, -1)
        elif len(self.config.CLASSES) >= 3 and shape == self.config.CLASSES[2]:
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
        elif len(self.config.CLASSES) >= 4 and shape == self.config.CLASSES[3]:
            image = cv2.ellipse(image, (x, y), (s, s * 2 // 3), 0, 0, 360, color, thickness=-1)

        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three values:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        * @param buffer is the minimum size and coordinate of the generated shape
        """
        # Shape
        shape = random.choice(self.config.CLASSES)
        # Color
        if self.config.IS_CONST_COLOR_SHAPES == False:
            #  random color is chosen each iteration
            color = colorsys.hsv_to_rgb(random.rand(), 1.0, 1.0)  # ensures no brown color is chosen
            color = tuple([int(color[i]*255) for i in range(3)])  # convert color to 255 integer dim'
        else:
            #  select the color from pre chosen pallet (see in load_shape)
            color = self.COLOR_PALETTE[shape]
        # Center x, y
        buffer = 50
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns a list of shape specifications that can be used to draw the image.
        """
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(self.config.RAND_NUM_OF_SHAPES[0], self.config.RAND_NUM_OF_SHAPES[1] + 1)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])

        keep_ixs_area = utils.suppress_inclusion(np.array(boxes), np.arange(N))
        keep_ixs_iou = utils.non_max_suppression(np.array(boxes), np.arange(N), self.config.OVERLAPING_MAX_IOU)
        keep_ixs = np.intersect1d(keep_ixs_area, keep_ixs_iou)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return shapes
