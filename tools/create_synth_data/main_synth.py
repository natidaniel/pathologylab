
import sys
import os

#  we based our code on shapes from mrcnn project. the script generates and saves synthetic images and annotated masks.
#  The synthetic data plots shapes, on specified images or white BG in varies sizes, colors, and other attributes.
import synth_generator

COCOCREATOR_DIR = os.path.join("pycococreator")
sys.path.append(COCOCREATOR_DIR)  # To find local version of the library

#  this script takes the images with the shapes and masks (both outputted from synth_generator),
#  and output json file describing the masks over the images in COCO format
#  this script base on pycoco creator and uses pycocotools package (which is available only for linux and macOS)
import examples.shapes.shapes_to_coco as shapes_to_coco

class ShapesConfig:
    def __init__(self):
        self.DATA_NAME = "PDL1"
        # the script supports 4 different classes at most
        # the code gives priority to classes with greater
        # indices in case of occlusion
        self.CLASSES = ["inf", "neg", "pos", "other"]
        # Use small images for faster training. Set the limits of the small side and
        # the large side, and that determines the image shape.
        self.IMAGE_WIDTH = 256 * 3
        self.IMAGE_HEIGHT = 256 * 3
        # adds the prefix to all the output directories
        self.ADDED_PREFIX = "train_"
        # path to the root where all the data is saved
        self.ROOT_SYNTH_DATA = os.path.join(".", "..", "..","Data","Synth_Data")
        # path to images to choose BG from
        self.PATH_TO_IMAGE_DIR = os.path.join(self.ROOT_SYNTH_DATA, "input")
        #  This number can be as large as we want, the BG (if not unified)
        #  will be randomize from the images at the given directory.
        self.NUM_OF_IMAGES_TO_GENERATE = 1000
        # determines the number of shapes per image (lower bound, upper bound)
        self.RAND_NUM_OF_SHAPES = [2,5]
        #  takes values in range [0,1] - if the thresh is not satisfied
        #  the overlapped shape will be removed
        self.OVERLAPING_MAX_IOU = 0.5
        #  if True all the shpaes of the same type will have the same color
        self.IS_CONST_COLOR_SHAPES = True
        #  If the BG is not UNIFIED then the algorithm will randomize an
        #  image as BG from the given directory.
        self.IS_BG_UNIFIED = False

        self.logic_init()


    def logic_init(self):
        # Number of classes (including background)
        self.NUM_CLASSES = 1 + len(self.CLASSES)  # background + 4 classes
        # all the output will be under this folder
        self.PATH_SAVE_DATA = os.path.join(self.ROOT_SYNTH_DATA ,
                                self.ADDED_PREFIX + "output" +"_"+str(self.OVERLAPING_MAX_IOU)+
                                  "_"+str(self.IS_CONST_COLOR_SHAPES) + "_" + str(self.IS_BG_UNIFIED))
        # the path to save the json file
        self.ROOT_DIR_PYCOCOCREATOR = self.PATH_SAVE_DATA
        # the path to save the generated images
        self.IMAGE_DIR = os.path.join(self.PATH_SAVE_DATA, self.ADDED_PREFIX + "image")
        # the path to save the generated masks
        self.ANNOTATION_DIR = os.path.join(self.PATH_SAVE_DATA, self.ADDED_PREFIX + "mask")
        # the json file name
        self.JSON_OUTPUT_FILENAME = self.ADDED_PREFIX + 'synth_data.json'
        self.CLASS_TO_ID = dict(zip(self.CLASSES,range(len(self.CLASSES))))


def create_dirs(config):
    try:
        os.makedirs(config.IMAGE_DIR)
        os.makedirs(config.ANNOTATION_DIR)
    except:
        pass

if __name__ == "__main__":
    config = ShapesConfig()
    create_dirs(config)
    a = 0
    # create the Dataset
    data = synth_generator.ShapesDataset(config)
    data.load_shapes()
    data.prepare()

    # save the images and masks from the generated dataset
    for i in range(config.NUM_OF_IMAGES_TO_GENERATE):
        data.load_image(i)
        data.load_mask(i)

    # export the masks to json file in coco format
    shapes_to_coco.main(config)
