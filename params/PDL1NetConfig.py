from algo.mrcnn.config import Config

############################################################
#  Configurations
############################################################


class PDL1NetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "PDL1"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + [inf, pos, ned, other]

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 40

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    BACKBONE = "resnet50"

