from imgaug import augmenters as iaa
from datautils import pdl1_net_data_loader as loaderlib
import PIL


class PDL1NetTrainer:
    """
    class represents a PDL1 Net Trainer
    """
    def __init__(self, model, config, args, augment=True):
        # Training dataset.
        self.dataset_train = loaderlib.PDL1NetDataset()
        self.dataset_train.load_pdl1net_dataset(args.dataset, "train")
        self.dataset_train.prepare()

        # Validation dataset
        self.dataset_val = loaderlib.PDL1NetDataset()
        self.dataset_val.load_pdl1net_dataset(args.dataset, "val")
        self.dataset_val.prepare()
        self.augment = augment
        
        self.model = model
        self.config = config

    @staticmethod
    def augmenter():
        seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
        ], random_order=True) # apply augmenters in random order
        return seq

    def train(self, augment=None):
        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        print("Training network heads")
        if augment is not None:
            augmenter = PDL1NetTrainer.augmenter()
        else:
            augmenter = None
        self.model.train(self.dataset_train, self.dataset_val,
                learning_rate=self.config.LEARNING_RATE,
                augmentation=augmenter,
                epochs=10,
                layers='heads')
