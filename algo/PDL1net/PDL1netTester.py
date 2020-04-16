class PDL1NetTester:
    """
    class represents a PDL1 net Tester
    """

    def __init__(self, model):
        self.model = model

    def test(self, images):
        if images is None:
            raise NameError("None was sent, but list of images is expected")
        # if only one image was sent wrap it in list
        if not isinstance(images, list):
            images = [images]
        return model.detect(images, verbose=1)[0]
    
    def color_splash(image, mask):
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


    def detect_and_color_splash(model, image_path=None):
        assert image_path, "image path is missing"
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = self.test(image)
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        print("Saved to ", file_name)
    