import os
import sys
import multiprocessing
sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.join("..", "..", "algo"))
import params.PDL1NetConfig as pdl1_config
import algo.PDL1Net.PDL1NetTester as Tester

import algo.mrcnn.model as modellib

ROOT_DIR = os.path.join(os.getcwd(), "..", "..")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

def run_test(args):
    class InferenceConfig(pdl1_config.PDL1NetConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BACKBONE = args.backbone

    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    model.load_weights(args.weights, by_name=True)

    path, epoch_name = os.path.split(args.weights)
    path, log_name = os.path.split(path)
    epoch = os.path.splitext(epoch_name)[0].split("_")[-1]
    output_dir_name = "{}_{}".format(log_name, epoch)

    tester = Tester.PDL1NetTester(model, args)
    tester.test_sequence(result_dir_name=output_dir_name)

class Arguments:
    def __init__(self, weights, backbone, dataset):
        self.weights = weights
        self.backbone = backbone
        self.dataset = dataset
        self.logs = DEFAULT_LOGS_DIR


if __name__ == "__main__":

    weights_root = r"D:\Nati\Itamar_n_Shai\pathologylab\logs"
    logs_name = ["50_augm1_07"]
    logs = [os.path.join(weights_root, name) for name in logs_name]

    dataset = r"D:\Nati\Itamar_n_Shai\Datasets\data_yael\DataMaskRCNN"

    epochs = ["30", "60", "90"]

    for epoch in epochs:
        for index, log_path in enumerate(logs):
            backbone = "resnet" + logs_name[index].split("_")[0]
            lst_dir = os.listdir(log_path)
            weight = None
            for file in lst_dir:
                if os.path.splitext(file)[1] != ".h5":
                    continue
                if epoch not in file:
                    continue
                weight = os.path.join(log_path, file)
                break
            if weight is None:
                raise(ValueError, "weight epoch {} was not found in {}".format(epoch, log_path))
            args = Arguments(weight, backbone, dataset)
            p = multiprocessing.Process(target=run_test, args=(args,))
            p.start()
            p.join()
            # run_test(args)
    pass
