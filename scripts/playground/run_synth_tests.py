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

    weights_root = r"D:\Nati\Itamar_n_Shai\Mask_RCNN\logs"
    logs_name = [
                    'synth_iou05_c1_bg0',
                    'synth_iou0_c0_bg1',
                    'synth_iou0_c1_bg0',
                    'synth_iou0_c1_bg1'
                ]
    logs = [os.path.join(weights_root, name) for name in logs_name]

    dataset_root = r"D:\Nati\Itamar_n_Shai\Datasets\DataSynth"
    datasets =  [
                    'output_IoU0.5_C1_BG0',
                    'output_IoU0_C0_BG1',
                    'output_IoU0_C1_BG0',
                    'output_IoU0_C1_BG1'
                ]
    datasets = [os.path.join(dataset_root, name) for name in datasets]
    backbone = "resnet50"
    for log_path, dataset_path in zip(logs, datasets):
        weight_path = os.path.join(log_path, "mask_rcnn_pdl1_0090.h5")
        args = Arguments(weight_path, backbone, dataset_path)
        # p = multiprocessing.Process(target=run_test, args=(args,))
        # p.start()
        # p.join()
        run_test(args)
    pass
