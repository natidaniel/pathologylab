
class PDL1NetTrainer:
    """
    class represents a PDL1 Net Trainer
    """
    def __init__(self, model, datasets_path):
        # Training dataset.
        self.dataset_train = PDL1NetDataset()
        self.dataset_train.load_pdl1net_dataset(datasets_path, "train")
        self.dataset_train.prepare()

        # Validation dataset
        self.dataset_val = PDL1NetDataset()
        self.dataset_val.load_pdl1net_dataset(datasets_path, "val")
        self.dataset_val.prepare()
        
        self.model = model

    def train(self):
        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        print("Training network heads")
        self.model.train(self.dataset_train, self.dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
