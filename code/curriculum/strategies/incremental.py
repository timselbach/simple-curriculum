# strategies/incremental.py

from code.curriculum.training.two_phase_training import train_phase_incremental


class IncrementalTraining():
    def train(self, model, device, tokenizer, datasets, config, val_dataset):
        """
        Executes incremental training on a dataset that can be seperated by levels

        Args:
            model: The pre-initialized model
            device: The device to train on
            tokenizer: The tokenizer being used
            datasets: Tokenized dataset by level
            config: Training configuration
            val_dataset: Tokenized validation dataset

        """
        train_phase_incremental(model, device, tokenizer, datasets, config, val_dataset)