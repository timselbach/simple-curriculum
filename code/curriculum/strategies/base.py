# strategies/base.py

from abc import ABC, abstractmethod

class TrainingStrategy(ABC):
    @abstractmethod
    def train(self, model, device, tokenizer, simple_dataset, normal_dataset, config):
        """
        Execute the training strategy.

        Parameters:
        - model: The pre-initialized model.
        - device: The device to train on.
        - tokenizer: The tokenizer being used.
        - simple_dataset: Dataset with simple language.
        - normal_dataset: Dataset with normal language.
        - config: Configuration settings.
        """
        pass
