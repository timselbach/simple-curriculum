# strategies/sequential.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, DataCollatorForLanguageModeling

from code.curriculum.training.two_phase_training import train_phase_sequential


class SequentialTraining:
    def train(self, model, device, tokenizer, datasets, config, val_dataset):
        """
        Executes sequential training on a dataset that can be seperated by levels

        Args:
            model: The pre-initialized model
            device: The device to train on
            tokenizer: The tokenizer being used
            datasets: Tokenized dataset by level
            config: Training configuration
            val_dataset: Tokenized validation dataset

        """
        train_phase_sequential(model, device, tokenizer, datasets, config, val_dataset)