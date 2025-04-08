# model/model_setup.py

import torch
from transformers import BertConfig, BertForMaskedLM

from simple_project.curriculum.config import BERT_CONFIG


def initialize_model():
    """
    Creates a BERT model and moves it to cuda device if available.

    Returns:
        model (BertForMaskedLM), device (torch.device)
    """
    config = BertConfig(**BERT_CONFIG)
    model = BertForMaskedLM(config)
    print("Initialized BERT model for MLM.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device




