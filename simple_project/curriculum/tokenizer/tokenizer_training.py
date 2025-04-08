# tokenizer/tokenizer_training.py

import os
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizerFast

from simple_project.curriculum.config import VOCAB_SIZE, MIN_FREQUENCY, SPECIAL_TOKENS


def train_and_save_tokenizer(texts,tokenizer_path):
    """
    Train a BertWordPieceTokenizer on the dataset's text
    Args:
        texts: list of all sentences
        tokenizer_path: path to save tokenizer

    Returns:
        Trained tokenizer
    """
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train_from_iterator(
        texts,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS
    )

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    print("Tokenizer trained and saved.")
    return tokenizer


