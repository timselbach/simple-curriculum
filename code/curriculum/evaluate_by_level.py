import numpy as np
import random
import sys
import torch
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import set_seed, BertTokenizer, BertTokenizerFast, BertForMaskedLM

from code.curriculum.config import TOKENIZER_SAVE_BASE_PATH, MODEL_SAVE_BASE_PATH
from code.curriculum.create_paths import *
from code.curriculum.training.two_phase_training import evaluate
from config import (
    TRAINING_STRATEGY
)
from data.data_loader import load_and_prepare_dataset
from data.metrics import compute_difficulty, adjust_difficulty, compute_cdf


strategy_type = TRAINING_STRATEGY['type'].lower()
seed = 48

#adjust based on dataset
level_list = [1,2,3]

EVALUATE_LEVEL_WISE = False

#sgerman
tokenizer_path = TOKENIZER_SAVE_BASE_PATH + "/tokenizer_sger"
extra_path = "/curriculum_SimpleGerman_length_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250126_215513"

#swiki
#tokenizer_path = TOKENIZER_SAVE_BASE_PATH + "/tokenizer_swiki"
#extra_path = "/curriculum_SimpleWiki_word_rarity_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250219_174512"


model_path = MODEL_SAVE_BASE_PATH + extra_path

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
model = BertForMaskedLM.from_pretrained(model_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

print(model)

if strategy_type == 'sequential' or strategy_type == 'incremental':
    datasets_by_level = load_and_prepare_dataset(seed=seed, split='train', split_by_level=True)
    val_dataset = load_and_prepare_dataset(seed=seed, split='val', split_by_level=False)
else:  # competence
    dataset = load_and_prepare_dataset(seed=seed, split='train', split_by_level=False)
    val_dataset = load_and_prepare_dataset(seed=seed, split='val', split_by_level=False)

# Step 5: Compute difficulty metrics
print("Computing difficulty metrics...")
if strategy_type == 'sequential' or strategy_type == 'incremental':
    for level, ds in datasets_by_level.items():
        print(f"Processing level: {level}")
        ds = ds.map(compute_difficulty)
        ds = adjust_difficulty(ds, DIFFICULTY_METRIC)
        cdf_scores = compute_cdf(ds)
        ds = ds.add_column("cdf_score", cdf_scores)
        datasets_by_level[level] = ds
        print(f"Completed processing for level: {level}")
else:  # competence
    print("Processing combined dataset for curriculum training...")
    dataset = dataset.map(compute_difficulty)
    dataset = adjust_difficulty(dataset, DIFFICULTY_METRIC)
    cdf_scores = compute_cdf(dataset)
    dataset = dataset.add_column("cdf_score", cdf_scores)
    print("Completed processing for curriculum training.")
    print("Example from dataset: ")
    print(dataset[0])




# Step 6: Tokenize datasets
print("Tokenizing datasets...")


def tokenize_function(examples):
    sentences = [" ".join(sentence) if isinstance(sentence, list) else sentence for sentence in examples["sentence"]]
    return tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True,
    )


if strategy_type == 'sequential' or strategy_type == 'incremental':
    for level, ds in datasets_by_level.items():
        print(f"Tokenizing level: {level}")
        ds_tokenized = ds.map(tokenize_function, batched=True)
        ds_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        datasets_by_level[level] = ds_tokenized
        print(f"Tokenization completed for level: {level}")
else:  # curriculum
    print("Tokenizing combined dataset for curriculum training...")
    dataset_tokenized = dataset.map(tokenize_function, batched=True)
    dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    print("Tokenization completed for curriculum training.")
    print("Dataset Tokenized entry: ")
    print(dataset_tokenized[0])
    print(dataset_tokenized['cdf_score'][0])
    print(dataset_tokenized['length'][0])
    print(dataset_tokenized['word_rarity'][0])
    print(dataset_tokenized['difficulty'][0])
    print(dataset_tokenized['sentence'][0])

print("Tokenizing validation dataset...")
val_dataset_tokenized = val_dataset.map(tokenize_function, batched=True)
print("val_dataset_tokenized entry: ",val_dataset_tokenized[0])

if EVALUATE_LEVEL_WISE:
    print("\nEvaluating level-wise on the validation dataset:")
    for level in level_list:
        # Filter the original validation dataset by level, then tokenize it.
        val_dataset_filtered = val_dataset.filter(lambda example: example["level"] == level)
        val_dataset_filtered = val_dataset_filtered.map(tokenize_function, batched=True)
        val_dataset_filtered.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        #print(val_dataset_filtered[0])

        avg_loss, perplexity, _ = evaluate(model, device=device, tokenizer=tokenizer, val_dataset=val_dataset_filtered)
        print(f"\nLevel: {level}")
        print(f"Average loss: {avg_loss}")
        print(f"Perplexity: {perplexity}")
else:
    print("\nEvaluating on the whole validation dataset:")
    # Use the already tokenized validation dataset.
    val_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    avg_loss, perplexity, _ = evaluate(model, device=device, tokenizer=tokenizer, val_dataset=val_dataset_tokenized)
    print("Evaluation on whole validation dataset:")
    print(f"Average loss: {avg_loss}")
    print(f"Perplexity: {perplexity}")