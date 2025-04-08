
import numpy as np
import random
import sys
import torch
from transformers import set_seed, BertTokenizerFast

from simple_project.curriculum.config import TOKENIZER_SAVE_BASE_PATH
from simple_project.curriculum.create_paths import *
from simple_project.curriculum.tokenizer.tokenizer_training import train_and_save_tokenizer
from config import (
    TRAINING_STRATEGY,
    MODEL_SAVE_PATH,
    TOKENIZER_SAVE_PATH,
    SEED, TRAIN_TOKENIZER
)
from data.data_loader import load_and_prepare_dataset
from data.metrics import compute_difficulty, adjust_difficulty, compute_cdf
from model.model_setup import initialize_model
from strategies.competence import CompetenceTraining
from strategies.incremental import IncrementalTraining
from strategies.sequential import SequentialTraining


def main():
    """
    Main function to initialize BERT MLM pretraining with different training strategies.
    """
    seed = SEED  # or any integer you choose
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # from transformers



    hyperparams = get_current_training_params(TRAINING_STRATEGY)


    # Set the dynamic save paths in config.py
    model_path, tokenizer_path = get_save_paths(MODEL_SAVE_PATH,TOKENIZER_SAVE_PATH,hyperparams)

    print(model_path)

    # Ensure the directories exist
    os.makedirs(model_path,exist_ok=True)
    

    print(f"Model will be saved to: {model_path}")
    print(f"Tokenizer will be saved to: {tokenizer_path}")



    # Step 1: Select Training Strategy
    strategy_type = TRAINING_STRATEGY['type'].lower()
    print(f"Selected training strategy: {strategy_type}")

    if strategy_type == 'sequential' or strategy_type == 'incremental':
        split_by_level = True
    elif strategy_type == 'competence':
        split_by_level = False
    else:
        raise ValueError(f"Unknown training strategy type: {strategy_type}")

    # Step 2: Load and prepare datasets
    print("Loading and preparing datasets...")
    if strategy_type == 'sequential' or strategy_type == 'incremental':
        datasets_by_level = load_and_prepare_dataset(seed=seed, split='train', split_by_level=True)
        val_dataset = load_and_prepare_dataset(seed=seed, split='val', split_by_level=False)
    else:  # competence
        dataset = load_and_prepare_dataset(seed=seed, split='train', split_by_level=False)
        val_dataset = load_and_prepare_dataset(seed=seed, split='val', split_by_level=False)



    # Step 3: Train new tokenizer on datasets or load one
    if TRAIN_TOKENIZER == True:
        print("Preparing texts for tokenizer training...")
        if strategy_type == 'sequential' or strategy_type == 'incremental':
            combined_texts = []
            for level, ds in datasets_by_level.items():
                combined_texts.extend(ds["sentence"])
            print(f"Total sentences for tokenizer training: {len(combined_texts)}")
        else:  # competence
            combined_texts = dataset["sentence"]
            print(f"Total sentences for tokenizer training: {len(combined_texts)}")

        print("Training and saving tokenizer...")
        tokenizer = train_and_save_tokenizer(
            texts=combined_texts,tokenizer_path=tokenizer_path
        )

    else:
        print("Load tokenizer...")
        if DATASET_NAME == "SimpleWiki":
            tokenizer_path = TOKENIZER_SAVE_BASE_PATH + "/tokenizer_swiki"
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        elif DATASET_NAME == "SimpleGerman":
            tokenizer_path = TOKENIZER_SAVE_BASE_PATH + "/tokenizer_sger"
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

    # Step 4: Compute difficulty metrics
    print("Computing difficulty metrics...")
    if strategy_type == 'sequential' or strategy_type == 'incremental':
        for level, ds in datasets_by_level.items():
            print(f"Processing level: {level}")

            #read out difficulty and add column to dataframe
            ds = ds.map(compute_difficulty)
            ds = adjust_difficulty(ds, DIFFICULTY_METRIC)

            #compute cdf scores and add column to dataframe
            cdf_scores = compute_cdf(ds)
            ds = ds.add_column("cdf_score", cdf_scores)
            datasets_by_level[level] = ds
            print(f"Completed processing for level: {level}")
    else:  # competence
        print("Processing combined dataset for competence-based training...")

        #compute difficulty and cdf score
        dataset = dataset.map(compute_difficulty)
        dataset = adjust_difficulty(dataset, DIFFICULTY_METRIC)
        cdf_scores = compute_cdf(dataset)
        dataset = dataset.add_column("cdf_score", cdf_scores)
        print("Completed processing for competence-based training.")
        print("Example from dataset: ")
        print(dataset[0])



    # Step 5: Tokenize datasets
    print("Tokenizing datasets...")

    def tokenize_function(examples):
        """
        Tokenize sentences in for further processing with BERT training.
        """
        sentences = [" ".join(sentence) if isinstance(sentence, list) else sentence for sentence in
                     examples["sentence"]]
        return tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_special_tokens_mask=True)


    if strategy_type == 'sequential' or strategy_type == 'incremental':
        for level, ds in datasets_by_level.items():
            print(f"Tokenizing level: {level}")

            #tokenize datasets from every level
            ds_tokenized = ds.map(tokenize_function, batched=True)
            ds_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            datasets_by_level[level] = ds_tokenized
            print(f"Tokenization completed for level: {level}")
    else:  # competence
        print("Tokenizing combined dataset for competence-based training...")

        #tokenize combined dataset
        dataset_tokenized = dataset.map(tokenize_function, batched=True)

        print("Dataset size: ", len(dataset_tokenized))

        dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        print("Tokenization completed for competence-based training.")
        print("Dataset Tokenized entry: ")
        print(dataset_tokenized[0])
        print(dataset_tokenized['cdf_score'][0])
        print(dataset_tokenized['length'][0])
        print(dataset_tokenized['word_rarity'][0])
        print(dataset_tokenized['difficulty'][0])
        print(dataset_tokenized['sentence'][0])

    print("Tokenizing validation dataset...")
    val_dataset_tokenized = val_dataset.map(tokenize_function, batched=True)
    val_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])


    # Step 6: Initialize the model
    print("Initializing the BERT model...")
    model, device = initialize_model()

    # Step 7: Select and execute training strategy based on config
    if strategy_type == 'sequential':
        strategy = SequentialTraining()

        #order in which the model trains on the classes
        if DATASET_NAME=="SimpleGerman":
            level_order = [2,1,3]  # Adjust based on your actual levels
        elif DATASET_NAME=="SimpleWiki":
            level_order = [1,3]

        #order levels in dataset
        ordered_levels = [level for level in level_order if level in datasets_by_level]
        ordered_datasets = {level: datasets_by_level[level] for level in ordered_levels}
        print(TRAINING_STRATEGY['sequential'])

        print("Starting Sequential Training...")
        strategy.train(
            model=model,
            device=device,
            tokenizer=tokenizer,
            datasets=ordered_datasets,
            config=TRAINING_STRATEGY['sequential'],
            val_dataset=val_dataset_tokenized
        )
    elif strategy_type == 'competence':
        strategy = CompetenceTraining()
        print("Starting competence-based Training...")
        strategy.train(
            model=model,
            device=device,
            tokenizer=tokenizer,
            dataset=dataset_tokenized,
            config=TRAINING_STRATEGY['competence'],
            val_dataset=val_dataset_tokenized
        )
    elif strategy_type == 'incremental':
        strategy = IncrementalTraining()

        #order in which the model trains on the classes
        if DATASET_NAME=="SimpleGerman":
            level_order = [2,1,3]
        elif DATASET_NAME=="SimpleWiki":
            level_order = [1,3]

        # order levels in dataset
        ordered_levels = [level for level in level_order if level in datasets_by_level]
        ordered_datasets = {level: datasets_by_level[level] for level in ordered_levels}
        print(TRAINING_STRATEGY['incremental'])
        print("Starting Incremental Training...")
        strategy.train(model=model,
                       device=device,
                       tokenizer=tokenizer,
                       datasets=ordered_datasets,
                       config=TRAINING_STRATEGY['incremental'],
                        val_dataset=val_dataset_tokenized)
    else:
        raise ValueError(f"Unknown training strategy type: {strategy_type}")

    # Step 8: Save the final model and tokenizer
    print("Saving the final model and tokenizer...")
    os.makedirs(model_path, exist_ok=True)
    #os.makedirs(tokenizer_path, exist_ok=True)
    model.save_pretrained(model_path)
    #tokenizer.save_pretrained(tokenizer_path)
    print(f"Model saved to {model_path}.")
    #print(f"Tokenizer saved to {tokenizer_path}.")

if __name__ == "__main__":
    main()
