

#import evaluate
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer

from datasets import load_dataset

from code.curriculum.config import TOKENIZER_SAVE_BASE_PATH, MODEL_SAVE_BASE_PATH


# large pre-trained models for comparison

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-german-cased")
# model = BertForSequenceClassification.from_pretrained("bert-base-german-cased", num_labels=3)

def tokenize_function(example):
    # Tokenize the sentence pair (premise and hypothesis)
    return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding="max_length", max_length=128)

# Paths for the custom tokenizer and model
tokenizer_path = TOKENIZER_SAVE_BASE_PATH + "/tokenizer_sger"
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
extra_path = "/sequential_SimpleGerman_training_steps_per_levelsteps100k_100k_300k_bs8_lr0p00010_ue10000_20250128_104312"
model_path = MODEL_SAVE_BASE_PATH + extra_path

# Load the pre-trained model for sequence classification (3 labels)
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=3
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters: ",count_parameters(model))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)
print(model)

# Load the German subset of the XNLI dataset
xnli_de = load_dataset("xnli", "de")

dataset = xnli_de["train"]

# Tokenize the selected dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Rename the "label" column to "labels" for Trainer
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])


# split dataset into 80/20 split
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)

print(len(split_dataset["train"]))
print(len(split_dataset["test"]))


# Load the accuracy metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_xnli_de_fast",
    save_strategy="no",
    learning_rate=1e-4,
    do_eval=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    metric_for_best_model="accuracy",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=split_dataset["test"])
print("Evaluation results:", eval_results)
