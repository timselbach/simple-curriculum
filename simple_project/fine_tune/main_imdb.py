import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import load_dataset, DatasetDict

from simple_project.curriculum.config import MODEL_SAVE_BASE_PATH, TOKENIZER_SAVE_BASE_PATH

# ---------------------------------------------------------------------------
# 1. Load IMDb dataset and rename "test" split
# ---------------------------------------------------------------------------
imdb = load_dataset("imdb")
imdb_renamed = DatasetDict({
    "train": imdb["train"],
    "validation": imdb["test"]
})


# Rename "label" column -> "labels" for Trainer
imdb_renamed = imdb_renamed.rename_column("label", "labels")

# ---------------------------------------------------------------------------
# 2. Load tokenizer & model
# ---------------------------------------------------------------------------

tokenizer_path = TOKENIZER_SAVE_BASE_PATH + "/tokenizer_swiki"
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

# Define your model path
extra_path = "/incremental_SimpleWiki_training_steps_per_levelsteps250k_250k_bs8_lr0p00010_ue10000_20250128_233633"
model_path = MODEL_SAVE_BASE_PATH + extra_path

model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2
)
print(model)


# large pre-trained models for comparison
#tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
#model = BertForSequenceClassification.from_pretrained("bert-base-uncased")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters: ",count_parameters(model))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)
print(model)

# ---------------------------------------------------------------------------
# 3. Tokenize the dataset
# ---------------------------------------------------------------------------
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

imdb_encoded = imdb_renamed.map(preprocess_function, batched=True)
imdb_encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------------------------------------------------------------
# 4. Prepare training & validation sets
# ---------------------------------------------------------------------------
train_dataset = imdb_encoded["train"]
eval_dataset = imdb_encoded["validation"]

# ---------------------------------------------------------------------------
# 5. Define evaluation metrics using scikit-learn
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# ---------------------------------------------------------------------------
# 6. TrainingArguments
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="my-imdb-finetune-results",
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    optim="adamw_torch",
    num_train_epochs=2,
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),
)

# ---------------------------------------------------------------------------
# 7. Create a Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()

log_history = trainer.state.log_history

epochs_train = []
training_losses = []
epochs_eval = []
eval_losses = []
eval_accs = []

for entry in log_history:
    # Training loss
    if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
        epochs_train.append(entry["epoch"])
        training_losses.append(entry["loss"])

    # Evaluation loss
    elif "eval_loss" in entry and "epoch" in entry:
        epochs_eval.append(entry["epoch"])
        eval_losses.append(entry["eval_loss"])
        eval_accs.append(entry["eval_accuracy"])

plt.figure(figsize=(10, 6))
plt.plot(epochs_train, training_losses, label="Training Loss", marker='o')
plt.plot(epochs_eval, eval_losses, label="Validation Loss", marker='o')
plt.plot(epochs_eval, eval_accs, label="Validation Accuracys", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss on IMDb")
plt.legend()
plt.grid(True)
plt.show()
