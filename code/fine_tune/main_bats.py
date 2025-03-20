import itertools
import numpy as np
import os
import pandas as pd
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMaskedLM, BertForSequenceClassification

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to the tokenizer and model (adjust as needed)
tokenizer_path = "/home/iailab34/selbacht0/Sync/results/tokenizer/curriculum_SimpleWiki_flesch_kincaid_bs8_lr0p00010_msph500000_ue15000_c00p05000_max_t_steps300000_20250123_143725"
extra_path = "curriculum_SimpleWiki_word_rarity_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250130_215929"
model_path = os.path.join("/home/iailab34/selbacht0/Sync/results/model", extra_path)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# Load tokenizer and model
#tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
#model = BertForMaskedLM.from_pretrained(model_path)
model.eval()
model.to(device)

# Function to load the BATS dataset
def load_bats_dataset(directory):
    bats_data = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".txt"):
                with open(os.path.join(root, filename), 'r') as file:
                    for line in file:
                        if not line.startswith('#') and line.strip():
                            word_pair = line.strip().split('\t')
                            if len(word_pair) == 2:
                                # Use the last directory name as the relation label,
                                # and the filename (without extension) as the subcategory.
                                relation = os.path.basename(root)
                                subcategory = filename[:-4]
                                bats_data.append((relation, subcategory, word_pair[0], word_pair[1]))
    return pd.DataFrame(bats_data, columns=["Relation", "Subcategory", "Word1", "Word2"])

# Load the dataset
bats_path = "/home/iailab34/selbacht0/Sync/datasets/BATS_3.0/"
bats_data = load_bats_dataset(bats_path)

# Function to get the embedding of a word (using its tokenization)
def get_word_embedding(word: str):
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        # Use the underlying BERT encoder (without the MLM head)
        outputs = model.bert(**inputs)
    # Get token embeddings excluding special tokens ([CLS] and [SEP])
    embeddings = outputs.last_hidden_state[0, 1:-1, :]
    if embeddings.shape[0] == 0:  # Fallback in case tokenization yields no valid tokens
        embeddings = outputs.last_hidden_state[0, 0, :].unsqueeze(0)
    return embeddings.mean(dim=0).cpu().numpy()

# The three_cos_add function that implements the analogy:
# Given words a, b and c, it computes: target = c - a + b,
# then returns the word d (from vocab V \ {a, b, c}) that maximizes cosine similarity.
def three_cos_add(a: str, b: str, c: str, vocab):
    a_vec = get_word_embedding(a)
    b_vec = get_word_embedding(b)
    c_vec = get_word_embedding(c)
    target_vec = c_vec - a_vec + b_vec

    best_word = None
    best_sim = -np.inf

    for word in vocab:
        # Exclude the input words from the candidate set
        if word in {a, b, c}:
            continue
        word_vec = get_word_embedding(word)
        sim = cosine_similarity([target_vec], [word_vec])[0][0]
        if sim > best_sim:
            best_sim = sim
            best_word = word

    return best_word

# For efficiency, we use a reduced vocabulary (here: first 1000 tokens) [:1000]
vocab_list = list(tokenizer.get_vocab().keys())

# We'll sample only a few combinations for each (Relation, Subcategory) group.
max_samples_per_group = 10

# Generate analogies.
# For each (Relation, Subcategory) group, we first form all possible distinct pair combinations.
# Then, we randomly select up to max_samples_per_group combinations.
predicted_analogies = []
grouped = bats_data.groupby(["Relation", "Subcategory"])

# First, calculate the total number of analogy questions to process (for the progress bar)
total_questions = 0
group_combos = {}  # To store chosen index pairs for each group
for key, group in grouped:
    pairs = group[["Word1", "Word2"]].values
    num_pairs = len(pairs)
    # Create all ordered pairs of indices (i, j) where i != j
    all_index_pairs = list(itertools.permutations(range(num_pairs), 2))
    if len(all_index_pairs) > max_samples_per_group:
        chosen_pairs = random.sample(all_index_pairs, max_samples_per_group)
    else:
        chosen_pairs = all_index_pairs
    group_combos[key] = chosen_pairs
    total_questions += len(chosen_pairs)

pbar = tqdm(total=total_questions, desc="Processing Analogies")

# Iterate over each group and process only the selected combinations
for (relation, subcategory), group in grouped:
    pairs = group[["Word1", "Word2"]].values
    # Retrieve the pre-sampled index pairs for this group
    chosen_pairs = group_combos[(relation, subcategory)]
    for i, j in chosen_pairs:
        a, b = pairs[i]
        c, d_expected = pairs[j]
        prediction = three_cos_add(a, b, c, vocab_list)
        quadruplet = (relation, subcategory, a, b, c, d_expected, prediction)
        predicted_analogies.append(quadruplet)
        # Print the prediction for this quadruplet
        print(f"Relation: {relation}, Subcategory: {subcategory}, a: {a}, b: {b}, c: {c}, d_expected: {d_expected}, d_predicted: {prediction}")
        pbar.update(1)
pbar.close()

# Convert results to a DataFrame for analysis
predicted_df = pd.DataFrame(predicted_analogies,
                            columns=["Relation", "Subcategory", "a", "b", "c", "d_expected", "d_predicted"])

# Calculate accuracy per relation by comparing predicted d with expected d.
accuracy_by_relation = predicted_df.groupby("Relation").apply(
    lambda df: np.mean(df["d_predicted"] == df["d_expected"])
).reset_index(name="Accuracy")

print("\nAccuracy by Relation:")
print(accuracy_by_relation)

# Optionally, save the results:
# predicted_df.to_csv("/mnt/data/bats_analogies_results.csv", index=False)
# print("Results saved to /mnt/data/bats_analogies_results.csv")
