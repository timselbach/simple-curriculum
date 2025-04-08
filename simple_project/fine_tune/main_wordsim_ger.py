import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import BertTokenizerFast, BertForMaskedLM, BertForSequenceClassification

from simple_project.curriculum.config import MODEL_SAVE_BASE_PATH, DATASETS_BASE_PATH, TOKENIZER_SAVE_BASE_PATH

# --------------------------
# 1. Load the tokenizer and model
# --------------------------
tokenizer_path = TOKENIZER_SAVE_BASE_PATH + "/tokenizer_sger"
extra_path = "/sequential_SimpleGerman_training_steps_per_levelsteps100k_100k_300k_bs8_lr0p00010_ue10000_20250128_104312"
model_path = MODEL_SAVE_BASE_PATH + extra_path

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
model = BertForMaskedLM.from_pretrained(model_path)


#large pre-trained models for comparison

#tokenizer = BertTokenizerFast.from_pretrained("bert-base-german-dbmdz-uncased")
#model = BertForSequenceClassification.from_pretrained("bert-base-german-dbmdz-uncased")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

print(model)

# --------------------------
# 2. Load the data from the new text structure
# --------------------------
file_path = DATASETS_BASE_PATH+"/Analogies/de_re-rated_Schm280.txt"

# There are two columns in each line:
#   "word1-word2    similarity"
data = pd.read_csv(file_path, sep="\t", header=None, names=["combined", "similarity"])

# Now split the 'combined' column into 'word1' and 'word2'
split_cols = data["combined"].str.split("-", expand=True)
data["word1"] = split_cols[0]
data["word2"] = split_cols[1]

# Drop the 'combined' column
data.drop(columns=["combined"], inplace=True)

# --------------------------
# 3. Function to get word embedding
# --------------------------
def get_word_embedding(word: str):
    """
    Outputs the token embedding after getting passed through the model.
    When a word is split into multiple tokens, the average word embedding is computed.

    Returns:
         Mean of the token embeddings for the input word.
    """
    inputs = tokenizer(word, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model.bert(**inputs)



    # Exclude [CLS] and [SEP] tokens from the output embeddings
    embeddings = outputs.last_hidden_state[0, 1:-1, :]
    print(embeddings.shape)

    # Fallback to [CLS] if no tokens were produced
    if embeddings.shape[0] == 0:
        embeddings = outputs.last_hidden_state[0, 0, :].unsqueeze(0)

    return embeddings.mean(dim=0)


cos_sim = torch.nn.CosineSimilarity(dim=0)

# --------------------------
# 4. Compute similarities for each word pair
# --------------------------
predicted_sims = []
gold_sims = []

print("\nWord Pair Similarities:")
for _, row in data.iterrows():
    word1, word2, gold_score = row["word1"], row["word2"], row["similarity"]

    emb1 = get_word_embedding(word1)
    emb2 = get_word_embedding(word2)

    # Compute cosine similarity (as a float)
    sim = cos_sim(emb1, emb2).item()

    predicted_sims.append(sim)
    gold_sims.append(gold_score)

    # If desired, print each pair with predicted and gold scores
    print(f"{word1} - {word2} | Predicted Similarity: {sim:.4f} | Gold Similarity: {gold_score}")

# --------------------------
# 5. Evaluate with pearson and spearman correlations
# --------------------------
pearson_corr, _ = pearsonr(predicted_sims, gold_sims)
spearman_corr, _ = spearmanr(predicted_sims, gold_sims)

print("\nEvaluation Results:")
print(f"Pearson correlation:  {pearson_corr:.4f}")
print(f"Spearman correlation: {spearman_corr:.4f}")


# plot a scatter plot of predictions and standard similarities

# plt.figure(figsize=(6, 6))
# plt.scatter(gold_sims, predicted_sims, alpha=0.6)
# plt.xlabel("Gold Similarities")
# plt.ylabel("Predicted Similarities")
# plt.title("Gold vs. Predicted Similarities")
# plt.grid(True)
# plt.show()
