# config/config.py

# ----------------------------
# General Configurations
# ----------------------------

# Set the difficulty metric here. Choose one from the available metrics.
# Available options:
# 'length', 'uni_entropy', 'bi_entropy', 'tri_entropy',
# 'word_rarity', 'diversity', 'complexity',
# 'lix', 'gunning_fog', 'coleman_liau',
# 'autom_reading', 'flesch_kincaid'

DIFFICULTY_METRIC = 'word_rarity'  # Change this value to use a different metric

DATASETS_BASE_PATH = "C:/Users/timse/Documents/Test repo/simple-curriculum/datasets"



# ----------------------------
# Tokenizer Settings
# ----------------------------
TRAIN_TOKENIZER = False

TOKENIZER_SAVE_BASE_PATH = "C:/Users/timse/Documents/Test repo/simple-curriculum/results/tokenizer"
VOCAB_SIZE = 30000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# ----------------------------
# Model Settings
# ----------------------------

MODEL_SAVE_BASE_PATH = "C:/Users/timse/Documents/Test repo/simple-curriculum/results/model"
BERT_CONFIG = {
    'vocab_size': VOCAB_SIZE,
    'hidden_size': 128,
    'num_hidden_layers': 2,
    'num_attention_heads': 2,
    'intermediate_size': 512,
    'max_position_embeddings': 512,
    'type_vocab_size': 2,
}

# ----------------------------
# Dataset Setting
# ----------------------------
DATASET_NAME = "SimpleGerman"

# ----------------------------
# Training Settings
# ----------------------------

# ----------------------------
# Training Strategy Configuration
# ----------------------------

SEED = 53

RANDOM_CDF = False

TRAINING_STRATEGY = {
    'type': 'incremental',  # Options: 'sequential', 'competence', 'incremental'
    'sequential': {
        'training_steps_per_level': {
            1: 250000,
            3: 250000
        },
        'batch_size': 8,
        'learning_rate': 1e-4,
        'update_every': 20000
    },
    'competence': {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'max_steps_phase': 500000,
        'update_every': 5000,
        'c0': 0.05,
        'max_t_steps': 50000,

    },
    'incremental': {
        'training_steps_per_level': {
            2: 50000,
            1: 150000,
            3: 300000
        },
        'batch_size': 8,
        'learning_rate': 1e-4,
        'update_every': 10000

    }
}


# Initialize the global variables to default paths or None
MODEL_SAVE_PATH = MODEL_SAVE_BASE_PATH
TOKENIZER_SAVE_PATH = TOKENIZER_SAVE_BASE_PATH