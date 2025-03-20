# data/data_loader.py
import pandas as pd
from operator import index

from code.curriculum.config import DATASET_NAME
from code.data.all_datasets import SimpleGermanDataset, SimpleWikiDataset, \
    get_classification_xy  # Adjust the import as necessary
from datasets import Dataset


def load_and_prepare_dataset(seed = 42, split='train', split_by_level=False):
    """
    Load and prepare dataset for a specified seed and split.

    Args:
        seed: Set seed for reproducibility
        split: Specified split of the dataset to use
        split_by_level: Get dataset in a format that allows it to access each level

    Returns:
        A huggingface compatible Dataset which is grouped by levels for split_by_level=True
    """

    # Load the appropriate dataset based on DATASET_NAME
    if DATASET_NAME == 'SimpleWiki':
        dataset = SimpleWikiDataset(split=split, seed=seed)
    elif DATASET_NAME == 'SimpleGerman':
        dataset = SimpleGermanDataset(split=split, seed=seed)
    else:
        raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")
    print(f"Loaded dataset with {len(dataset)} samples.")

    metric_names = list(dataset[0]["metrics"].keys())
    x_train,y_train,sents_train = get_classification_xy(dataset, balanced=False, return_sents=True)


    # Convert dataset to a list of dictionaries
    data = [{'sentence': sents_train[index], 'metrics': x_train[index], 'level': y_train[index]} for index in range(len(x_train))]
    print("Converted dataset to list of dictionaries.")

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    print(f"Created DataFrame with shape: {df.shape}")
    print(df.head())

    # Verify that all 'metrics' lists have the correct length
    metrics_length = df['metrics'].apply(len)
    expected_length = len(metric_names)
    if not (metrics_length == expected_length).all():
        raise ValueError("Mismatch between number of metrics and metric_names length.")

    # Split 'metrics' into separate columns
    df_metrics = pd.DataFrame(df['metrics'].tolist(), columns=metric_names)
    df = pd.concat([df.drop(columns=['metrics']), df_metrics], axis=1)
    print("Split 'metrics' into separate columns.")
    print("Columns in DataFrame after splitting:", df.columns.tolist())



    # Initialize a set to keep track of outlier indices
    outlier_indices = set()

    # Detect and collect outliers for each metric
    short_sentence_indices = df[df["length"] < 4].index
    outlier_indices.update(short_sentence_indices)

    # Filter out the outliers from the DataFrame
    df = df[~df.index.isin(outlier_indices)]
    print(f"Filtered DataFrame now has {len(df)} rows.")

    if split_by_level:
        # Group the DataFrame by 'level' and create separate HuggingFace Datasets
        grouped_datasets = {}
        for level, group_df in df.groupby('level'):
            hf_dataset = Dataset.from_pandas(group_df.reset_index(drop=True))
            grouped_datasets[level] = hf_dataset
            print(f"Created HuggingFace Dataset for level '{level}' with {len(hf_dataset)} rows.")
        return grouped_datasets
    else:
        # Create a single HuggingFace Dataset from the entire DataFrame
        hf_dataset = Dataset.from_pandas(df)
        print(f"Created HuggingFace Dataset with {len(hf_dataset)} rows and columns: {hf_dataset.column_names}")
        return hf_dataset
