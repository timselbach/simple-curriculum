# data/metrics.py
import numpy as np

from code.curriculum.config import (DIFFICULTY_METRIC, RANDOM_CDF)


def compute_difficulty(example, difficulty_metric=DIFFICULTY_METRIC):
    """
    Read out difficulty metric and create attribute for it

    Args:
        example: dictionary with a sentence and all difficulty metrics
        difficulty_metric: specify which difficulty metric to use

    Returns:

    """
    metric_value = example.get(difficulty_metric, 0.0)  # Default to 0.0 if missing
    example["difficulty"] = metric_value
    return example

def adjust_difficulty(dataset, difficulty_metric=DIFFICULTY_METRIC):
    """
    For metrics where a higher score means lower difficulty the values are inverted so they fit
    Args:
        dataset:
        difficulty_metric:

    Returns:
        Dataset with inverted difficulty values
    """

    difficulties = dataset["difficulty"]

    #TODO: add all metrics which need to be inverted
    inverted_metrics = {'flesch_kincaid', 'autom_reading'}

    if difficulty_metric in inverted_metrics:
        max_difficulty = max(difficulties)
        adjusted_difficulties = [max_difficulty - d for d in difficulties]
    else:
        adjusted_difficulties = difficulties

    dataset = dataset.remove_columns("difficulty")
    dataset = dataset.add_column("difficulty", adjusted_difficulties)
    print("Adjusted difficulty values based on the metric.")
    return dataset


def compute_cdf(dataset):
    """
    Compute cdf values for the dataset by their rank

    Args:
        dataset: Dataset without respective cdf values

    Returns:
        CDF scores for each sentence
    """
    # Extract difficulty scores
    if RANDOM_CDF == False:
        difficulties = np.array(dataset["difficulty"])

        # Rank the difficulties
        sorted_indices = difficulties.argsort()
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(difficulties) + 1)

        # Compute CDF
        cdf_scores = ranks / len(difficulties)
        print("Computed CDF scores for all samples.")


    # Set cdf scores randomly
    else:
        cdf_scores = np.random.rand(len(dataset["difficulty"]))

    return cdf_scores
