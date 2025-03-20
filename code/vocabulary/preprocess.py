
import pandas as pd
import numpy as np



def filter_by_sentence_length(X, y, sents, metric_names,label_mapping, min_length: int = 4, max_length: int = 300,
    n_print: int = 5):
    """
    Filters out sentences that are too short or too long based on precomputed 'length' metric.
    Prints the first 'n_print' removed sentences for inspection.

    Args:
        X : Features array of shape (n_samples, n_features)
        y : Labels array of length n_samples
        sents : List of sentences corresponding to the data points
        metric_names : List of metric names corresponding to columns in X
        label_mapping :  Dictionary mapping label values to class names
        min_length : Minimum allowed sentence length. Defaults to 4
        max_length : Maximum allowed sentence length. Defaults to 300
        n_print : Number of removed sentences to print. Defaults to 5

    Returns:
        tuple: Filtered X, y, and sents without sentences outside the allowed length range.
    """

    # Create a DataFrame from X
    metrics_df = pd.DataFrame(X, columns=metric_names)
    # Add the labels
    metrics_df['class_label'] = y

    # Map labels to class names if provided
    if label_mapping:
        metrics_df['class_label'] = metrics_df['class_label'].map(label_mapping)

    # Check if 'length' metric exists
    if 'length' not in metrics_df.columns:
        print("The 'length' metric is not in metric_names. Cannot filter by sentence length.")
        return X, y, sents

    # Create a mask for sentences within the acceptable length range
    length_mask = (metrics_df['length'] >= min_length) & (metrics_df['length'] <= max_length)

    # Identify removed sentences
    removed_mask = ~length_mask
    removed_sents = [s for s, rm in zip(sents, removed_mask) if rm]

    # Apply the mask
    X_filtered = X[length_mask]
    y_filtered = y[length_mask]
    sents_filtered = [s for s, keep in zip(sents, length_mask) if keep]

    num_excluded = len(X) - len(X_filtered)
    print(f"Removed {num_excluded} data point(s) due to length constraints.")
    print(f"Remaining data points: {len(X_filtered)}.")

    # Identify removed indices from the removed_mask
    removed_indices = np.where(removed_mask)[0]

    # Print a sample of removed sentences
    if num_excluded > 0:
        print(f"\nFirst {min(n_print, num_excluded)} removed sentences:")
        for i, (idx, sent) in enumerate(zip(removed_indices[:n_print], removed_sents[:n_print]), 1):
            sent_length = metrics_df.iloc[idx]['length']
            print(f"{i}. {sent} (Length: {sent_length})")
    else:
        print("No sentences were removed.")

    return X_filtered, y_filtered, sents_filtered

