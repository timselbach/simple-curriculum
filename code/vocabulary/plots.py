import os

import torch
import torchtext
import torchdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Tuple


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_distribution_by_complexity(X, y, metric_names, dataset_name = "SimpleWiki",output_dir=None, label_mapping=None, bins=10, hist_type='poly', x_limits=None, display_plot=True):
    """
    Creates histograms for each metric, showing the distribution by class label.

    Args:

        X : Features array of shape (n_samples, n_features)
        y : Labels array of length n_samples
        metric_names : List of metric names corresponding to columns in X
        output_dir : Directory path to save the plots
        label_mapping :  Dictionary mapping label values to class names
        bins : Number of histogram bins
        hist_type : Type of histogram element to use
        x_limits : Dictionary mapping metric names to x-axis limits as (lower, upper) tuples
        display_plot :  If True displays the plot

    Returns:
        None
    """

    # Create a DataFrame from X
    metrics_df = pd.DataFrame(X, columns=metric_names)
    metrics_df['class_label'] = y

    # Map labels to class names if provided
    if label_mapping is not None:
        metrics_df['class_label'] = metrics_df['class_label'].map(label_mapping)
        # Handle unmapped labels by assigning a default value
        metrics_df['class_label'] = metrics_df['class_label'].fillna('Unknown')
        if 'Unknown' in metrics_df['class_label'].values:
            class_order = list(label_mapping.values()) + ['Unknown']
        else:
            class_order = list(label_mapping.values())
    else:
        class_order = sorted(metrics_df['class_label'].unique())

    metrics_df['class_label'] = pd.Categorical(metrics_df['class_label'], categories=class_order, ordered=True)


    if dataset_name == "SimpleGerman":
        class_order = ["Simple (LS)","Everyday","Simple (ES)"]


    # Iterate over each metric
    for metric in metric_names:
        metric_data = metrics_df[[metric, 'class_label']].dropna()

        if metric_data.empty:
            print(f"No valid data for metric '{metric}'. Skipping plot.")
            continue

        # Check if the metric is numeric
        if metric_data[metric].dtype.kind not in ['f', 'i']:
            print(f"Metric '{metric}' is non-numeric, skipping histogram.")
            continue

        # Create the figure
        plt.figure(figsize=(12, 6))

        sns.set_theme('talk')
        sns.set_context('talk')

        # Create the histogram plot using seaborn
        sns.histplot(
            data=metric_data,
            x=metric,
            hue='class_label',
            hue_order=class_order,
            element=hist_type,
            kde=False,
            alpha=0.5,
            bins=bins,
            legend=True
        )

        # Apply x-axis limits if specified
        if x_limits and metric in x_limits:
            lower, upper = x_limits[metric]
            if lower is not None and upper is not None:
                plt.xlim(lower, upper)
            elif lower is not None:
                plt.xlim(left=lower)
            elif upper is not None:
                plt.xlim(right=upper)

        # Optionally save the plot if an output directory is provided
        if output_dir is not None:
            pdf_filename = os.path.join(output_dir, f"{metric}.pdf")
            plt.savefig(pdf_filename)
            print(f"Plot for metric '{metric}' saved as {pdf_filename}")

        # Display the plot if requested
        if display_plot:
            plt.tight_layout()
            plt.show()

        # Close the figure to free up memory
        plt.close()


def plot_metric_correlation_heatmap(X, metric_names, corr_method='pearson', figsize=(10, 8), annot=True,
                                    cmap='coolwarm'):
    """
    Creates a seaborn heatmap of the correlation matrix computed between each metric.

    Args:

        X : Features array of shape (n_samples, n_features)
        metric_names : List of metric names corresponding to columns in X
        corr_method : Correlation method to use (default is 'pearson')
        figsize : Size of the figure for the heatmap (default is (10, 8))
        annot : Whether to annotate the heatmap with correlation values (default is True)
        cmap : Color map to use for the heatmap (default is 'coolwarm').

    Returns:
        None
    """
    # Create a DataFrame from X using the provided metric names
    #metrics_df = pd.DataFrame(X, columns=metric_names)
    desired_indices = [0, 1, 2, 3, 4, 11]
    metrics_df = pd.DataFrame(X[:, desired_indices], columns=["length", "uni_entropy", "bi_entropy", "tri_entropy", "word_rarity", "flesch_kincaid"])

    # Compute the correlation matrix using the specified method
    corr_matrix = metrics_df.corr(method=corr_method)

    sns.set_theme('talk')
    sns.set_context('talk')

    # Create a figure for the heatmap
    plt.figure(figsize=figsize)

    # Plot the heatmap using seaborn
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, square=True, linewidths=0.5, linecolor='white', fmt=".2f")


    # Adjust layout and display the heatmap
    plt.tight_layout()
    plt.savefig("/home/iailab34/selbacht0/Sync/results/figures/corr_matrix.pdf")
    plt.show()
    plt.close()

