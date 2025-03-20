import os
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

SAVE_DIR = Path("/home/iai/user/toborek/projects/informed-curriculum/plots/classification")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)


def plot_cm(y_test, y_hat, labels, title=None, save=False):
    cm = confusion_matrix(y_test, y_hat, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    if title:
        plt.title(title)
    if save:
        print(f"saving to: {(SAVE_DIR / f'{save}.pdf').resolve()}")
        fig = plt.gcf()
        fig.savefig(SAVE_DIR / f"{save}.pdf")
    plt.show()


def plot_boxplots_from_cv_results(results: dict, scores: str = None, title=None, save=False):
    """
    Method to plot the effects of hyperparameter search of a given scikit learn model. Will plot the effect of each
    parameters on one subplot averaging over all other hyperparameters.
    Args:
        results: dict from the gridsearch containing all different models under the key "estimator"
        score: one score to choose from
        title:
        save:

    Returns:

    """
    all_cv_results = []
    for e in results["estimator"]:
        all_cv_results.append(pd.DataFrame.from_dict(e.cv_results_))

    score = "F1" if not scores else scores
    evals = {"Accuracy": "mean_test_acc", "Precision": "mean_test_prec", "Recall": "mean_test_rec", "F1": "mean_test_f1"}
    assert score in evals.keys(), f"Choose score from {evals.keys()}, not {score}."
    all_params = [col for col in all_cv_results[0] if col.startswith("param_")]
    all_cols = all_params + [evals[score]]
    merged_results = pd.concat(all_cv_results)
    fig, axes = plt.subplots(len(all_params), 1, figsize=(8, 4*len(all_params)))
    for i, param in enumerate(all_params):
        grouped_results = merged_results[all_cols].groupby(param)
        group_dict = {k: grouped_results.get_group(k).values[:, i+1] for k in grouped_results.groups.keys()}
        if len(all_params) == 1:
            axes.boxplot(group_dict.values(), showmeans=True, patch_artist=True, boxprops=dict(facecolor="white"),
                            meanprops=dict(color="green", marker="x"))
            axes.set_xticklabels(group_dict.keys())
            axes.set(title=f"Effect of {param[len('param_'):]} on {score}", ylabel=f"{score}")
        else:
            axes[i].boxplot(group_dict.values(), showmeans=True, patch_artist=True, boxprops=dict(facecolor="white"),
                       meanprops=dict(color="green", marker="x"))
            axes[i].set_xticklabels(group_dict.keys())
            axes[i].set(title=f"Effect of {param[len('param_'):]} on {score}", ylabel=f"{score}")
    if title:
        fig.suptitle(title)
    if save:
        fig = plt.gcf()
        fig.savefig(SAVE_DIR / f"{save}.pdf")
    median_line = plt.Line2D([], [], color='orange', linestyle='-', label='Median')
    mean_marker = plt.Line2D([], [], color='green', linestyle='', marker='x', label='Mean')
    plt.legend(handles=[median_line, mean_marker])
    plt.tight_layout()
    plt.show()


def plot_tsne_comparison(X_val, y_val, predictions, filter=None, title=None, save=None):
    """
    Function to plot tsne dimensionality reduction for the given data points with their original labels and their
    predicted labels.
    Args:
        X_val:
        y_val:
        predictions:
        filter: filter data points wrt to class only for scatter plot, not for the dim reduction
        title:
        save:

    Returns:

    """
    p = sns.color_palette("colorblind")
    labels = [1, 2, 3]
    assert set(y_val).issubset(labels), "The labels of y_val are not in [1, 2, 3]"

    my_palette = {1: p[0], 2: p[1], 3: p[4]}
    tsne = TSNE()
    reduced = tsne.fit_transform(X_val)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    if filter:
        pass
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=y_val, palette=my_palette, ax=ax[0], legend=False, s=30, )
    ax[0].set(title="Dataset with original labels", ylabel="tSNE_2", xlabel="tSNE_1")

    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=predictions, palette=my_palette, ax=ax[1], legend=True,
                    s=30, )
    ax[1].set(title="Dataset with predicted labels", xlabel="tSNE_1")
    # handles, labels = ax[1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", ncol=3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save:
        fig.savefig(SAVE_DIR / f"{save}.pdf")
    plt.show()