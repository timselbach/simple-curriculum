import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

SAVE_DIR = Path("../plots/eval_metrics")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)


def plot_all_histograms(X_dev, X_val, metric_names, title=None, save=None):
    nrows = 3
    ncols = int(np.ceil(len(metric_names) / nrows))
    xsize = 18
    ysize = 9

    # histogram plots
    fig, axes = plt.subplots(nrows, ncols, figsize=(xsize, ysize))
    for i, data in enumerate(zip(axes.reshape(-1), metric_names)):
        ax, m = data
        ax.hist(X_dev[:, i], bins=20, color="blue", alpha=0.5, label="dev set")
        ax.hist(X_val[:, i], bins=20, color="green", alpha=0.5, label="val set")
        ax.set(title=f"{m}", ylabel="Sentence level", xlabel=f"{m}")
    fig.legend(["dev set", "val set"], loc="lower right", fontsize=14)
    if title:
        fig.suptitle(title, size=16)
    plt.tight_layout()
    if save:
        fig.savefig(SAVE_DIR / f"{save}.pdf")
    plt.show()


def plot_all_metrics_scatter(X_dev, y_dev, metric_names, title=None, save=None):
    nrows = 3
    ncols = int(np.ceil(len(metric_names) / nrows))
    xsize = 18
    ysize = 9

    fig, axes = plt.subplots(nrows, ncols, figsize=(xsize, ysize))
    for i, data in enumerate(zip(axes.reshape(-1), metric_names)):
        ax, m = data
        ax.plot(X_dev[:, i], y_dev, 'o', alpha=0.4)
        ax.set_ylim([min(y_dev)-0.15, max(y_dev)+0.15])
        ax.yaxis.set_ticks(list(set(y_dev)))
        ax.set(title=f"{m}", ylabel="Sentence level", xlabel=f"{m}")
    if title:
        fig.suptitle(title, size=16)
    plt.tight_layout()
    if save:
        fig.savefig(SAVE_DIR / f"{save}.pdf")
    plt.show()


def plot_all_metrics_boxplots(X_dev, y_dev, metric_names, save=None):
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(5, 32))
    for i, data in enumerate(zip(axes.reshape(-1), metric_names)):
        ax, m = data
        my_dict = {"1": [data[i] for j, data in enumerate(X_dev) if y_dev[j] == 1],
                   "2": [data[i] for j, data in enumerate(X_dev) if y_dev[j] == 2],
                   "3": [data[i] for j, data in enumerate(X_dev) if y_dev[j] == 3]}
        ax.boxplot(my_dict.values())
        ax.set_xticklabels(my_dict.keys())
        ax.set_ylabel(m)
    plt.tight_layout()
    if save:
        fig.savefig(SAVE_DIR / f"{save}.pdf")
    plt.show()


def plot_metric_pairplot(X, y, metric_names, title, save):
    scaler = StandardScaler()
    data = scaler.fit_transform(X)
    # add labels as column to the data matrix
    data = np.hstack((data, y.reshape(-1, 1)))
    columns = metric_names + ["level"]
    df2 = pd.DataFrame(data=data, columns=columns)

    # %%
    # tell pairplot which column corresponds to the hue
    # hue_order = [1.0, 2.0, 3.0]
    # palette = sns.color_palette("husl", n_colors=len(hue_order)).as_hex()
    p = sns.color_palette("colorblind")
    labels = [1, 2, 3]
    assert set(y).issubset(labels), "The labels of y_val are not in [1, 2, 3]"
    my_palette = {1: p[0], 2: p[1], 3: p[4]}

    sns.pairplot(df2, hue="level", palette=my_palette)
    if title:
        fig = plt.gcf()
        fig.suptitle(title, size=14, y=1.05)
    plt.tight_layout()
    if save:
        fig.savefig(SAVE_DIR / f"{save}.pdf")
    plt.show()
