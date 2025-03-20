
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plot_curriculum_loss_smooth(log_file_path, rolling_window=80):
    """
    Plots smoothed training loss (via rolling average) and validation loss
    curves from a training log file

    Args:
        log_file_path: Path to the curriculum training log CSV file
        rolling_window: Window size for rolling average to smooth train loss
    """
    if not os.path.exists(log_file_path):
        print(f"Log file {log_file_path} does not exist")
        return


    # Read the log file
    df = pd.read_csv(log_file_path)

    # Apply a rolling mean to smooth the training loss
    df['train_loss_smooth'] = df['train_loss'].rolling(window=rolling_window, min_periods=1).mean()
    print(df['val_loss'].dropna().head(10))
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['train_loss_smooth'], label="Train Loss (Smoothed)")
    valid_val_loss_df = df.dropna(subset=['val_loss'])
    plt.plot(valid_val_loss_df['step'], valid_val_loss_df['val_loss'], label='Validation Loss')

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Curriculum Training: Smoothed Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_curriculum_loss(log_file_path):
    """
    Plots training and validation loss curves from a curriculum training log file without smoothing

    Args:
        log_file_path: Path to the curriculum training log CSV file.
    """
    if not os.path.exists(log_file_path):
        print(f"Log file {log_file_path} does not exist.")
        return

    # Read the log file
    df = pd.read_csv(log_file_path)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['train_loss'], label="Train Loss")
    plt.plot(df['step'], df['val_loss'], label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)
    plt.show()



base_path = "/home/iailab34/selbacht0/Sync/results/logging/"

#sequential swiki
# curr_paths = [
#    base_path+"sequential_SimpleWiki_training_steps_per_levelsteps250k_250k_bs8_lr0p00010_ue20000_20250227_101831.csv",
#    base_path+"sequential_SimpleWiki_training_steps_per_levelsteps250k_250k_bs8_lr0p00010_ue20000_20250227_101857.csv",
#    base_path+"sequential_SimpleWiki_training_steps_per_levelsteps250k_250k_bs8_lr0p00010_ue20000_20250227_102346.csv"
# ]



# #incremental swiki
# curr_paths = [
#      base_path+"incremental_SimpleWiki_training_steps_per_levelsteps250k_250k_bs8_lr0p00010_ue10000_20250218_194509.csv",
#      base_path+"incremental_SimpleWiki_training_steps_per_levelsteps250k_250k_bs8_lr0p00010_ue10000_20250218_194735.csv",
#      base_path+"incremental_SimpleWiki_training_steps_per_levelsteps250k_250k_bs8_lr0p00010_ue10000_20250218_194727.csv"
#  ]
#
#competence swiki
# curr_paths = [
#     base_path +"curriculum_SimpleWiki_word_rarity_lr0.0001_bs8_c00.05_50000_20250130_220019.csv",
#     base_path +"curriculum_SimpleWiki_word_rarity_lr0.0001_bs8_c00.05_50000_20250130_220058.csv",
#     base_path +"curriculum_SimpleWiki_word_rarity_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250218_124319.csv"
# ]

# curr_paths = [
#    base_path+"curriculum_SimpleWiki_length_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250227_123251.csv",
#    base_path+"curriculum_SimpleWiki_length_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250227_131341.csv",
#    base_path+"curriculum_SimpleWiki_length_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250227_123419.csv"
# ]


#
#baseline swiki
# base_paths = [
#     base_path+"curriculum_SimpleWiki_length_lr0.0001_bs8_c01_50000_20250213_172557.csv",
#     base_path+"curriculum_SimpleWiki_length_lr0.0001_bs8_c01_50000_20250213_172630.csv",
#     base_path+"curriculum_SimpleWiki_length_lr0.0001_bs8_c01_50000_20250213_172702.csv"
# ]
#
#
# #sequential sgerman
curr_paths = [
    base_path+"sequential_SimpleGerman_lr0.0001_bs8_100k_100k_300k_20250128_104320.csv",
    base_path+"sequential_SimpleGerman_lr0.0001_bs8_100k_100k_300k_20250128_104313.csv",
    base_path+"sequential_SimpleGerman_lr0.0001_bs8_100k_100k_300k_20250128_104327.csv"
]
#
# #incremental sgerman
# curr_paths = [
#     base_path+"incremental_SimpleGerman_training_steps_per_levelsteps50k_150k_300k_bs8_lr0p00010_ue10000_20250319_121358.csv",
#     base_path+"incremental_SimpleGerman_training_steps_per_levelsteps50k_150k_300k_bs8_lr0p00010_ue10000_20250319_121338.csv",
#     base_path+"incremental_SimpleGerman_training_steps_per_levelsteps50k_150k_300k_bs8_lr0p00010_ue10000_20250319_121408.csv"
# ]
#
# #competence sgerman
# curr_paths = [
#     base_path+"curriculum_SimpleGerman_length_lr0.0001_bs8_c00.05_50000_20250130_161541.csv",
#     base_path+"curriculum_SimpleGerman_length_lr0.0001_bs8_c00.05_50000_20250206_095726.csv",
#     base_path+"curriculum_SimpleGerman_length_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250218_194646.csv"
# ]

# curr_paths = [
#     base_path + "curriculum_SimpleGerman_word_rarity_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250227_160845.csv",
#     base_path + "curriculum_SimpleGerman_word_rarity_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250227_160914.csv",
#     base_path + "curriculum_SimpleGerman_word_rarity_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250227_160926.csv"
# ]

#
# #baseline sgerman
base_paths = [
    base_path+"curriculum_SimpleGerman_length_bs8_lr0p00010_msph500000_ue20000_c01_max_t_steps50000_20250217_195554.csv",
    base_path+"curriculum_SimpleGerman_length_bs8_lr0p00010_msph500000_ue5000_c01_max_t_steps50000_20250219_104109.csv",
    base_path+"curriculum_SimpleGerman_length_bs8_lr0p00010_msph500000_ue5000_c01_max_t_steps50000_20250219_104218.csv"
]

def plot_average_runs(
    curriculum_paths,
    baseline_paths,
    rolling_window=80,
    legend_font_size=14,
    label_font_size=14,
    tick_font_size=12,
    label1="",
    label2="Baseline",
    separate_curriculum=False,  # New option: if True, plot each curriculum run separately
):
    """
    Reads multiple CSV log files for two models (Curriculum and Baseline),
    averages the smoothed train loss and validation loss across runs and plots them

    Args:
        curriculum_paths: List of file paths for the Curriculum model runs
        baseline_paths: List of file paths for baseline model runs
        rolling_window: Window size for smoothing the training loss
        legend_font_size: Font size for the legend
        label_font_size: Font size for the x and y labels
        tick_font_size: Font size for the tick labels
        label1: Label prefix for the Curriculum model
        label2: Label prefix for the Baseline model
        separate_curriculum: If True, plots each curriculum run separately
    """
    def load_and_process(paths):
        dfs = []
        for path in paths:
            print(f"{path}:")
            if not os.path.exists(path):
                print(f"Log file {path} does not exist.")
                continue
            df = pd.read_csv(path)
            print(f"{path}:")
            print(df[['step', 'train_loss', 'val_loss']].head())

            # Compute the rolling average for train loss
            df['train_loss_smooth'] = df['train_loss'].rolling(window=rolling_window, min_periods=1).mean()
            # Keep only the relevant columns: step, train_loss_smooth, and val_loss
            dfs.append(df[['step', 'train_loss_smooth', 'val_loss']])
        if not dfs:
            return None
        #concatenate and group by 'step' to average across runs
        combined = pd.concat(dfs, axis=0)
        avg_df = combined.groupby('step', as_index=False).mean()
        return avg_df

    plt.figure(figsize=(10, 6))

    # curriculum model plotting
    if separate_curriculum:
        # plot each curriculum run separately
        for i, path in enumerate(curriculum_paths):
            if not os.path.exists(path):
                print(f"Log file {path} does not exist.")
                continue
            df = pd.read_csv(path)
            df['train_loss_smooth'] = df['train_loss'].rolling(window=rolling_window, min_periods=1).mean()
            # Define labels for each run
            run_label = label1 if label1 else "Curriculum"
            label_train = f"{run_label} Run {i+1} Train Loss"
            label_val = f"{run_label} Run {i+1} Validation Loss"
            plt.plot(
                df['step'],
                df['train_loss_smooth'],
                label=label_train,
                color="#006ec7",
                alpha=0.5
            )
            valid_val = df['val_loss'].notna()
            plt.plot(
                df.loc[valid_val, 'step'],
                df.loc[valid_val, 'val_loss'],
                label=label_val,
                color="#006ec7"
            )
    else:
        # Average curriculum runs across CSVs
        df_curr = load_and_process(curriculum_paths)
        if df_curr is None:
            print("Not enough valid data to plot for curriculum.")
            return
        plt.plot(
            df_curr['step'],
            df_curr['train_loss_smooth'],
            label=label1 + " Train Loss" if label1 else "Curriculum Train Loss",
            color="#006ec7",
            alpha=0.5
        )
        valid_val_curr = df_curr['val_loss'].notna()
        plt.plot(
            df_curr.loc[valid_val_curr, 'step'],
            df_curr.loc[valid_val_curr, 'val_loss'],
            label=label1 + " Validation Loss" if label1 else "Curriculum Validation Loss",
            color="#006ec7"
        )

    # Baseline model: plot averaged baseline data
    df_base = load_and_process(baseline_paths)
    if df_base is None:
        print("Not enough valid data to plot for baseline.")
        return
    plt.plot(
        df_base['step'],
        df_base['train_loss_smooth'],
        label=label2 + " Train Loss",
        color="#ec7931",
        alpha=0.5
    )
    valid_val_base = df_base['val_loss'].notna()
    plt.plot(
        df_base.loc[valid_val_base, 'step'],
        df_base.loc[valid_val_base, 'val_loss'],
        label=label2 + " Validation Loss",
        color="#ec7931"
    )

    plt.xlabel("Step", fontsize=label_font_size)
    plt.ylabel("Loss", fontsize=label_font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/iailab34/selbacht0/Sync/results/figures/loss_curve.pdf")
    plt.show()



plot_average_runs(curr_paths, base_paths,label1="Sequential",rolling_window=80,separate_curriculum=False)

def plot_comparison_loss(
    log_file_path_1,
    log_file_path_2,
    model_name_1="Curriculum",
    model_name_2="Baseline",
    rolling_window=80,
    legend_font_size=10,
    label_font_size=12,
    tick_font_size=10
):
    """
    Plots training and validation loss curves of two single model runs

    Args:
        log_file_path_1: Path to the first model's log CSV file
        log_file_path_2: Path to the second model's log CSV file
        model_name_1: Name/label for the first model
        model_name_2: Name/label for the second model
        rolling_window: Window size for rolling average of train loss
        legend_font_size: Font size for the legend
        label_font_size: Font size for the x and y labels
        title_font_size: Font size for the plot title
        tick_font_size: Font size for the tick labels
    """

    #check if file paths exist
    if not os.path.exists(log_file_path_1):
        print(f"Log file {log_file_path_1} does not exist.")
        return
    if not os.path.exists(log_file_path_2):
        print(f"Log file {log_file_path_2} does not exist.")
        return

    # read the log files
    df1 = pd.read_csv(log_file_path_1)
    df2 = pd.read_csv(log_file_path_2)

    # smooth the train losses using a rolling window
    df1['train_loss_smooth'] = df1['train_loss'].rolling(window=rolling_window, min_periods=1).mean()
    df2['train_loss_smooth'] = df2['train_loss'].rolling(window=rolling_window, min_periods=1).mean()

    # remove NaN values from val_loss so we don't plot missing points
    df1_val = df1.dropna(subset=['val_loss'])
    df2_val = df2.dropna(subset=['val_loss'])

    plt.figure(figsize=(10, 6))

    plt.plot(df1['step'], df1['train_loss_smooth'], label=f"{model_name_1} Train Loss (Smoothed)",color = "#006ec7")
    plt.plot(df1_val['step'], df1_val['val_loss'], label=f"{model_name_1} Validation Loss",color = "#006ec7", alpha=0.5)

    plt.plot(df2['step'], df2['train_loss_smooth'], label=f"{model_name_2} Train Loss (Smoothed)", color = "#ec7931")
    plt.plot(df2_val['step'], df2_val['val_loss'], label=f"{model_name_2} Validation Loss", color = "#ec7931", alpha=0.5)

    # Set font sizes for labels, title, and ticks
    plt.xlabel("Step", fontsize=label_font_size)
    plt.ylabel("Loss", fontsize=label_font_size)
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.grid(True)
    plt.savefig("loss_comparison_curr_wiki.pdf", format="pdf", bbox_inches="tight")
    plt.show()




path1 = base_path+"curriculum_SimpleWiki_length_lr0.0001_bs8_c01_50000_20250213_172557.csv"
#path2 = base_path+"curriculum_SimpleWiki_word_rarity_lr0.0001_bs8_c00.05_50000_20250130_220019.csv"
path2 = base_path+"curriculum_SimpleWiki_length_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250220_120231.csv"

#path1 = base_path+"curriculum_SimpleGerman_length_bs8_lr0p00010_msph500000_ue20000_c01_max_t_steps50000_20250217_195554.csv"
#path2 = base_path+"curriculum_SimpleGerman_length_bs8_lr0p00010_msph500000_ue5000_c00p05000_max_t_steps50000_20250218_194646.csv"

#plot_comparison_loss(path2,path1,rolling_window=80,legend_font_size=10,label_font_size=12)




def plot_comparison_loss_seaborn(
        log_file_path_1,
        log_file_path_2,
        model_name_1="Baseline",
        model_name_2="Curriculum",
        rolling_window=80
):
    """
    Plots training and validation loss curves of two single model runs using Seaborn

    Args:
        log_file_path_1: Path to the first model's log CSV file
        log_file_path_2: Path to the second model's log CSV file
        model_name_1: Name/label for the first model
        model_name_2: Name/label for the second model
        rolling_window: Window size for rolling average of train loss
    """
    # check if file paths exist
    if not os.path.exists(log_file_path_1):
        print(f"Log file {log_file_path_1} does not exist.")
        return
    if not os.path.exists(log_file_path_2):
        print(f"Log file {log_file_path_2} does not exist.")
        return

    #read the log files
    try:
        df1 = pd.read_csv(log_file_path_1)
        df2 = pd.read_csv(log_file_path_2)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    required_columns = {'step', 'train_loss', 'val_loss'}
    if not required_columns.issubset(df1.columns):
        print(f"First log file is missing required columns: {required_columns - set(df1.columns)}")
        return
    if not required_columns.issubset(df2.columns):
        print(f"Second log file is missing required columns: {required_columns - set(df2.columns)}")
        return


    df1 = df1.sort_values(by='step')
    df2 = df2.sort_values(by='step')

    # smooth the train losses using a rolling window
    df1['train_loss_smooth'] = df1['train_loss'].rolling(window=rolling_window, min_periods=1).mean()
    df2['train_loss_smooth'] = df2['train_loss'].rolling(window=rolling_window, min_periods=1).mean()


    df1_val = df1.dropna(subset=['val_loss'])
    df2_val = df2.dropna(subset=['val_loss'])


    df1_train = df1[['step', 'train_loss_smooth']].copy()
    df1_train.rename(columns={'train_loss_smooth': 'Loss'}, inplace=True)
    df1_val_plot = df1_val[['step', 'val_loss']].copy()
    df1_val_plot.rename(columns={'val_loss': 'Loss'}, inplace=True)

    df2_train = df2[['step', 'train_loss_smooth']].copy()
    df2_train.rename(columns={'train_loss_smooth': 'Loss'}, inplace=True)
    df2_val_plot = df2_val[['step', 'val_loss']].copy()
    df2_val_plot.rename(columns={'val_loss': 'Loss'}, inplace=True)

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))


    sns.lineplot(
        data=df1_train,
        x='step',
        y='Loss',
        label=f"{model_name_1} - Train (Smoothed)",
        color="C0",
        alpha=0.5,
        linewidth=2
    )
    sns.lineplot(
        data=df1_val_plot,
        x='step',
        y='Loss',
        label=f"{model_name_1} - Validation",
        color="C0",
        alpha=1,
        linewidth=2
    )

    sns.lineplot(
        data=df2_train,
        x='step',
        y='Loss',
        label=f"{model_name_2} - Train (Smoothed)",
        color="C1",
        alpha=0.5,
        linewidth=2
    )
    sns.lineplot(
        data=df2_val_plot,
        x='step',
        y='Loss',
        label=f"{model_name_2} - Validation",
        color="C1",
        alpha=1,
        linewidth=2
    )

    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Loss Comparison: Smoothed Train & Validation Loss", fontsize=16)
    plt.legend(fontsize=12, title_fontsize=12)
    plt.tight_layout()

    # Save the plot as a PDF before displaying
    plt.savefig("loss_comparison_curr_wiki.pdf", format="pdf", bbox_inches="tight")

    plt.show()


