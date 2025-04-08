import os
from datetime import datetime
from pathlib import Path

from simple_project.curriculum.config import DATASET_NAME, DIFFICULTY_METRIC, RANDOM_CDF


def sanitize_hyperparam_value(value):
    """
    Change hyperparameter values to be filesystem-friendly
    """
    if isinstance(value, float):
        return f"{value:.5f}".replace('.', 'p')  # Example: 0.001 -> 0p001
    elif isinstance(value, str):
        return value.replace('/', '_').replace('\\', '_')  # Replace slashes with underscore
    return str(value)


def abbreviate_hyperparams(hyperparams):
    """shorten hyperparameter keys to reduce path length"""

    abbreviations = {
        "batch_size": "bs",
        "learning_rate": "lr",
        "max_steps_phase": "msph",
        "update_every": "ue",
        # Add more abbreviations as necessary
    }
    abbreviated = {abbreviations.get(k, k): v for k, v in hyperparams.items()}
    return abbreviated


def shorten_timestamp(fmt="%Y%m%d_%H%M%S"):
    """Generate a shortened timestamp"""
    return datetime.now().strftime(fmt)


def compress_number(num: int) -> str:
    """
    Converts large integer to a more compact representation:
      1 -> "1"
      50_000 -> "50k"
      450_000 -> "450k"
    """
    if num >= 1000:
        return f"{num // 1000}k"
    return str(num)


def format_steps_compact(steps_dict: dict) -> str:
    """
    Given a dict of training steps per level, e.g. {1: 1, 2: 50000, 3: 450000},
    return a compact string like "steps1_50k_450k"
    """

    step_strs = [compress_number(steps_dict[level]) for level in steps_dict.keys()]
    # Example output: "steps1_50k_450k"
    return "steps" + "_".join(step_strs)


def get_save_paths(base_model_path: str,
                   base_tokenizer_path: str,
                   hyperparams: dict) -> tuple:
    """
    Generates unique save paths for tho model and tokenizer according to the given hyperparameters

    Args:
        base_model_path: base path to model
        base_tokenizer_path: base path to tokenizer
        hyperparams: hyperparameter dictionary

    Returns:
     tuple: (model_save_path_str, tokenizer_save_path_str)
    """

    # grab the training type and dataset name
    training_type = hyperparams.get("type", "unknown_type")
    dataset_name = DATASET_NAME


    # format hyperparameter keys and values

    abbreviated = abbreviate_hyperparams(hyperparams)

    sanitized_hyperparams = {}
    for k, v in abbreviated.items():
        if k == "training_steps_per_level" and isinstance(v, dict):
            # If this hyperparameter is the 'training_steps_per_level' dict,
            # we’ll create a single compact string
            sanitized_hyperparams[k] = format_steps_compact(v)
        else:
            sanitized_hyperparams[k] = sanitize_hyperparam_value(v)


    # build a string of hyperparams and skip type

    parts = []
    for k, val in sanitized_hyperparams.items():
        if k == "type":
            continue
        parts.append(f"{k}{val}")

    # join them with underscores, e.g. "bs8_lr0p0001_steps1_50k_450k_ue25000"
    hyperparams_str = "_".join(parts)

    # If training type is competence, include DIFFICULTY_METRIC in the name.
    timestamp = shorten_timestamp()
    if training_type == "competence":
        # Add the difficulty metric to the directory name
        dir_name = f"{training_type}_{dataset_name}_{DIFFICULTY_METRIC}_{hyperparams_str}_{timestamp}"
        if RANDOM_CDF == True:
            dir_name = "random_"+dir_name
    else:
        dir_name = f"{training_type}_{dataset_name}_{hyperparams_str}_{timestamp}"


    model_save_path = os.path.join(base_model_path, dir_name)
    tokenizer_save_path = os.path.join(base_tokenizer_path, dir_name)

    return model_save_path, tokenizer_save_path


def get_current_training_params(training_strategy: dict) -> dict:
    """
    Returns the parameters for the current training strategy
    """
    strategy_type = training_strategy.get('type')
    if not strategy_type:
        raise ValueError("TRAINING_STRATEGY must have a 'type' key specifying the strategy.")

    params = training_strategy.get(strategy_type)
    if not params:
        raise ValueError(f"No parameters found for the strategy type '{strategy_type}'.")

    # Convert params into a dict and inject the 'type' key
    params_with_type = dict(params)
    params_with_type["type"] = strategy_type
    return params_with_type
