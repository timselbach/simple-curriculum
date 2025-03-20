
import copy
import math
import os
import torch
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity as TextPerplexity
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from code.curriculum.config import TRAINING_STRATEGY
from code.curriculum.create_paths import get_current_training_params
from code.curriculum.create_paths import get_save_paths
from code.curriculum.training.utils import competence_function


def evaluate(model, device, tokenizer, val_dataset, batch_size=32):
    """
    Evaluate the model on the validation dataset using TorchMetrics' TextPerplexity


    Returns:
        validation_loss: The average loss on the validation set
        perplexity: The computed perplexity using TorchMetrics' TextPerplexity
        accuracy: The fraction of correctly predicted masked tokens
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling
    import torch
    import math

    # Set model to eval mode
    model.eval()


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    # TorchMetrics perplexity
    ppl_metric = TextPerplexity(ignore_index=-100).to(device=device)

    total_loss = 0.0
    total_steps = 0


    # For accuracy
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in val_dataloader:

            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            if torch.isnan(loss):
                continue  # skip NaN losses

            total_loss += loss.item()
            total_steps += 1

            # Only valid for masked positions, ignoring `-100`
            ppl_metric.update(logits, labels)

            # compute accuracy for masked positions
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct_predictions += (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

    # Average loss
    avg_loss = total_loss / total_steps if total_steps > 0 else float('inf')

    # TorchMetrics perplexity
    perplexity = ppl_metric.compute().item()

    # Accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return avg_loss, perplexity, accuracy



def train_phase_competence(model, device, tokenizer, dataset, max_steps, update_every, batch_size, learning_rate, warmup_steps, c0,
                           val_dataset, max_t_steps):
    """
    Training loop for a competence-based curriculum learning approach

    Parameters:
    - model: The pre-initialized model
    - device: The device to train on
    - tokenizer: The tokenizer being used
    - dataset: The dataset to train on
    - max_steps: Maximum number of training steps
    - update_every: Number of steps after which to update competence and dataset
    - batch_size: Training batch size
    - learning_rate: Learning rate for the optimizer
    - c0: Initial competence level
    - val_dataset: validation dataset
    - max_t_steps: Maximum number of training steps where competence gets updated
    """

    log_dir = '/home/iailab34/selbacht0/Sync/results/logging'


    hyperparams = get_current_training_params(TRAINING_STRATEGY)
    log_path,_ = get_save_paths(log_dir,log_dir,hyperparams)
    log_path = log_path+".csv"
    print("log_path: ",log_path)

    #prepare logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize or open the log file in append mode
    # write header
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a")
    if write_header:
        log_file.write("step,train_loss,val_loss,val_perplexity,val_accuracy\n")


    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Assuming 'cdf_score' is present in the dataset
    difficulties = dataset['cdf_score']

    pbar = tqdm(ncols=150, total=max_steps, desc="Training Progress")

    model.train()
    global_step = 0
    current_competence = competence_function(global_step, max_t_steps, c0=c0)

    # Initialize DataLoader with initial competence level
    indices = [i for i, d in enumerate(difficulties) if d <= current_competence]
    if not indices:
        indices = list(range(len(dataset)))  # Use all data as fallback

    subset_dataset = dataset.select(indices)
    print("Length of subset dataset: ",len(subset_dataset))
    train_dataloader = DataLoader(subset_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    data_iter = iter(train_dataloader)

    # patience based early stopping
    patience = 3
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Re-initialize data_iter
            data_iter = iter(train_dataloader)
            batch = next(data_iter)


        # Move batch to device
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)


        # Forward pass
        outputs = model(input_ids=inputs,attention_mask=attention_mask,labels=labels)
        loss = outputs.loss

        # Check for NaN loss if no tokens got masked
        if torch.isnan(loss):
            continue  # Skip this batch

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Update global step
        global_step += 1


        # Update competence and reload data every 'update_every' steps if current_competence is smaller 1
        if global_step > 0 and global_step % update_every == 0 and current_competence < 1:

            val_loss, val_perplexity, val_accuracy = evaluate(model, device, tokenizer, val_dataset)
            print(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")

            # New best model validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"Validation loss did not improve for {patience} consecutive checks. Reverting and stopping training.")
                    model.load_state_dict(best_model_state)
                    pbar.close()
                    break


            current_train_loss = loss.item()

            # Log to file
            with open(log_path, "a") as log_file:
                log_file.write(f"{global_step},{current_train_loss},{val_loss},{val_perplexity},{val_accuracy}\n")

            # Get current competence and therefor maximum allowed difficulty
            current_competence = competence_function(global_step, max_t_steps, c0=c0)
            max_difficulty = current_competence

            # Filter dataset based on updated competence level
            indices = [i for i, d in enumerate(difficulties) if d <= max_difficulty]
            if not indices:
                indices = list(range(len(dataset)))  # Use all data as fallback

            # Create a new subset of the dataset
            subset_dataset = dataset.select(indices)
            print(f"Number of samples in subset_dataset: {len(subset_dataset)}")

            # Create a new DataLoader
            train_dataloader = DataLoader(
                subset_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
            )
            data_iter = iter(train_dataloader)

        # Monitor train loss every 100 steps
        if global_step % 100 == 0:
            current_train_loss = loss.item()
            # Log only the training loss to the CSV
            with open(log_path, "a") as log_file:
                log_file.write(f"{global_step},{current_train_loss},,\n")


        # Dont update the competence because it is already 1 and monitor validation loss every 25000 steps
        if global_step % 25000 == 0 and current_competence == 1:
            current_train_loss = loss.item()

            val_loss, val_perplexity, val_accuracy = evaluate(model, device, tokenizer, val_dataset)
            print(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")


            # Log to file
            with open(log_path, "a") as log_file:
                log_file.write(f"{global_step},{current_train_loss},{val_loss},{val_perplexity}\n")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(
                        f"Validation loss did not improve for {patience} consecutive checks. Reverting and stopping training.")
                    model.load_state_dict(best_model_state)
                    pbar.close()
                    break



        # Update tqdm progress bar
        pbar.set_postfix_str(f"Loss: {loss:.2f} | Comp: {current_competence:.4f}")
        pbar.update(1)

    # Close the progress bar
    pbar.close()
    print(f"Logging saved at:{log_path}")
    print("Training phase completed.")
    return


def train_phase_sequential(model, device, tokenizer, datasets, config, val_dataset):
        """
        Train the model with a sequential curriculum learning approach

        Args:
            model: The BERT model to train
            device: The device to train on
            tokenizer: The tokenizer
            datasets: Tokenized datasets that can be seperated by levels
            config: Training configuration parameters
            val_dataset: validation dataset
        """


        log_dir = '/home/iailab34/selbacht0/Sync/results/logging'

        hyperparams = get_current_training_params(TRAINING_STRATEGY)
        log_path, _ = get_save_paths(log_dir, log_dir, hyperparams)
        log_path = log_path + ".csv"
        print("log_path: ", log_path)

        # prepare logging
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)


        # Initialize or open the log file in append mode
        # write header
        write_header = not os.path.exists(log_path)
        log_file = open(log_path, "a")
        if write_header:
            log_file.write("level,step,train_loss,val_loss,val_perplexity,val_accuracy\n")

        update_every = config.get('update_every')
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 5e-5))

        global_step = 0

        # patience based early stopping
        patience = 3
        best_val_loss = float('inf')
        best_model_state = copy.deepcopy(model.state_dict())

        for level, steps in config['training_steps_per_level'].items():
            patience_counter = 0

            if level not in datasets:
                print(f"Level '{level}' not found in datasets. Skipping...")
                continue

            dataset = datasets[level]
            print("Length current dataset: ",len(dataset))
            dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True, collate_fn=data_collator)
            dataloader_iter = iter(dataloader)

            model.train()
            pbar = tqdm(ncols=150,total=steps, desc=f"Training on level: {level}")


            for step in range(steps):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    # Restart the iterator if the dataset is exhausted
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)


                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Check for NaN loss
                if torch.isnan(loss):
                    continue  # Skip this batch

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1


                if step % update_every == 0:
                    val_loss, val_perplexity,val_accuracy = evaluate(model, device, tokenizer, val_dataset)
                    print(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")
                    current_train_loss = loss.item()
                    # Log to file
                    with open(log_path, "a") as log_file:
                        log_file.write(f"{level},{global_step},{current_train_loss},{val_loss},{val_perplexity},{val_accuracy}\n")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += 1
                        print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

                        if patience_counter >= patience:
                            print(
                                f"Validation loss did not improve for {patience} consecutive checks. Reverting and stopping training.")
                            model.load_state_dict(best_model_state)
                            pbar.close()
                            break

                if global_step % 100 == 0:
                    current_train_loss = loss.item()
                    # Log only the training loss to the CSV
                    with open(log_path, "a") as log_file:
                        log_file.write(f"{level},{global_step},{current_train_loss},,\n")

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            pbar.close()

            # evaluate at the end of phase
            val_loss, val_perplexity, val_accuracy = evaluate(model, device, tokenizer, val_dataset)
            current_train_loss = loss.item()
            print(f"** End-of-phase Validation **: Level={level}, Loss={val_loss}, Perplexity={val_perplexity}, Accuracy={val_accuracy}")
            # Optionally log it
            with open(log_path, "a") as log_file:
                log_file.write(f"{level},{global_step},{current_train_loss},{val_loss},{val_perplexity},{val_accuracy}\n")


            print(f"Logging saved at:{log_path}")
            print(f"Completed training on level: {level}")

        return

from datasets import concatenate_datasets

def train_phase_incremental(model, device, tokenizer, datasets, config, val_dataset):
    """
        Train the model with an incremental curriculum learning approach

        Args:
            model: The BERT model to train
            device: The device to train on
            tokenizer: The tokenizer
            datasets: Tokenized datasets that can be seperated by levels
            config: Training configuration parameters
            val_dataset: validation dataset

    """
    log_dir = '/home/iailab34/selbacht0/Sync/results/logging'


    hyperparams = get_current_training_params(TRAINING_STRATEGY)
    log_path,_ = get_save_paths(log_dir,log_dir,hyperparams)
    log_path = log_path+".csv"
    print("log_path: ",log_path)


    update_every = config.get('update_every')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    # Initialize log file
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a")
    if write_header:
        log_file.write("level,step,train_loss,val_loss,val_perplexity,val_accuracy\n")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', 5e-5))

    global_step = 0

    # patience based approach
    patience = 3
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    # Initialize an empty cumulative dataset
    cumulative_dataset = None

    for level, steps in config['training_steps_per_level'].items():
        patience_counter = 0
        if level not in datasets:
            print(f"Level '{level}' not found in datasets. Skipping...")
            continue

        # Concatenate current level dataset with cumulative dataset
        if cumulative_dataset is None:
            # First level, just set it
            cumulative_dataset = datasets[level]
        else:
            # Concatenate with previously accumulated data
            cumulative_dataset = concatenate_datasets([cumulative_dataset, datasets[level]])
        print("Length current dataset: ", len(cumulative_dataset))
        # Create a new DataLoader for the cumulative dataset
        dataloader = DataLoader(cumulative_dataset,
                                batch_size=config.get('batch_size', 32),
                                shuffle=True,
                                collate_fn=data_collator)
        dataloader_iter = iter(dataloader)

        model.train()
        pbar = tqdm(ncols=150, total=steps, desc=f"Training on level: {level}")



        for step in range(steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # Re-iterate over the cumulative dataset if exhausted
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Check for NaN loss
            if torch.isnan(loss):
                # If NaN, skip this batch
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1


            # Periodically evaluate on the validation set
            if step % update_every == 0:
                val_loss, val_perplexity, val_accuracy = evaluate(model, device, tokenizer, val_dataset)
                print(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")
                current_train_loss = loss.item()
                with open(log_path, "a") as log_file:
                    log_file.write(f"{level},{global_step},{current_train_loss},{val_loss},{val_perplexity},{val_accuracy}\n")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        print(
                            f"Validation loss did not improve for {patience} consecutive checks. Reverting and stopping training.")
                        model.load_state_dict(best_model_state)
                        pbar.close()
                        break


            if global_step % 100 == 0:
                current_train_loss = loss.item()
                # Log only the training loss to the CSV
                with open(log_path, "a") as log_file:
                    log_file.write(f"{level},{global_step},{current_train_loss},,\n")

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

        #evaluate at the end of phase
        val_loss, val_perplexity, val_accuracy = evaluate(model, device, tokenizer, val_dataset)
        current_train_loss = loss.item()
        print(f"** End-of-phase Validation **: Level={level}, Loss={val_loss}, Perplexity={val_perplexity}")


        pbar.close()
        print(f"Logging saved at:{log_path}")
        print(f"Completed training on level: {level}")


    return