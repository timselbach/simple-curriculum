# training/training_loop.py

import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling

from simple_project.curriculum.training.utils import competence_function


def train_model(model, device, tokenizer, dataset, max_steps, update_every, batch_size, learning_rate, warmup_steps,
                c0):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    print("Initialized DataCollator for MLM.")

    difficulties = dataset['cdf_score']

    model.train()
    global_step = 0
    current_competence = competence_function(global_step, max_steps, c0=c0)

    indices = [i for i, d in enumerate(difficulties) if d <= current_competence]
    if not indices:
        indices = list(range(len(dataset)))

    subset_dataset = dataset.select(indices)
    train_dataloader = DataLoader(
        subset_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    data_iter = iter(train_dataloader)

    pbar = tqdm(total=max_steps, desc="Training Progress")

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        global_step += 1

        if global_step % update_every == 0:
            current_competence = competence_function(global_step, max_steps, c0=c0)
            max_difficulty = current_competence

            indices = [i for i, d in enumerate(difficulties) if d <= max_difficulty]
            if not indices:
                indices = list(range(len(dataset)))

            subset_dataset = dataset.select(indices)
            print("Number of samples in subset_dataset:", len(subset_dataset))

            train_dataloader = DataLoader(
                subset_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
            )
            data_iter = iter(train_dataloader)

        pbar.update(1)
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Competence': f"{current_competence:.4f}"
        })

    pbar.close()
    print("Training completed.")
