# strategies/competence.py

from code.curriculum.config import TRAINING_STRATEGY
from code.curriculum.strategies.base import TrainingStrategy
from code.curriculum.training.two_phase_training import train_phase_competence


class CompetenceTraining(TrainingStrategy):
    def train(self, model, device, tokenizer, dataset,config, val_dataset):
        """
        Executes competence-based training on a combined dataset

        Args:
          model: The pre-initialized model
          device: The device to train on
          tokenizer: The tokenizer being used
          dataset: Tokenized dataset with combined language complexity
          config: Training configuration
          val_dataset: Tokenized validation dataset
        """
        print("Starting Competence-Based Training")

        curriculum_config = TRAINING_STRATEGY.get('competence', {})

        # Retrieve parameters with default values if not specified
        batch_size = curriculum_config.get('batch_size', 32)
        learning_rate = curriculum_config.get('learning_rate', 5e-5)
        warmup_steps = curriculum_config.get('warmup_steps', 500)
        epochs = curriculum_config.get('epochs', 5)


        # Define total training steps
        max_steps = curriculum_config.get('max_steps_phase', 10000)  # Example default
        update_every = curriculum_config.get('update_every', 100)  # Example default
        c0 = curriculum_config.get('c0', 1.0)  # Example default
        max_t_steps = curriculum_config.get('max_t_steps', 10000)

        print(f"Total training steps: {max_steps}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Epochs: {epochs}")
        print(f"Max T_steps: {max_t_steps}")


        # Execute the training phase using the combined dataset
        train_phase_competence(
            model=model,
            device=device,
            tokenizer=tokenizer,
            dataset=dataset,
            max_steps=max_steps,
            update_every=update_every,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            c0=c0,
            val_dataset=val_dataset,
            max_t_steps=max_t_steps
        )

        print("Competence-Based Training completed.")


