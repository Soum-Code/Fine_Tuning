import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from typing import Dict, Any
import os
from datetime import datetime

class BatchProcessor:
    def __init__(self, training_config: Dict[str, Any]):
        self.config = training_config
        self.output_dir = f"./checkpoints/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

    def create_training_args(self) -> TrainingArguments:
        """Create training arguments for efficient batch processing"""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=int(self.config['training']['batch_size']),
            gradient_accumulation_steps=int(self.config['training']['gradient_accumulation_steps']),
            num_train_epochs=float(self.config['training']['num_epochs']),
            learning_rate=float(self.config['training']['learning_rate']),
            warmup_ratio=float(self.config['training']['warmup_ratio']),
            weight_decay=float(self.config['training']['weight_decay']),
            max_grad_norm=float(self.config['training']['max_grad_norm']),
            optim=self.config['optimizer']['name'],
            lr_scheduler_type=self.config['optimizer']['lr_scheduler_type'],
            logging_steps=int(self.config['logging']['log_steps']),
            save_steps=int(self.config['checkpointing']['save_steps']),
            save_total_limit=int(self.config['checkpointing']['save_total_limit']),
            fp16=bool(self.config['device']['use_mixed_precision']),
            gradient_checkpointing=bool(self.config['device']['gradient_checkpointing']),
            dataloader_pin_memory=False,  # Reduce memory usage
            remove_unused_columns=False,
            report_to=["wandb"] if self.config['logging']['wandb_project'] else ["none"]
        )

    def train_model(self, model, train_dataset, eval_dataset=None):
        """Train model with batch-wise processing"""
        training_args = self.create_training_args()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Start training
        trainer.train(resume_from_checkpoint=self.config['checkpointing']['resume_from_checkpoint'])

        # Save final model
        trainer.save_model()
        trainer.save_state()

        return trainer
