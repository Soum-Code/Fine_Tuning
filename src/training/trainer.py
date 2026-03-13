"""
Industrial-grade training orchestrator for LoRA fine-tuning.
Integrates model management, data processing, and checkpoint handling.
"""

import yaml
import logging
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add src to path if necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.model_manager import ModelManager
from src.model.checkpoint_manager import CheckpointManager
from src.data_processing.dataset_loader import DatasetLoader
from src.data_processing.data_formatter import DataFormatter, PROMPT_TEMPLATES
from src.training.batch_processor import BatchProcessor
from src.utils.logger import setup_logger, TrainingLogger
from src.utils.monitor import TrainingMonitor, ResourceMonitor
from src.utils.validator import validate_all_configs, validate_dataset

logger = setup_logger("trainer")


class IndustrialTrainer:
    """
    Complete training pipeline for LoRA fine-tuning.

    Features:
    - Automatic model loading with quantization
    - LoRA configuration and application
    - Checkpoint management
    - Resource monitoring
    - Data validation and formatting
    """

    def __init__(
        self,
        model_config_path: str,
        training_config_path: str,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs"
    ):
        """
        Initialize trainer.

        Args:
            model_config_path: Path to model configuration
            training_config_path: Path to training configuration
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.model_config_path = model_config_path
        self.training_config_path = training_config_path

        # Load configurations
        with open(model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)

        with open(training_config_path, 'r') as f:
            self.training_config = yaml.safe_load(f)

        # Initialize components
        self.model_manager = ModelManager(model_config_path)
        self.dataset_loader = DatasetLoader()
        self.batch_processor = BatchProcessor(self.training_config)

        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=self.training_config.get('checkpointing', {}).get('save_total_limit', 3),
            metric_name="loss",
            mode="min"
        )

        # Setup training logger
        self.training_logger = TrainingLogger(log_dir)

        # Setup resource monitor
        self.resource_monitor = ResourceMonitor()

        # Track training state
        self.model = None
        self.tokenizer = None
        self.current_step = 0

    def validate_configs(self) -> bool:
        """
        Validate configuration files.

        Returns:
            True if valid, raises exception otherwise
        """
        result = validate_all_configs(
            self.model_config_path,
            self.training_config_path
        )

        if not result.is_valid:
            for error in result.errors:
                logger.error(f"Config error: {error}")
            raise ValueError("Invalid configuration")

        for warning in result.warnings:
            logger.warning(f"Config warning: {warning}")

        return True

    def validate_dataset(self, dataset_path: str) -> bool:
        """
        Validate training dataset.

        Args:
            dataset_path: Path to dataset file

        Returns:
            True if valid
        """
        result = validate_dataset(dataset_path)

        if not result.is_valid:
            for error in result.errors:
                logger.error(f"Dataset error: {error}")
            raise ValueError("Invalid dataset")

        for warning in result.warnings:
            logger.warning(f"Dataset warning: {warning}")

        return True

    def prepare_dataset(
        self,
        dataset_path: str,
        template: str = "alpaca",
        max_length: int = 512
    ):
        """
        Load and prepare dataset for training.

        Args:
            dataset_path: Path to dataset file
            template: Prompt template to use
            max_length: Maximum sequence length

        Returns:
            Tokenized dataset
        """
        logger.info(f"Loading dataset from {dataset_path}")

        # Load raw data
        if dataset_path.endswith('.json'):
            dataset = self.dataset_loader.load_from_file(dataset_path, format="json")
        elif dataset_path.endswith('.csv'):
            dataset = self.dataset_loader.load_from_file(dataset_path, format="csv")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")

        logger.info(f"Loaded {len(dataset)} samples")

        # Format as instruction dataset
        formatter = DataFormatter(template=template, max_length=max_length)
        dataset = self.dataset_loader.format_as_instruction_dataset(dataset)

        # Tokenize
        tokenized_dataset = self.dataset_loader.tokenize_dataset(
            dataset,
            self.tokenizer,
            max_length=max_length
        )

        logger.info(f"Dataset prepared: {len(tokenized_dataset)} samples")
        return tokenized_dataset

    def run_training(
        self,
        model_key: str,
        dataset_path: str,
        template: str = "alpaca",
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Run complete training pipeline.

        Args:
            model_key: Key for model in config
            dataset_path: Path to training data
            template: Prompt template
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results
        """
        try:
            # Log training start
            self.training_logger.log_training_start({
                "model_key": model_key,
                "dataset_path": dataset_path,
                "config": self.training_config
            })

            # Start resource monitoring
            self.resource_monitor.start_monitoring()

            # 1. Validate configurations
            logger.info("Validating configurations...")
            self.validate_configs()

            # 2. Validate dataset
            logger.info("Validating dataset...")
            self.validate_dataset(dataset_path)

            # 3. Load base model
            logger.info(f"Loading model: {model_key}")
            self.model, self.tokenizer = self.model_manager.load_base_model(model_key)

            # Log model info
            model_info = self.model_manager.get_model_info(self.model)
            logger.info(f"Model info: {model_info}")

            # 4. Apply LoRA
            logger.info("Applying LoRA configuration...")
            self.model = self.model_manager.apply_lora(self.model)

            # 5. Prepare dataset
            logger.info("Preparing dataset...")
            train_dataset = self.prepare_dataset(dataset_path, template=template)

            # Split for validation if configured
            eval_dataset = None
            if self.training_config.get('training', {}).get('eval_split', False):
                split_ratio = self.training_config['training'].get('eval_split_ratio', 0.1)
                train_dataset = train_dataset.train_test_split(test_size=split_ratio)
                eval_dataset = train_dataset['test']
                train_dataset = train_dataset['train']

            # 6. Train model
            logger.info("Starting training...")
            trainer = self.batch_processor.train_model(
                self.model,
                train_dataset,
                eval_dataset=eval_dataset
            )

            # 7. Save final checkpoint
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_checkpoint_path = os.path.join(
                self.checkpoint_manager.checkpoint_dir,
                f"final_{timestamp}"
            )

            self.checkpoint_manager.save_checkpoint(
                self.model,
                self.tokenizer,
                metrics={"step": self.current_step},
                step=self.current_step,
                extra_data={"training_config": self.training_config}
            )

            # Stop monitoring
            self.resource_monitor.stop_monitoring()

            # Log training end
            self.training_logger.log_training_end({
                "final_checkpoint": final_checkpoint_path,
                "model_info": model_info
            })

            logger.info("Training completed successfully!")

            return {
                "trainer": trainer,
                "model": self.model,
                "tokenizer": self.tokenizer,
                "checkpoint_path": final_checkpoint_path
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.training_logger.log_error(e, "training")
            self.resource_monitor.stop_monitoring()
            raise

    def run_inference(self, prompt: str, **kwargs) -> str:
        """
        Run inference with the trained model.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not loaded. Run training first or load a checkpoint.")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        default_params = {
            "max_new_tokens": kwargs.get("max_new_tokens", 200),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("do_sample", True)
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **default_params)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def export_model(self, output_path: str, merge_weights: bool = True):
        """
        Export the trained model.

        Args:
            output_path: Path to save model
            merge_weights: Whether to merge LoRA weights with base model
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if merge_weights:
            self.model_manager.merge_and_save(
                self.model,
                self.tokenizer,
                output_path
            )
        else:
            self.model_manager.save_lora_weights(
                self.model,
                output_path
            )

        logger.info(f"Model exported to {output_path}")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Industrial LoRA Trainer")
    parser.add_argument("--model", type=str, default="qwen2_5_0_5b",
                        help="Model key from config")
    parser.add_argument("--dataset", type=str, default="./data/training_data.json",
                        help="Path to training data")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml",
                        help="Path to model config")
    parser.add_argument("--training-config", type=str, default="config/training_config.yaml",
                        help="Path to training config")
    parser.add_argument("--template", type=str, default="alpaca",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help="Prompt template to use")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Log directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    args = parser.parse_args()

    # Create trainer
    trainer = IndustrialTrainer(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Run training
    trainer.run_training(
        model_key=args.model,
        dataset_path=args.dataset,
        template=args.template,
        resume_from_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()