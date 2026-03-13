"""
Industrial LoRA Fine-Tuning System
===================================

A production-ready system for fine-tuning large language models
using PEFT (LoRA/QLoRA) techniques.

Main Components:
- ModelManager: Load and configure models with LoRA
- DatasetLoader: Load and process training data
- BatchProcessor: Handle batch training with gradient accumulation
- IndustrialTrainer: Complete training pipeline
- ModelDeployer: Deploy models for inference

Example Usage:
    from src.training.trainer import IndustrialTrainer

    trainer = IndustrialTrainer(
        model_config_path="config/model_config.yaml",
        training_config_path="config/training_config.yaml"
    )

    trainer.run_training(
        model_key="qwen2_5_0_5b",
        dataset_path="./data/training_data.json"
    )
"""

__version__ = "1.0.0"
__author__ = "Industrial LoRA Team"

# Import main components for easy access
from src.model.model_manager import ModelManager
from src.data_processing.dataset_loader import DatasetLoader
from src.training.batch_processor import BatchProcessor
from src.training.trainer import IndustrialTrainer
from src.inference.model_deployer import ModelDeployer

__all__ = [
    "ModelManager",
    "DatasetLoader",
    "BatchProcessor",
    "IndustrialTrainer",
    "ModelDeployer",
]