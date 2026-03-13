#!/usr/bin/env python3
"""
Example script demonstrating how to use the Industrial LoRA Fine-Tuning System.

This script shows:
1. How to validate configurations
2. How to run training
3. How to export the trained model
4. How to run inference

Usage:
    python examples/example_usage.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.model_manager import ModelManager, create_model_manager
from src.model.lora_configurator import LoRAConfigurator, get_recommended_preset, LORA_PRESETS
from src.data_processing.dataset_loader import DatasetLoader
from src.data_processing.data_formatter import DataFormatter, PROMPT_TEMPLATES
from src.utils.logger import setup_logger
from src.utils.monitor import print_memory_status, ResourceMonitor
from src.utils.validator import ConfigValidator, DatasetValidator

logger = setup_logger("example")


def example_validate_configs():
    """Example: Validate configuration files."""
    print("\n" + "=" * 60)
    print("Example 1: Configuration Validation")
    print("=" * 60)

    validator = ConfigValidator()

    # Validate model config
    print("\nValidating model configuration...")
    model_result = validator.validate_model_config("config/model_config.yaml")
    print(f"Valid: {model_result.is_valid}")
    if model_result.errors:
        print(f"Errors: {model_result.errors}")
    if model_result.warnings:
        print(f"Warnings: {model_result.warnings}")

    # Validate training config
    print("\nValidating training configuration...")
    training_result = validator.validate_training_config("config/training_config.yaml")
    print(f"Valid: {training_result.is_valid}")
    if training_result.errors:
        print(f"Errors: {training_result.errors}")
    if training_result.warnings:
        print(f"Warnings: {training_result.warnings}")


def example_validate_dataset():
    """Example: Validate training dataset."""
    print("\n" + "=" * 60)
    print("Example 2: Dataset Validation")
    print("=" * 60)

    validator = DatasetValidator()

    print("\nValidating training dataset...")
    result = validator.validate_json_dataset("data/training_data.json")
    print(f"Valid: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")


def example_lora_presets():
    """Example: Using LoRA presets."""
    print("\n" + "=" * 60)
    print("Example 3: LoRA Presets")
    print("=" * 60)

    # List available presets
    print("\nAvailable LoRA presets:")
    for preset_name in LoRAConfigurator.list_presets():
        desc = LoRAConfigurator.get_preset_description(preset_name)
        print(f"  - {preset_name}: {desc}")

    # Get recommended preset based on dataset size
    print("\nRecommended preset for different dataset sizes:")
    for size in [500, 5000, 50000]:
        preset = get_recommended_preset(dataset_size=size, task_type="general")
        print(f"  Dataset size {size}: {preset}")

    # Create a LoRA config
    configurator = LoRAConfigurator(model_type="qwen")
    config = configurator.create_config(preset="standard")
    print(f"\nLoRA Configuration Summary:")
    print(configurator.get_config_summary(config))


def example_prompt_templates():
    """Example: Using prompt templates."""
    print("\n" + "=" * 60)
    print("Example 4: Prompt Templates")
    print("=" * 60)

    print("\nAvailable prompt templates:")
    for name in PROMPT_TEMPLATES.keys():
        template = PROMPT_TEMPLATES[name]
        print(f"  - {name}")

    # Format with different templates
    instruction = "Write a Python function to calculate the Fibonacci sequence"
    response = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"

    print("\nFormatted examples:")
    for name in ["alpaca", "chatml", "vicuna"]:
        formatter = DataFormatter(template=name)
        formatted = formatter.format_sample(instruction, response)
        print(f"\n{name} template:")
        print("-" * 40)
        print(formatted[:200] + "..." if len(formatted) > 200 else formatted)


def example_data_processing():
    """Example: Data processing pipeline."""
    print("\n" + "=" * 60)
    print("Example 5: Data Processing")
    print("=" * 60)

    from src.data_processing.data_augmentation import DataAugmenter, AugmentationConfig

    # Sample data
    samples = [
        {"instruction": "Write a hello world program", "response": "print('Hello, World!')"},
        {"instruction": "Explain Python", "response": "Python is a programming language..."}
    ]

    print(f"\nOriginal samples: {len(samples)}")

    # Configure augmentation
    config = AugmentationConfig(
        enabled=True,
        paraphrase_prob=0.3
    )

    augmenter = DataAugmenter(config)
    augmented = augmenter.augment_dataset(samples, augmentation_factor=2.0)

    print(f"Augmented samples: {len(augmented)}")


def example_resource_monitoring():
    """Example: Resource monitoring."""
    print("\n" + "=" * 60)
    print("Example 6: Resource Monitoring")
    print("=" * 60)

    print("\nCurrent system status:")
    print_memory_status()

    # Create monitor
    monitor = ResourceMonitor(log_interval=1.0)

    # Get detailed stats
    stats = monitor.get_all_stats()
    print(f"\nDetailed stats:")
    print(f"  CPU: {stats.cpu_percent:.1f}%")
    print(f"  Memory: {stats.memory_percent:.1f}% ({stats.memory_used_gb:.1f}GB used)")
    print(f"  Disk: {stats.disk_percent:.1f}%")

    if stats.gpu_memory_allocated_gb is not None:
        print(f"  GPU Memory: {stats.gpu_memory_allocated_gb:.1f}GB allocated")


def example_model_info():
    """Example: Model information utilities."""
    print("\n" + "=" * 60)
    print("Example 7: Model Information")
    print("=" * 60)

    # Create model manager
    manager = create_model_manager("config/model_config.yaml")

    # List available models
    from src.model.model_manager import list_available_models
    models = list_available_models("config/model_config.yaml")
    print("\nAvailable models in config:")
    for key, name in models.items():
        print(f"  {key}: {name}")

    # Estimate memory requirements
    print("\nMemory estimates for different model sizes:")
    for params, name in [(7_000_000_000, "7B"), (13_000_000_000, "13B"), (30_000_000_000, "30B")]:
        for quant in ["4bit", "8bit", "fp16"]:
            estimate = manager.estimate_memory_requirements(params, quant, training=True)
            print(f"  {name} ({quant}): {estimate['total_memory_gb']:.1f}GB estimated")


def example_checkpoint_management():
    """Example: Checkpoint management."""
    print("\n" + "=" * 60)
    print("Example 7: Checkpoint Management")
    print("=" * 60)

    from src.model.checkpoint_manager import CheckpointManager

    # Create checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir="./checkpoints",
        max_checkpoints=3,
        save_best_only=False
    )

    print("\nCheckpoint manager created with:")
    print(f"  Directory: {ckpt_manager.checkpoint_dir}")
    print(f"  Max checkpoints: {ckpt_manager.max_checkpoints}")
    print(f"  Metric tracked: {ckpt_manager.metric_name}")

    # List existing checkpoints
    checkpoints = ckpt_manager.get_checkpoint_list()
    print(f"\nExisting checkpoints: {len(checkpoints)}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Industrial LoRA Fine-Tuning System - Usage Examples")
    print("=" * 60)

    try:
        example_validate_configs()
    except FileNotFoundError as e:
        print(f"Skipping config validation: {e}")

    try:
        example_validate_dataset()
    except FileNotFoundError as e:
        print(f"Skipping dataset validation: {e}")

    example_lora_presets()
    example_prompt_templates()
    example_data_processing()
    example_resource_monitoring()

    try:
        example_model_info()
    except FileNotFoundError:
        print("\nSkipping model info example (config not found)")

    example_checkpoint_management()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()