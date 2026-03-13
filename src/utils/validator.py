"""
Data and configuration validation utilities for industrial fine-tuning.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

    def __str__(self) -> str:
        parts = [f"Valid: {self.is_valid}"]
        if self.errors:
            parts.append(f"Errors: {self.errors}")
        if self.warnings:
            parts.append(f"Warnings: {self.warnings}")
        return " | ".join(parts)


class ConfigValidator:
    """Validate configuration files for training."""

    REQUIRED_MODEL_KEYS = ['model_name', 'tokenizer_name', 'model_class', 'tokenizer_class']
    REQUIRED_TRAINING_KEYS = ['batch_size', 'gradient_accumulation_steps', 'num_epochs', 'learning_rate']
    REQUIRED_LORA_KEYS = ['r', 'lora_alpha', 'lora_dropout', 'target_modules']

    def __init__(self):
        self.result = ValidationResult(is_valid=True)

    def validate_model_config(self, config_path: str) -> ValidationResult:
        """Validate model configuration file."""
        self.result = ValidationResult(is_valid=True)

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.result.add_error(f"Failed to load config: {str(e)}")
            return self.result

        # Check base_models section
        if 'base_models' not in config:
            self.result.add_error("Missing 'base_models' section")
        else:
            for model_key, model_config in config['base_models'].items():
                for key in self.REQUIRED_MODEL_KEYS:
                    if key not in model_config:
                        self.result.add_error(f"Model '{model_key}' missing key: {key}")

        # Check quantization section
        if 'quantization' in config:
            if config['quantization'].get('use_4bit', False):
                required_quant_keys = ['bnb_4bit_quant_type', 'bnb_4bit_compute_dtype']
                for key in required_quant_keys:
                    if key not in config['quantization']:
                        self.result.add_warning(f"Quantization missing recommended key: {key}")

        # Check LoRA section
        if 'lora' in config:
            for key in self.REQUIRED_LORA_KEYS:
                if key not in config['lora']:
                    self.result.add_error(f"LoRA config missing key: {key}")
        else:
            self.result.add_error("Missing 'lora' configuration section")

        return self.result

    def validate_training_config(self, config_path: str) -> ValidationResult:
        """Validate training configuration file."""
        self.result = ValidationResult(is_valid=True)

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.result.add_error(f"Failed to load config: {str(e)}")
            return self.result

        # Check training section
        if 'training' not in config:
            self.result.add_error("Missing 'training' section")
        else:
            for key in self.REQUIRED_TRAINING_KEYS:
                if key not in config['training']:
                    self.result.add_error(f"Training config missing key: {key}")

            # Validate values
            training = config['training']
            try:
                if float(training.get('batch_size', 1)) < 1:
                    self.result.add_error("batch_size must be >= 1")
                if float(training.get('learning_rate', 0)) <= 0:
                    self.result.add_error("learning_rate must be > 0")
                if float(training.get('gradient_accumulation_steps', 1)) < 1:
                    self.result.add_error("gradient_accumulation_steps must be >= 1")
            except (ValueError, TypeError):
                self.result.add_error("Numerical parameters must be valid numbers")

        # Check optimizer section
        if 'optimizer' not in config:
            self.result.add_warning("Missing 'optimizer' section, using defaults")

        # Check checkpointing section
        if 'checkpointing' not in config:
            self.result.add_warning("Missing 'checkpointing' section, using defaults")

        return self.result


class DatasetValidator:
    """Validate training datasets."""

    def __init__(self):
        self.result = ValidationResult(is_valid=True)

    def validate_json_dataset(self, file_path: str, required_keys: List[str] = None) -> ValidationResult:
        """Validate JSON format dataset."""
        self.result = ValidationResult(is_valid=True)
        required_keys = required_keys or ['instruction', 'response']

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.result.add_error(f"Invalid JSON format: {str(e)}")
            return self.result
        except FileNotFoundError:
            self.result.add_error(f"File not found: {file_path}")
            return self.result

        if not isinstance(data, list):
            self.result.add_error("Dataset must be a JSON array")
            return self.result

        if len(data) == 0:
            self.result.add_error("Dataset is empty")
            return self.result

        # Check each entry
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                self.result.add_error(f"Entry {i} is not a dictionary")
                continue

            for key in required_keys:
                if key not in entry:
                    self.result.add_error(f"Entry {i} missing required key: {key}")

        # Check for duplicate instructions
        instructions = [entry.get('instruction', '') for entry in data if isinstance(entry, dict)]
        if len(instructions) != len(set(instructions)):
            self.result.add_warning("Dataset contains duplicate instructions")

        # Check for very short entries
        short_entries = sum(1 for entry in data
                          if isinstance(entry, dict)
                          and len(str(entry.get('response', ''))) < 10)
        if short_entries > 0:
            self.result.add_warning(f"{short_entries} entries have very short responses (< 10 chars)")

        return self.result

    def validate_csv_dataset(self, file_path: str, required_columns: List[str] = None) -> ValidationResult:
        """Validate CSV format dataset."""
        import pandas as pd
        self.result = ValidationResult(is_valid=True)
        required_columns = required_columns or ['instruction', 'response']

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self.result.add_error(f"Failed to read CSV: {str(e)}")
            return self.result

        # Check required columns
        for col in required_columns:
            if col not in df.columns:
                self.result.add_error(f"Missing required column: {col}")

        # Check for empty values
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    self.result.add_warning(f"Column '{col}' has {null_count} null values")

        return self.result


class ModelValidator:
    """Validate model compatibility and resources."""

    @staticmethod
    def check_gpu_memory(required_memory_gb: float) -> ValidationResult:
        """Check if sufficient GPU memory is available."""
        import torch
        result = ValidationResult(is_valid=True)

        if not torch.cuda.is_available():
            result.add_warning("CUDA not available, will use CPU (slower)")
            return result

        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < required_memory_gb:
            result.add_warning(
                f"GPU memory ({gpu_memory:.1f}GB) may be insufficient "
                f"(recommended: {required_memory_gb:.1f}GB)"
            )

        return result

    @staticmethod
    def check_disk_space(required_space_gb: float, path: str = "./") -> ValidationResult:
        """Check if sufficient disk space is available."""
        import shutil
        result = ValidationResult(is_valid=True)

        try:
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (1024**3)

            if free_gb < required_space_gb:
                result.add_warning(
                    f"Low disk space: {free_gb:.1f}GB free "
                    f"(recommended: {required_space_gb:.1f}GB)"
                )
        except Exception as e:
            result.add_warning(f"Could not check disk space: {str(e)}")

        return result


def validate_all_configs(model_config: str, training_config: str) -> ValidationResult:
    """Validate all configuration files."""
    config_validator = ConfigValidator()

    model_result = config_validator.validate_model_config(model_config)
    training_result = config_validator.validate_training_config(training_config)

    combined = ValidationResult(is_valid=True)
    combined.errors = model_result.errors + training_result.errors
    combined.warnings = model_result.warnings + training_result.warnings
    combined.is_valid = len(combined.errors) == 0

    return combined


def validate_dataset(file_path: str) -> ValidationResult:
    """Validate dataset file based on extension."""
    dataset_validator = DatasetValidator()

    if file_path.endswith('.json'):
        return dataset_validator.validate_json_dataset(file_path)
    elif file_path.endswith('.csv'):
        return dataset_validator.validate_csv_dataset(file_path)
    else:
        result = ValidationResult(is_valid=False)
        result.add_error(f"Unsupported file format: {file_path}")
        return result