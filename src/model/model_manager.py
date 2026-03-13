"""
Model management for loading, configuring, and saving models.
Integrates with quantization, LoRA configuration, and checkpoint management.
"""

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, AutoConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import yaml
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manage model loading, configuration, and LoRA application.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize model manager.

        Args:
            config_path: Path to model configuration file
        """
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

    def load_base_model(
        self,
        model_key: str = None,
        model_name: str = None,
        device_map: str = "auto",
        quantization_config: Dict = None
    ) -> Tuple:
        """
        Load base model with optional quantization.

        Args:
            model_key: Key from config file for model
            model_name: Direct model name (overrides model_key)
            device_map: Device mapping strategy
            quantization_config: Custom quantization config

        Returns:
            Tuple of (model, tokenizer)
        """
        # Get model info
        if model_name:
            model_info = {'model_name': model_name, 'tokenizer_name': model_name}
        elif model_key and model_key in self.config.get('base_models', {}):
            model_info = self.config['base_models'][model_key]
        else:
            raise ValueError(f"Must provide model_name or valid model_key")

        logger.info(f"Loading model: {model_info['model_name']}")

        # Configure quantization
        if quantization_config:
            bnb_config = quantization_config
        else:
            # Check if model is already quantized on HF
            try:
                remote_config = AutoConfig.from_pretrained(model_info['model_name'], trust_remote_code=True)
                has_remote_quant = hasattr(remote_config, "quantization_config")
            except Exception:
                has_remote_quant = False

            if has_remote_quant:
                logger.info("Model detected as pre-quantized. Skipping BitsAndBytesConfig.")
                bnb_config = None
            elif self.config.get('quantization', {}).get('use_4bit'):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config['quantization'].get('bnb_4bit_quant_type', 'nf4'),
                    bnb_4bit_compute_dtype=getattr(torch, self.config['quantization'].get('bnb_4bit_compute_dtype', 'float16')),
                    bnb_4bit_use_double_quant=self.config['quantization'].get('bnb_4bit_use_double_quant', True)
                )
            else:
                bnb_config = None

        # Load model
        try:
            load_params = {
                "device_map": device_map,
                "trust_remote_code": True
            }
            if bnb_config:
                load_params["quantization_config"] = bnb_config
            
            model = AutoModelForCausalLM.from_pretrained(
                model_info['model_name'],
                **load_params
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_info.get('tokenizer_name', model_info['model_name']),
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare for k-bit training if quantization is used
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        elif hasattr(model.config, "quantization_config"):
            # If model is pre-quantized, still prepare for k-bit training
            model = prepare_model_for_kbit_training(model)
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded: {total_params:,} total params, {trainable_params:,} trainable")

        return model, tokenizer

    def apply_lora(
        self,
        model,
        lora_config_dict: Dict[str, Any] = None,
        use_default: bool = True
    ):
        """
        Apply LoRA configuration to model.

        Args:
            model: The model to apply LoRA to
            lora_config_dict: Custom LoRA configuration
            use_default: Whether to use default config if no custom config provided

        Returns:
            Model with LoRA applied
        """
        # Get config
        if lora_config_dict is None and use_default:
            lora_config_dict = self.config.get('lora', {})

        if not lora_config_dict:
            logger.warning("No LoRA config provided, using defaults")
            lora_config_dict = {
                'r': 8,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj'],
                'lora_dropout': 0.1,
                'bias': 'none'
            }

        lora_config = LoraConfig(
            r=lora_config_dict.get('r', 8),
            lora_alpha=lora_config_dict.get('lora_alpha', 32),
            target_modules=lora_config_dict.get('target_modules', ['q_proj', 'v_proj']),
            lora_dropout=lora_config_dict.get('lora_dropout', 0.1),
            bias=lora_config_dict.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable_params / all_params
        logger.info(f"LoRA applied: {trainable_params:,} trainable params ({trainable_pct:.4f}%)")

        return model

    def merge_and_save(
        self,
        model,
        tokenizer,
        output_path: str,
        safe_serialization: bool = True
    ):
        """
        Merge LoRA weights with base model and save.

        Args:
            model: PEFT model with LoRA
            tokenizer: Tokenizer
            output_path: Path to save merged model
            safe_serialization: Use safe serialization format
        """
        logger.info(f"Merging and saving model to {output_path}")

        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Merge weights
        try:
            merged_model = model.merge_and_unload()
        except Exception as e:
            logger.warning(f"Could not merge model: {str(e)}")
            merged_model = model

        # Save
        merged_model.save_pretrained(
            output_path,
            safe_serialization=safe_serialization
        )
        tokenizer.save_pretrained(output_path)

        logger.info(f"Model saved to {output_path}")

    def save_lora_weights(
        self,
        model,
        output_path: str,
        adapter_name: str = "default"
    ):
        """
        Save only LoRA weights (not full model).

        Args:
            model: PEFT model
            output_path: Path to save adapter
            adapter_name: Name for the adapter
        """
        logger.info(f"Saving LoRA weights to {output_path}")

        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Save adapter
        model.save_pretrained(output_path)

        logger.info(f"LoRA weights saved to {output_path}")

    def load_lora_weights(
        self,
        model,
        adapter_path: str,
        adapter_name: str = "default"
    ):
        """
        Load LoRA weights into model.

        Args:
            model: Base model
            adapter_path: Path to saved adapter
            adapter_name: Name for the adapter

        Returns:
            Model with loaded adapter
        """
        from peft import PeftModel

        logger.info(f"Loading LoRA weights from {adapter_path}")

        model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)

        return model

    def get_model_info(self, model) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: The model

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get model size in bytes
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Estimate memory
        if torch.cuda.is_available() and hasattr(model, 'device') and model.device.type == 'cuda':
            model_memory = torch.cuda.memory_allocated() / (1024**3)
        else:
            model_memory = param_size / (1024**3)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
            "model_size_gb": model_memory,
            "dtype": str(next(model.parameters()).dtype) if total_params > 0 else "unknown"
        }

    def estimate_memory_requirements(
        self,
        model_params: int,
        quantization: str = "4bit",
        training: bool = True
    ) -> Dict[str, float]:
        """
        Estimate memory requirements for a model.

        Args:
            model_params: Number of parameters
            quantization: Quantization type (4bit, 8bit, fp16, fp32)
            training: Whether training (requires more memory)

        Returns:
            Dictionary with memory estimates
        """
        # Bytes per parameter
        quant_map = {
            "4bit": 0.5,
            "8bit": 1.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "fp32": 4.0
        }

        bytes_per_param = quant_map.get(quantization, 4.0)

        # Base memory for parameters
        param_memory = model_params * bytes_per_param / (1024**3)  # GB

        # Additional memory for training
        # Gradients: same size as params
        # Optimizer states: 2x params for Adam
        # Activations: varies, estimate ~50% of params
        if training:
            gradient_memory = param_memory
            optimizer_memory = 2 * param_memory if quantization in ["fp16", "bf16", "fp32"] else param_memory
            activation_memory = 0.5 * param_memory
            total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
        else:
            total_memory = param_memory

        return {
            "parameter_memory_gb": param_memory,
            "total_memory_gb": total_memory,
            "recommended_gpu_memory_gb": total_memory * 1.2  # 20% buffer
        }


def create_model_manager(config_path: str = None) -> ModelManager:
    """Factory function to create model manager."""
    return ModelManager(config_path)


def list_available_models(config_path: str) -> Dict[str, str]:
    """List available models from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return {
        key: info.get('model_name', key)
        for key, info in config.get('base_models', {}).items()
    }