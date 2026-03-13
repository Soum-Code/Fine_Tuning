"""
LoRA configuration utilities for different model architectures.
Provides presets and optimization for various use cases.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from peft import LoraConfig, TaskType

logger = logging.getLogger(__name__)


@dataclass
class LoRAPreset:
    """Preset configuration for LoRA."""
    name: str
    description: str
    config: Dict[str, Any]


# Common LoRA presets for different scenarios
LORA_PRESETS: Dict[str, LoRAPreset] = {
    "conservative": LoRAPreset(
        name="conservative",
        description="Minimal parameter tuning, good for small datasets",
        config={
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj"]
        }
    ),
    "standard": LoRAPreset(
        name="standard",
        description="Balanced configuration for general use",
        config={
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    ),
    "aggressive": LoRAPreset(
        name="aggressive",
        description="More parameters for larger datasets",
        config={
            "r": 16,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
    ),
    "code_finetuning": LoRAPreset(
        name="code_finetuning",
        description="Optimized for code generation tasks",
        config={
            "r": 16,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "mlp.fc1", "mlp.fc2"]
        }
    ),
    "conversation": LoRAPreset(
        name="conversation",
        description="For chat/instruction tuning",
        config={
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    )
}


class LoRAConfigurator:
    """
    Configure LoRA adapters for different models and use cases.

    Provides utilities for:
    - Selecting optimal LoRA configurations
    - Model-specific target module detection
    - Configuration optimization
    """

    # Model-specific target modules
    MODEL_TARGET_MODULES: Dict[str, List[str]] = {
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "opt": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "stablelm": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }

    def __init__(self, model_type: Optional[str] = None):
        """
        Initialize LoRA configurator.

        Args:
            model_type: Type of model (llama, qwen, mistral, etc.)
        """
        self.model_type = model_type

    @staticmethod
    def get_preset(preset_name: str) -> LoRAPreset:
        """
        Get a LoRA configuration preset.

        Args:
            preset_name: Name of the preset

        Returns:
            LoRAPreset configuration
        """
        if preset_name not in LORA_PRESETS:
            available = list(LORA_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        return LORA_PRESETS[preset_name]

    @staticmethod
    def list_presets() -> List[str]:
        """Get list of available presets."""
        return list(LORA_PRESETS.keys())

    @staticmethod
    def get_preset_description(preset_name: str) -> str:
        """Get description of a preset."""
        preset = LORA_PRESETS.get(preset_name)
        if preset:
            return preset.description
        return f"Unknown preset: {preset_name}"

    def create_config(
        self,
        preset: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        task_type: TaskType = TaskType.CAUSAL_LM
    ) -> LoraConfig:
        """
        Create a LoRA configuration.

        Args:
            preset: Name of preset to use
            custom_config: Custom configuration overrides
            task_type: Task type for the adapter

        Returns:
            LoraConfig object
        """
        if preset:
            base_config = self.get_preset(preset).config.copy()
        else:
            base_config = {
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
                "target_modules": ["q_proj", "v_proj"]
            }

        # Override with custom config
        if custom_config:
            base_config.update(custom_config)

        # Detect target modules if model type specified
        if self.model_type and "target_modules" not in custom_config:
            base_config["target_modules"] = self._detect_target_modules()

        return LoraConfig(
            r=base_config.get("r", 8),
            lora_alpha=base_config.get("lora_alpha", 32),
            target_modules=base_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=base_config.get("lora_dropout", 0.1),
            bias=base_config.get("bias", "none"),
            task_type=task_type
        )

    def _detect_target_modules(self) -> List[str]:
        """Detect appropriate target modules for the model type."""
        if not self.model_type:
            return ["q_proj", "v_proj"]

        # Normalize model type
        model_type_lower = self.model_type.lower()

        for key, modules in self.MODEL_TARGET_MODULES.items():
            if key in model_type_lower:
                logger.info(f"Detected model type '{key}', using target modules: {modules}")
                return modules

        # Default fallback
        logger.warning(f"Unknown model type '{self.model_type}', using default target modules")
        return ["q_proj", "v_proj"]

    def optimize_for_memory(
        self,
        config: LoraConfig,
        available_memory_gb: float
    ) -> LoraConfig:
        """
        Optimize LoRA configuration for available memory.

        Args:
            config: Original configuration
            available_memory_gb: Available GPU memory in GB

        Returns:
            Optimized LoraConfig
        """
        # Estimate memory requirements based on rank
        # Higher rank = more parameters = more memory
        if available_memory_gb < 8:
            # Low memory: reduce rank and target modules
            optimized_r = min(config.r, 4)
            optimized_modules = config.target_modules[:2]  # Only q_proj, v_proj
        elif available_memory_gb < 16:
            # Medium memory: moderate settings
            optimized_r = min(config.r, 8)
            optimized_modules = config.target_modules[:4]
        else:
            # High memory: use original config
            optimized_r = config.r
            optimized_modules = config.target_modules

        return LoraConfig(
            r=optimized_r,
            lora_alpha=config.lora_alpha,
            target_modules=optimized_modules,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=config.task_type
        )

    def estimate_trainable_params(
        self,
        model,
        config: LoraConfig
    ) -> Dict[str, int]:
        """
        Estimate number of trainable parameters.

        Args:
            model: The model
            config: LoRA configuration

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())

        # Estimate LoRA parameters
        # Each target module gets 2 matrices: A and B
        # A: hidden_size x r, B: r x hidden_size
        # Approximate based on common architectures

        lora_params_per_module = config.r * 2  # Simplified estimate
        num_modules = len(config.target_modules)

        # More accurate estimation would require model introspection
        estimated_lora_params = lora_params_per_module * num_modules * 1000000  # Rough estimate

        trainable_percent = (estimated_lora_params / total_params) * 100

        return {
            "total_params": total_params,
            "estimated_lora_params": estimated_lora_params,
            "trainable_percent": trainable_percent
        }

    def get_config_summary(self, config: LoraConfig) -> str:
        """Get a human-readable summary of the configuration."""
        summary = [
            "LoRA Configuration Summary:",
            f"  Rank (r): {config.r}",
            f"  Alpha: {config.lora_alpha}",
            f"  Dropout: {config.lora_dropout}",
            f"  Bias: {config.bias}",
            f"  Target Modules: {config.target_modules}",
            f"  Task Type: {config.task_type}"
        ]
        return "\n".join(summary)


def get_recommended_preset(dataset_size: int, task_type: str = "general") -> str:
    """
    Get recommended LoRA preset based on dataset size and task type.

    Args:
        dataset_size: Number of training examples
        task_type: Type of task (general, code, conversation)

    Returns:
        Recommended preset name
    """
    # Based on dataset size
    if dataset_size < 1000:
        size_recommendation = "conservative"
    elif dataset_size < 10000:
        size_recommendation = "standard"
    else:
        size_recommendation = "aggressive"

    # Adjust for task type
    if task_type == "code":
        return "code_finetuning"
    elif task_type == "conversation":
        return "conversation"
    else:
        return size_recommendation