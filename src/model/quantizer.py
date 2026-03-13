"""
Model quantization utilities for memory-efficient training and inference.
Supports various quantization strategies including 4-bit, 8-bit, and mixed precision.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    quant_type: str = "4bit"  # 4bit, 8bit, fp16, bf16, fp32
    compute_dtype: str = "float16"
    use_double_quant: bool = True
    quant_storage_dtype: str = "uint8"

    def to_bnb_config(self):
        """Convert to BitsAndBytes configuration."""
        from transformers import BitsAndBytesConfig

        if self.quant_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self.compute_dtype),
                bnb_4bit_use_double_quant=self.use_double_quant
            )
        elif self.quant_type == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        else:
            return None


class ModelQuantizer:
    """
    Handles model quantization for efficient training and inference.

    Supports:
    - 4-bit quantization (QLoRA)
    - 8-bit quantization
    - Mixed precision (fp16/bf16)
    - CPU-optimized quantization
    """

    # Memory estimates for different quantization levels
    MEMORY_ESTIMATES = {
        "fp32": 4.0,  # 4 bytes per parameter
        "fp16": 2.0,  # 2 bytes per parameter
        "bf16": 2.0,
        "8bit": 1.0,  # 1 byte per parameter
        "4bit": 0.5   # 0.5 bytes per parameter
    }

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize quantizer.

        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()
        self._check_quantization_support()

    def _check_quantization_support(self):
        """Check if quantization libraries are available."""
        try:
            import bitsandbytes
            self.bnb_available = True
        except ImportError:
            self.bnb_available = False
            logger.warning("bitsandbytes not installed. 4-bit/8-bit quantization unavailable.")

    def estimate_memory_requirement(
        self,
        model_params: int,
        quant_type: str = None
    ) -> float:
        """
        Estimate GPU memory requirement for a model.

        Args:
            model_params: Number of parameters
            quant_type: Quantization type (uses config default if None)

        Returns:
            Estimated memory in GB
        """
        quant = quant_type or self.config.quant_type
        bytes_per_param = self.MEMORY_ESTIMATES.get(quant, 4.0)

        # Base memory for parameters
        param_memory = model_params * bytes_per_param / (1024**3)

        # Add overhead for activations, gradients, optimizer states
        # Rough estimate: 50% overhead for training, 20% for inference
        overhead = 0.5 if self._is_training else 0.2

        return param_memory * (1 + overhead)

    @staticmethod
    def get_available_memory() -> float:
        """Get available GPU memory in GB."""
        if torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info()[0]
            return free_memory / (1024**3)
        return 0.0

    def recommend_quantization(
        self,
        model_params: int,
        available_memory_gb: Optional[float] = None
    ) -> str:
        """
        Recommend quantization strategy based on available resources.

        Args:
            model_params: Number of model parameters
            available_memory_gb: Available GPU memory (auto-detected if None)

        Returns:
            Recommended quantization type
        """
        if available_memory_gb is None:
            available_memory_gb = self.get_available_memory()

        # Estimate memory for different quantization levels
        for quant_type in ["4bit", "8bit", "bf16", "fp16", "fp32"]:
            estimated = self.estimate_memory_requirement(model_params, quant_type)
            if estimated < available_memory_gb * 0.8:  # Leave 20% buffer
                logger.info(f"Recommended quantization: {quant_type} (est. {estimated:.1f}GB)")
                return quant_type

        logger.warning("Insufficient memory even with maximum quantization")
        return "4bit"

    def prepare_model_for_training(
        self,
        model,
        enable_gradient_checkpointing: bool = True
    ):
        """
        Prepare quantized model for training.

        Args:
            model: The model to prepare
            enable_gradient_checkpointing: Whether to enable gradient checkpointing

        Returns:
            Prepared model
        """
        from peft import prepare_model_for_kbit_training

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Enable gradient checkpointing
        if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Disable caching for training
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        return model

    def get_quantization_config(
        self,
        quant_type: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get BitsAndBytes configuration for quantization.

        Args:
            quant_type: Quantization type override

        Returns:
            BitsAndBytesConfig or None
        """
        if not self.bnb_available:
            logger.warning("bitsandbytes not available, returning None")
            return None

        quant = quant_type or self.config.quant_type

        if quant in ["4bit", "8bit"]:
            temp_config = QuantizationConfig(quant_type=quant)
            return temp_config.to_bnb_config()

        return None

    def load_model_with_quantization(
        self,
        model_name: str,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        quant_type: Optional[str] = None
    ):
        """
        Load a model with appropriate quantization.

        Args:
            model_name: Model name or path
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
            quant_type: Quantization type override

        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant = quant_type or self.config.quant_type

        logger.info(f"Loading model {model_name} with {quant} quantization")

        # Get quantization config
        quant_config = self.get_quantization_config(quant)

        # Determine torch dtype
        if quant in ["fp16"]:
            torch_dtype = torch.float16
        elif quant in ["bf16"]:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Load model
        if quant_config:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=trust_remote_code
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code
            )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer


class CPUOptimizedLoader:
    """
    Load and run models on CPU with optimizations.
    Useful for inference on devices without GPU.
    """

    def __init__(self, model_path: str):
        """
        Initialize CPU loader.

        Args:
            model_path: Path to the model
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load(
        self,
        use_bf16: bool = False,
        low_cpu_mem_usage: bool = True
    ):
        """
        Load model for CPU inference.

        Args:
            use_bf16: Use bfloat16 if supported
            low_cpu_mem_usage: Reduce CPU memory during loading
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model on CPU: {self.model_path}")

        # Determine dtype
        if use_bf16 and hasattr(torch, 'bfloat16'):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully on CPU")
        return self.model, self.tokenizer


def get_memory_info() -> Dict[str, float]:
    """
    Get current memory information.

    Returns:
        Dictionary with memory statistics
    """
    info = {
        "cpu_memory_gb": 0.0,
        "gpu_memory_gb": 0.0,
        "gpu_memory_free_gb": 0.0
    }

    # CPU memory
    import psutil
    mem = psutil.virtual_memory()
    info["cpu_memory_gb"] = mem.total / (1024**3)
    info["cpu_memory_free_gb"] = mem.available / (1024**3)

    # GPU memory
    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_free = torch.cuda.mem_get_info()[0] / (1024**3)
        info["gpu_memory_gb"] = gpu_total
        info["gpu_memory_free_gb"] = gpu_free

    return info


def print_memory_status():
    """Print current memory status."""
    info = get_memory_info()

    print("=" * 50)
    print("Memory Status")
    print("=" * 50)
    print(f"CPU Memory: {info['cpu_memory_free_gb']:.1f}GB free / {info['cpu_memory_gb']:.1f}GB total")

    if info['gpu_memory_gb'] > 0:
        print(f"GPU Memory: {info['gpu_memory_free_gb']:.1f}GB free / {info['gpu_memory_gb']:.1f}GB total")
    else:
        print("GPU: Not available")

    print("=" * 50)