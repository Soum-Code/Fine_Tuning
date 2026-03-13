"""
Main entry point for Industrial LoRA Fine-Tuning System.
Provides CLI commands for training, inference, and model management.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import setup_logger
from src.utils.monitor import print_memory_status, get_system_summary

logger = setup_logger("main")


def train_command(args):
    """Run training pipeline."""
    from src.training.trainer import IndustrialTrainer

    trainer = IndustrialTrainer(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    result = trainer.run_training(
        model_key=args.model,
        dataset_path=args.dataset,
        template=args.template,
        resume_from_checkpoint=args.resume
    )

    # Export if requested
    if args.export:
        trainer.export_model(args.export, merge_weights=args.merge)

    logger.info(f"Training completed. Checkpoint: {result['checkpoint_path']}")


def inference_command(args):
    """Run inference with a trained model."""
    from src.inference.model_deployer import ModelDeployer
    from src.model.model_manager import ModelManager

    # Load model
    deployer = ModelDeployer(args.model_path, device=args.device)
    deployer.load_model(quantize=args.quantize)

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 60)
        print("Interactive Inference Mode")
        print("Type 'quit' to exit, 'clear' to clear history")
        print("=" * 60 + "\n")

        while True:
            try:
                prompt = input("User: ").strip()

                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                elif not prompt:
                    continue

                # Generate response
                response = deployer.generate_text(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )

                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        # Single prompt mode
        if args.prompt:
            response = deployer.generate_text(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(response)
        else:
            print("No prompt provided. Use --prompt or --interactive")


def benchmark_command(args):
    """Run benchmarking."""
    from scripts.benchmark import run_full_benchmark
    import json

    config = {
        "model_name": args.model,
        "quantization": args.quantization,
        "benchmark_inference": args.inference,
        "max_new_tokens": args.max_tokens
    }

    results = run_full_benchmark(config)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")


def validate_command(args):
    """Validate configurations and data."""
    from src.utils.validator import validate_all_configs, validate_dataset

    # Validate configs
    print("\nValidating configurations...")
    config_result = validate_all_configs(args.model_config, args.training_config)

    if config_result.is_valid:
        print("✓ Configuration files are valid")
    else:
        print("✗ Configuration errors found:")
        for error in config_result.errors:
            print(f"  - {error}")

    for warning in config_result.warnings:
        print(f"  ! {warning}")

    # Validate dataset
    if args.dataset:
        print("\nValidating dataset...")
        dataset_result = validate_dataset(args.dataset)

        if dataset_result.is_valid:
            print("✓ Dataset is valid")
        else:
            print("✗ Dataset errors found:")
            for error in dataset_result.errors:
                print(f"  - {error}")

        for warning in dataset_result.warnings:
            print(f"  ! {warning}")

    return config_result.is_valid and (not args.dataset or dataset_result.is_valid)


def info_command(args):
    """Display system and configuration info."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    # Memory info
    print_memory_status()

    # Model info
    if args.model_config:
        from src.model.model_manager import list_available_models
        print("\n" + "-" * 60)
        print("AVAILABLE MODELS")
        print("-" * 60)
        models = list_available_models(args.model_config)
        for key, name in models.items():
            print(f"  {key}: {name}")

    # Templates
    print("\n" + "-" * 60)
    print("PROMPT TEMPLATES")
    print("-" * 60)
    from src.data_processing.data_formatter import PROMPT_TEMPLATES
    for name in PROMPT_TEMPLATES.keys():
        print(f"  {name}")

    print("\n" + "=" * 60)


def export_command(args):
    """Export trained model."""
    from src.model.model_manager import ModelManager
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA weights
    if args.lora_path:
        logger.info(f"Loading LoRA weights from: {args.lora_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        model = base_model

    # Merge and save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.merge:
        logger.info("Merging LoRA weights with base model...")
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()

    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    logger.info(f"Model exported to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Industrial LoRA Fine-Tuning System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, default="qwen2_5_0_5b",
                               help="Model key from config")
    train_parser.add_argument("--dataset", type=str, default="./data/training_data.json",
                               help="Path to training data")
    train_parser.add_argument("--model-config", type=str, default="config/model_config.yaml",
                               help="Path to model config")
    train_parser.add_argument("--training-config", type=str, default="config/training_config.yaml",
                               help="Path to training config")
    train_parser.add_argument("--template", type=str, default="alpaca",
                               help="Prompt template")
    train_parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                               help="Checkpoint directory")
    train_parser.add_argument("--log-dir", type=str, default="./logs",
                               help="Log directory")
    train_parser.add_argument("--resume", type=str, default=None,
                               help="Resume from checkpoint path")
    train_parser.add_argument("--export", type=str, default=None,
                               help="Export model to path after training")
    train_parser.add_argument("--merge", action="store_true",
                               help="Merge LoRA weights when exporting")

    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference")
    inference_parser.add_argument("--model-path", type=str, required=True,
                                   help="Path to model")
    inference_parser.add_argument("--device", type=str, default="cpu",
                                   choices=["cpu", "cuda"],
                                   help="Device to use")
    inference_parser.add_argument("--quantize", action="store_true",
                                   help="Use quantization")
    inference_parser.add_argument("--prompt", type=str,
                                   help="Input prompt")
    inference_parser.add_argument("--interactive", action="store_true",
                                   help="Interactive mode")
    inference_parser.add_argument("--max-tokens", type=int, default=200,
                                   help="Max tokens to generate")
    inference_parser.add_argument("--temperature", type=float, default=0.7,
                                   help="Generation temperature")
    inference_parser.add_argument("--top-p", type=float, default=0.9,
                                   help="Top-p sampling")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--model", type=str,
                                   help="Model to benchmark")
    benchmark_parser.add_argument("--quantization", type=str, default="4bit",
                                   choices=["4bit", "8bit", "fp16", "fp32"],
                                   help="Quantization type")
    benchmark_parser.add_argument("--inference", action="store_true",
                                   help="Run inference benchmark")
    benchmark_parser.add_argument("--max-tokens", type=int, default=100,
                                   help="Max tokens for inference")
    benchmark_parser.add_argument("--output", type=str, default="benchmark_results.json",
                                   help="Output file")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configs and data")
    validate_parser.add_argument("--model-config", type=str, default="config/model_config.yaml",
                                  help="Path to model config")
    validate_parser.add_argument("--training-config", type=str, default="config/training_config.yaml",
                                  help="Path to training config")
    validate_parser.add_argument("--dataset", type=str,
                                  help="Path to dataset to validate")

    # Info command
    info_parser = subparsers.add_parser("info", help="Display system info")
    info_parser.add_argument("--model-config", type=str, default="config/model_config.yaml",
                              help="Path to model config")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export trained model")
    export_parser.add_argument("--base-model", type=str, required=True,
                                help="Base model name or path")
    export_parser.add_argument("--lora-path", type=str,
                                help="Path to LoRA weights")
    export_parser.add_argument("--output", type=str, required=True,
                                help="Output path")
    export_parser.add_argument("--merge", action="store_true",
                                help="Merge LoRA weights with base model")

    args = parser.parse_args()

    # Execute command
    if args.command == "train":
        train_command(args)
    elif args.command == "inference":
        inference_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)
    elif args.command == "validate":
        success = validate_command(args)
        sys.exit(0 if success else 1)
    elif args.command == "info":
        info_command(args)
    elif args.command == "export":
        export_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()