#!/usr/bin/env python3
"""
Benchmark script for measuring training and inference performance.
"""

import argparse
import time
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.monitor import ResourceMonitor, get_system_summary
from src.utils.logger import setup_logger

logger = setup_logger("benchmark")


def benchmark_model_loading(model_name: str, quantization: str = "4bit"):
    """Benchmark model loading time and memory."""
    logger.info(f"Benchmarking model loading: {model_name} ({quantization})")

    # Get initial memory state
    initial_stats = ResourceMonitor.get_system_stats()
    gpu_stats = ResourceMonitor.get_gpu_stats()

    start_time = time.time()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        # Configure quantization
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None

        # Load model
        if quant_config:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        load_time = time.time() - start_time

        # Get final memory state
        final_stats = ResourceMonitor.get_system_stats()
        final_gpu_stats = ResourceMonitor.get_gpu_stats()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results = {
            "model_name": model_name,
            "quantization": quantization,
            "load_time_seconds": load_time,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory": {
                "initial_cpu_memory_percent": initial_stats["memory_percent"],
                "final_cpu_memory_percent": final_stats["memory_percent"],
                "initial_gpu_memory_gb": gpu_stats.get("gpu_memory_allocated_gb", 0),
                "final_gpu_memory_gb": final_gpu_stats.get("gpu_memory_allocated_gb", 0)
            }
        }

        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")
        return {"error": str(e)}


def benchmark_inference(
    model_name: str,
    prompts: list,
    max_new_tokens: int = 100,
    quantization: str = "4bit"
):
    """Benchmark inference speed."""
    logger.info(f"Benchmarking inference: {model_name}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        import torch

        # Configure quantization
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = None

        # Load model
        if quant_config:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        results = []

        for prompt in prompts:
            start_time = time.time()

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7
                )

            inference_time = time.time() - start_time
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])

            results.append({
                "prompt_length": len(tokenizer.encode(prompt)),
                "generated_tokens": tokens_generated,
                "inference_time_seconds": inference_time,
                "tokens_per_second": tokens_generated / inference_time if inference_time > 0 else 0,
                "generated_text_length": len(generated_text)
            })

        # Calculate averages
        avg_time = sum(r["inference_time_seconds"] for r in results) / len(results)
        avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)

        summary = {
            "model_name": model_name,
            "quantization": quantization,
            "num_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
            "average_inference_time": avg_time,
            "average_tokens_per_second": avg_tps,
            "detailed_results": results
        }

        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return summary

    except Exception as e:
        logger.error(f"Error during inference benchmark: {str(e)}")
        return {"error": str(e)}


def benchmark_memory():
    """Benchmark current system memory status."""
    return {
        "system_summary": get_system_summary(),
        "detailed_stats": ResourceMonitor.get_system_stats(),
        "gpu_stats": ResourceMonitor.get_gpu_stats()
    }


def run_full_benchmark(config: dict):
    """Run complete benchmark suite."""
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory_benchmark": benchmark_memory()
    }

    if config.get("model_name"):
        model_name = config["model_name"]
        quantization = config.get("quantization", "4bit")

        results["loading_benchmark"] = benchmark_model_loading(model_name, quantization)

        if config.get("benchmark_inference", False):
            test_prompts = config.get("test_prompts", [
                "Write a Python function to sort a list.",
                "Explain what machine learning is.",
                "Create a simple REST API in Python."
            ])
            results["inference_benchmark"] = benchmark_inference(
                model_name,
                test_prompts,
                config.get("max_new_tokens", 100),
                quantization
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark model performance")
    parser.add_argument("--model", type=str, help="Model name to benchmark")
    parser.add_argument("--quantization", type=str, default="4bit",
                       choices=["4bit", "8bit", "fp16", "fp32"],
                       help="Quantization type")
    parser.add_argument("--inference", action="store_true",
                       help="Run inference benchmark")
    parser.add_argument("--memory-only", action="store_true",
                       help="Only run memory benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Max tokens for inference benchmark")

    args = parser.parse_args()

    logger.info("Starting benchmark...")

    if args.memory_only:
        results = benchmark_memory()
    else:
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

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    if "memory_benchmark" in results:
        print(f"\nMemory Status: {results['memory_benchmark']['system_summary']}")

    if "loading_benchmark" in results:
        lb = results["loading_benchmark"]
        print(f"\nModel Loading:")
        print(f"  Model: {lb.get('model_name', 'N/A')}")
        print(f"  Quantization: {lb.get('quantization', 'N/A')}")
        print(f"  Load Time: {lb.get('load_time_seconds', 0):.2f}s")
        print(f"  Parameters: {lb.get('total_parameters', 0):,}")

    if "inference_benchmark" in results:
        ib = results["inference_benchmark"]
        print(f"\nInference Performance:")
        print(f"  Average Time: {ib.get('average_inference_time', 0):.2f}s")
        print(f"  Average Speed: {ib.get('average_tokens_per_second', 0):.1f} tokens/s")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()