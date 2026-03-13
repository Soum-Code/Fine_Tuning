# Examples for Industrial LoRA Fine-Tuning System

This directory contains example scripts demonstrating various features:

## example_usage.py

Comprehensive demonstration of:
- Configuration validation
- Dataset validation
- LoRA presets
- Prompt templates
- Data processing
- Resource monitoring
- Model information utilities
- Checkpoint management

Run with:
```bash
python examples/example_usage.py
```

## simple_train.py

Minimal training example showing the basic workflow.

Run with:
```bash
python examples/simple_train.py
```

## inference_example.py

Example showing how to load a trained model and run inference.

Run with:
```bash
python examples/inference_example.py --model-path ./checkpoints/final_model
```