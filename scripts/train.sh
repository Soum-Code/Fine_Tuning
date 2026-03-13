#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=industrial-lora-finetune
export WANDB_MODE=disabled
export TRANSFORMERS_CACHE=./cache

# Run training
python src/training/trainer.py \
  --model qwen3_coder_30b \
  --dataset ./data/training_data.json
