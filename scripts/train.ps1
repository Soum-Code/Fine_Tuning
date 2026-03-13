# Set environment variables
$env:CUDA_VISIBLE_DEVICES = "0"
$env:WANDB_PROJECT = "industrial-lora-finetune"
$env:WANDB_MODE = "disabled"
$env:TRANSFORMERS_CACHE = "./cache"

# Run training
python src/training/trainer.py `
  --model qwen2_5_0_5b `
  --dataset ./data/training_data.json
