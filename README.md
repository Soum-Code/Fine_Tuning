# Industrial-Grade LoRA Fine-Tuning System

A production-ready system for fine-tuning large language models efficiently using PEFT (LoRA/QLoRA) techniques with comprehensive monitoring, validation, and deployment capabilities.

## Features

- **Memory Efficient Training**: 4-bit/8-bit quantization with LoRA/QLoRA support
- **Batch Processing**: Configurable batch size with gradient accumulation for memory optimization
- **Modular Architecture**: Separate components for data processing, model management, training, and inference
- **API Serving**: Built-in FastAPI server for real-time inference
- **Resource Monitoring**: Real-time tracking of CPU, GPU, and memory usage
- **Checkpoint Management**: Automatic checkpointing with versioning and cleanup
- **Data Validation**: Comprehensive validation for configurations and datasets
- **Multiple Model Support**: Pre-configured support for Qwen, LLaMA, Mistral, and more
- **Prompt Templates**: Built-in templates for various instruction formats (Alpaca, ChatML, etc.)
- **Export & Deployment**: Model export with LoRA weight merging

## Project Structure

```
industrial-lora-finetune/
├── config/
│   ├── model_config.yaml      # Model configurations
│   └── training_config.yaml   # Training hyperparameters
├── src/
│   ├── data_processing/
│   │   ├── dataset_loader.py  # Data loading utilities
│   │   ├── data_formatter.py  # Prompt formatting
│   │   └── data_augmentation.py # Data augmentation
│   ├── model/
│   │   ├── model_manager.py   # Model loading/LoRA
│   │   ├── lora_configurator.py # LoRA presets
│   │   ├── quantizer.py       # Quantization utilities
│   │   └── checkpoint_manager.py # Checkpoint handling
│   ├── training/
│   │   ├── trainer.py         # Main training orchestrator
│   │   └── batch_processor.py # Batch processing
│   ├── inference/
│   │   ├── model_deployer.py  # Model deployment
│   │   └── api_server.py      # FastAPI server
│   └── utils/
│       ├── logger.py          # Logging utilities
│       ├── monitor.py         # Resource monitoring
│       └── validator.py       # Configuration validation
├── scripts/
│   ├── train.sh               # Training script (Unix)
│   ├── train.ps1              # Training script (Windows)
│   ├── deploy.sh              # Deployment script
│   └── benchmark.py           # Benchmarking utilities
├── data/
│   └── training_data.json     # Sample training data
├── main.py                    # Main entry point
└── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a JSON file with instruction-response pairs:

```json
[
  {
    "instruction": "Write a Python function to calculate factorial",
    "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
  },
  {
    "instruction": "Explain what a neural network is",
    "response": "A neural network is a computing system inspired by biological neural networks..."
  }
]
```

### 3. Configure Training

Edit `config/model_config.yaml` to select your base model and LoRA settings:

```yaml
base_models:
  qwen2_5_0_5b:
    model_name: "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_name: "Qwen/Qwen2.5-0.5B-Instruct"

quantization:
  use_4bit: true
  bnb_4bit_quant_type: "nf4"

lora:
  r: 8
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

Edit `config/training_config.yaml` for training parameters:

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32
  num_epochs: 3
  learning_rate: 2e-4
```

### 4. Run Training

**Windows:**
```powershell
.\scripts\train.ps1
```

**Linux/Mac:**
```bash
bash scripts/train.sh
```

Or using the main entry point:
```bash
python main.py train --model qwen2_5_0_5b --dataset ./data/training_data.json
```

### 5. Deploy for Inference

```bash
python main.py inference --model-path ./checkpoints/final_model --interactive
```

Or start the API server:
```bash
python -m src.inference.api_server --model-path ./checkpoints/final_model --port 8000
```

Test the API:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to sort a list", "max_new_tokens": 150}'
```

## CLI Commands

### Training
```bash
python main.py train --model qwen2_5_0_5b --dataset ./data/my_data.json --template alpaca
```

Options:
- `--model`: Model key from config
- `--dataset`: Path to training data
- `--template`: Prompt template (alpaca, chatml, vicuna, qwen)
- `--resume`: Resume from checkpoint
- `--export`: Export model after training
- `--merge`: Merge LoRA weights when exporting

### Inference
```bash
python main.py inference --model-path ./checkpoints/model --interactive
```

Options:
- `--model-path`: Path to trained model
- `--device`: cpu or cuda
- `--quantize`: Enable quantization
- `--interactive`: Interactive mode
- `--prompt`: Single prompt for generation

### Benchmarking
```bash
python main.py benchmark --model Qwen/Qwen2.5-0.5B-Instruct --quantization 4bit
```

### Validation
```bash
python main.py validate --model-config config/model_config.yaml --dataset ./data/training_data.json
```

### System Info
```bash
python main.py info
```

## Configuration Details

### Model Configuration (`config/model_config.yaml`)

```yaml
base_models:
  model_key:
    model_name: "organization/model-name"
    tokenizer_name: "organization/model-name"

quantization:
  use_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true

lora:
  r: 8                    # LoRA rank
  lora_alpha: 32          # LoRA alpha
  lora_dropout: 0.1       # Dropout probability
  bias: "none"            # Bias type
  target_modules:         # Modules to apply LoRA
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
```

### Training Configuration (`config/training_config.yaml`)

```yaml
training:
  batch_size: 1
  gradient_accumulation_steps: 32
  num_epochs: 3
  learning_rate: 2e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0

optimizer:
  name: "adamw_torch"
  lr_scheduler_type: "cosine"

checkpointing:
  save_steps: 500
  save_total_limit: 3

device:
  use_cpu: false
  use_mixed_precision: true
  gradient_checkpointing: true
```

## Prompt Templates

The system supports multiple prompt templates:

- `alpaca`: Standard Alpaca format
- `chatml`: OpenAI ChatML format
- `vicuna`: Vicuna chat format
- `qwen`: Qwen instruction format
- `instruction`: Simple instruction format

## LoRA Presets

Use predefined LoRA configurations:

```python
from src.model.lora_configurator import LoRAConfigurator, LORA_PRESETS

# List available presets
print(LoRAConfigurator.list_presets())
# ['conservative', 'standard', 'aggressive', 'code_finetuning', 'conversation']

# Get recommended preset based on dataset size
from src.model.lora_configurator import get_recommended_preset
preset = get_recommended_preset(dataset_size=5000, task_type="code")
```

## Resource Monitoring

Monitor GPU and CPU usage during training:

```python
from src.utils.monitor import ResourceMonitor, print_memory_status

# Print current status
print_memory_status()

# Start monitoring
monitor = ResourceMonitor(log_interval=1.0)
monitor.start_monitoring()

# Get stats
stats = monitor.get_all_stats()
print(f"CPU: {stats.cpu_percent}%")
print(f"Memory: {stats.memory_percent}%")
print(f"GPU Memory: {stats.gpu_memory_allocated_gb}GB")

# Stop monitoring
monitor.stop_monitoring()
```

## Data Augmentation

Augment your training data:

```python
from src.data_processing.data_augmentation import DataAugmenter, AugmentationConfig

config = AugmentationConfig(
    enabled=True,
    paraphrase_prob=0.3
)

augmenter = DataAugmenter(config)
augmented_data = augmenter.augment_dataset(original_data, augmentation_factor=1.5)
```

## Model Export

Export your trained model:

```bash
# Export with merged LoRA weights
python main.py export --base-model Qwen/Qwen2.5-0.5B-Instruct \
                       --lora-path ./checkpoints/lora_weights \
                       --output ./exported_model \
                       --merge
```

## API Server

The FastAPI server provides endpoints:

- `POST /generate`: Generate text from prompt
- `GET /health`: Health check endpoint

Example request:
```json
{
  "prompt": "Write a Python function to reverse a string",
  "max_new_tokens": 200,
  "temperature": 0.7,
  "top_p": 0.9
}
```

## Best Practices

### Memory Optimization
1. Use 4-bit quantization for models > 7B parameters
2. Set `gradient_checkpointing: true` in training config
3. Use `gradient_accumulation_steps` to simulate larger batch sizes
4. Reduce `max_length` in data formatter if possible

### Training Tips
1. Start with conservative LoRA settings (r=8)
2. Use a small learning rate (2e-4) and adjust based on loss
3. Monitor GPU memory usage with `print_memory_status()`
4. Save checkpoints frequently with `save_steps: 500`

### Data Preparation
1. Validate your data with `python main.py validate`
2. Use appropriate prompt template for your model
3. Clean and deduplicate your training data
4. Consider data augmentation for small datasets

## Troubleshooting

### Out of Memory Errors
- Reduce batch size to 1
- Increase gradient accumulation steps
- Enable gradient checkpointing
- Use 4-bit quantization

### Slow Training
- Check if GPU is being utilized
- Reduce `max_length` in data formatting
- Increase `gradient_accumulation_steps`

### Model Loading Errors
- Ensure sufficient disk space for model cache
- Check internet connection for first-time downloads
- Verify model name in configuration

## License

This project is provided for educational and research purposes. Please ensure you comply with the licenses of the base models you use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.