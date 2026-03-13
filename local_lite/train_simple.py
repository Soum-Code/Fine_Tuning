# train_simple.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json
from tqdm import tqdm
import os

def create_sample_data():
    """Create minimal sample data for testing"""
    os.makedirs("data", exist_ok=True)
    data = [
        {"text": "<|user|>\nWrite a Python function to add two numbers\n<|assistant|>\ndef add_numbers(a, b):\n    return a + b"},
        {"text": "<|user|>\nWhat is machine learning?\n<|assistant|>\nMachine learning is a subset of artificial intelligence..."},
        {"text": "<|user|>\nExplain neural networks\n<|assistant|>\nNeural networks are computing systems inspired by biological neural networks..."}
    ] * 100  # Repeat for more training samples

    with open("data/sample_data.json", "w") as f:
        json.dump(data, f)

    return data

def load_model_and_tokenizer():
    """Load a smaller model for local testing"""
    print("Loading model... This may take a few minutes...")

    # Use a smaller model for local testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Only 1.1B parameters!

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with CPU offloading for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use FP32 for CPU compatibility
        low_cpu_mem_usage=True,
        device_map="auto"  # Automatically manage device placement
    )

    return model, tokenizer

def apply_lora(model):
    """Apply LoRA for parameter-efficient fine-tuning"""
    lora_config = LoraConfig(
        r=8,  # Low rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Only tune specific layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    return get_peft_model(model, lora_config)

def tokenize_data(data, tokenizer, max_length=256):
    """Tokenize the dataset"""
    texts = [item["text"] for item in data]

    def tokenize_function(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

    # Process in small batches to save memory
    tokenized_data = []
    for i in tqdm(range(0, len(texts), 10), desc="Tokenizing"):
        batch_texts = texts[i:i+10]
        tokenized_batch = tokenize_function(batch_texts)
        tokenized_data.extend([
            {
                "input_ids": tokenized_batch["input_ids"][j],
                "attention_mask": tokenized_batch["attention_mask"][j]
            }
            for j in range(len(batch_texts))
        ])

    return Dataset.from_list(tokenized_data)

def train_model(model, tokenized_dataset, tokenizer):
    """Simple training loop"""
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    import torch.nn.functional as F

    # Prepare model for training
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-4)

    # Create data loader with very small batch size
    dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True)

    print("Starting training...")
    for epoch in range(2):  # Just 2 epochs for demo
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Move to device
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Stop early for demo
            if step >= 5:  # Only train on 5 batches for quick demo
                break

        print(f"Epoch {epoch+1} Average Loss: {total_loss/(step+1):.4f}")

    # Save the fine-tuned model
    os.makedirs("./fine_tuned_model", exist_ok=True)
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Model saved to ./fine_tuned_model")

def main():
    """Main training function"""
    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Create sample data
    print("Creating sample data...")
    data = create_sample_data()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # Apply LoRA
    print("Applying LoRA...")
    model = apply_lora(model)
    print(f"Trainable parameters: {model.print_trainable_parameters()}")

    # Tokenize data
    print("Tokenizing data...")
    tokenized_dataset = tokenize_data(data, tokenizer)

    # Training model
    print("Training model...")
    train_model(model, tokenized_dataset, tokenizer)

    print("Training completed!")

if __name__ == "__main__":
    main()
