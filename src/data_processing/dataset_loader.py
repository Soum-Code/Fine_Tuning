from datasets import Dataset, load_dataset
from typing import List, Dict, Union
import pandas as pd

class DatasetLoader:
    def __init__(self):
        pass

    def load_from_file(self, file_path: str, format: str = "json") -> Dataset:
        """Load dataset from various formats"""
        if format == "json":
            return load_dataset("json", data_files=file_path)['train']
        elif format == "csv":
            df = pd.read_csv(file_path)
            return Dataset.from_pandas(df)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def format_as_instruction_dataset(self, dataset: Dataset) -> Dataset:
        """Format an existing dataset into instruction-response pairs"""
        def format_prompt(example):
            return {
                "text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}"
            }
        return dataset.map(format_prompt)

    def create_instruction_dataset(self, instructions: List[Dict]) -> Dataset:
        """Create dataset from instruction pairs"""
        def format_prompt(example):
            return {
                "text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}"
            }

        dataset = Dataset.from_list(instructions)
        return dataset.map(format_prompt)

    def tokenize_dataset(self, dataset: Dataset, tokenizer, max_length: int = 512):
        """Tokenize dataset for training"""
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return dataset.map(tokenize_function, batched=True)
