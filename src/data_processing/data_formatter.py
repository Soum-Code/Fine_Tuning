"""
Data formatting and preprocessing utilities for training data.
Handles various data formats and prompt templates.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Template for formatting prompts."""
    name: str
    system_prompt: str = ""
    user_prefix: str = "### Instruction:\n"
    user_suffix: str = "\n\n"
    response_prefix: str = "### Response:\n"
    response_suffix: str = ""
    eos_token: str = "</s>"

    def format(self, instruction: str, response: str = "") -> str:
        """Format instruction-response pair."""
        text = ""
        if self.system_prompt:
            text += self.system_prompt + "\n\n"
        text += self.user_prefix + instruction + self.user_suffix
        text += self.response_prefix + response
        if response and self.response_suffix:
            text += self.response_suffix
        text += self.eos_token
        return text


# Common prompt templates
PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "alpaca": PromptTemplate(
        name="alpaca",
        system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        user_prefix="### Instruction:\n",
        user_suffix="\n\n",
        response_prefix="### Response:\n",
        response_suffix="",
        eos_token="</s>"
    ),
    "chatml": PromptTemplate(
        name="chatml",
        system_prompt="",
        user_prefix="<|im_start|>user\n",
        user_suffix="<|im_end|>\n",
        response_prefix="<|im_start|>assistant\n",
        response_suffix="<|im_end|>\n",
        eos_token=""
    ),
    "vicuna": PromptTemplate(
        name="vicuna",
        system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        user_prefix="USER: ",
        user_suffix="\n",
        response_prefix="ASSISTANT: ",
        response_suffix="</s>",
        eos_token=""
    ),
    "instruction": PromptTemplate(
        name="instruction",
        system_prompt="",
        user_prefix="### Instruction:\n",
        user_suffix="\n\n",
        response_prefix="### Output:\n",
        response_suffix="",
        eos_token=""
    ),
    "qwen": PromptTemplate(
        name="qwen",
        system_prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
        user_prefix="<|im_start|>user\n",
        user_suffix="<|im_end|>\n",
        response_prefix="<|im_start|>assistant\n",
        response_suffix="",
        eos_token="<|im_end|>"
    ),
    "aot": PromptTemplate(
        name="aot",
        system_prompt="Decompose the following problem into atomic reasoning steps. Use <thought> for initial decomposition, <atom> for each self-contained reasoning step, and <final_answer> for the conclusion.",
        user_prefix="### Question:\n",
        user_suffix="\n\n",
        response_prefix="### Reasoning Chain:\n",
        response_suffix="",
        eos_token="</s>"
    )
}


class DataFormatter:
    """
    Format and preprocess training data for different model types.
    """

    def __init__(
        self,
        template: Union[str, PromptTemplate] = "alpaca",
        max_length: int = 512,
        truncate_strategy: str = "end"
    ):
        """
        Initialize data formatter.

        Args:
            template: Template name or PromptTemplate object
            max_length: Maximum sequence length
            truncate_strategy: How to truncate ('start', 'middle', 'end')
        """
        if isinstance(template, str):
            if template not in PROMPT_TEMPLATES:
                available = list(PROMPT_TEMPLATES.keys())
                raise ValueError(f"Unknown template '{template}'. Available: {available}")
            self.template = PROMPT_TEMPLATES[template]
        else:
            self.template = template

        self.max_length = max_length
        self.truncate_strategy = truncate_strategy

    def format_sample(
        self,
        instruction: str,
        response: str,
        context: Optional[str] = None
    ) -> str:
        """
        Format a single training sample.

        Args:
            instruction: The instruction/prompt
            response: The expected response
            context: Optional context/instruction

        Returns:
            Formatted text
        """
        # Add context if provided
        if context:
            instruction = f"{context}\n\n{instruction}"

        return self.template.format(instruction, response)

    def format_batch(
        self,
        samples: List[Dict[str, str]],
        add_context: bool = False
    ) -> List[str]:
        """
        Format a batch of samples.

        Args:
            samples: List of dictionaries with 'instruction' and 'response'
            add_context: Whether to add context from 'context' key

        Returns:
            List of formatted texts
        """
        formatted = []
        for sample in samples:
            instruction = sample.get('instruction', '')
            response = sample.get('response', '')
            context = sample.get('context') if add_context else None

            formatted.append(self.format_sample(instruction, response, context))

        return formatted

    def truncate_text(self, text: str, max_length: int = None) -> str:
        """
        Truncate text to maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length (uses self.max_length if None)

        Returns:
            Truncated text
        """
        max_len = max_length or self.max_length

        if len(text) <= max_len:
            return text

        if self.truncate_strategy == "end":
            return text[:max_len]
        elif self.truncate_strategy == "start":
            return text[-max_len:]
        elif self.truncate_strategy == "middle":
            half = max_len // 2
            return text[:half] + text[-(max_len - half):]
        else:
            return text[:max_len]

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (optional)
        # text = re.sub(r'[^\w\s.,!?\'"-]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text


class DatasetPreprocessor:
    """
    Preprocess datasets for training.
    """

    def __init__(
        self,
        formatter: DataFormatter,
        min_length: int = 10,
        max_length: int = 2048,
        remove_duplicates: bool = True
    ):
        """
        Initialize preprocessor.

        Args:
            formatter: DataFormatter instance
            min_length: Minimum text length
            max_length: Maximum text length
            remove_duplicates: Whether to remove duplicates
        """
        self.formatter = formatter
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates

    def preprocess(
        self,
        samples: List[Dict[str, str]],
        clean: bool = True,
        validate: bool = True
    ) -> List[Dict[str, str]]:
        """
        Preprocess a list of samples.

        Args:
            samples: List of sample dictionaries
            clean: Whether to clean text
            validate: Whether to validate samples

        Returns:
            Preprocessed samples
        """
        processed = []

        # Remove duplicates if requested
        if self.remove_duplicates:
            seen = set()
            unique_samples = []
            for sample in samples:
                key = sample.get('instruction', '') + sample.get('response', '')
                if key not in seen:
                    seen.add(key)
                    unique_samples.append(sample)
            samples = unique_samples

        for sample in samples:
            # Clean text if requested
            if clean:
                instruction = self.formatter.clean_text(sample.get('instruction', ''))
                response = self.formatter.clean_text(sample.get('response', ''))
            else:
                instruction = sample.get('instruction', '')
                response = sample.get('response', '')

            # Validate
            if validate:
                if not instruction or not response:
                    logger.warning(f"Skipping sample with empty instruction or response")
                    continue

                if len(instruction) < self.min_length or len(response) < self.min_length:
                    logger.warning(f"Skipping sample with short text")
                    continue

            # Format
            formatted_text = self.formatter.format_sample(instruction, response)

            # Truncate if needed
            if len(formatted_text) > self.max_length:
                formatted_text = self.formatter.truncate_text(formatted_text, self.max_length)

            processed.append({
                'instruction': instruction,
                'response': response,
                'text': formatted_text
            })

        logger.info(f"Preprocessed {len(processed)} samples (from {len(samples)} input)")
        return processed

    def split_dataset(
        self,
        samples: List[Dict],
        train_ratio: float = 0.9,
        shuffle: bool = True,
        seed: int = 42
    ) -> tuple:
        """
        Split dataset into train and validation sets.

        Args:
            samples: List of samples
            train_ratio: Ratio for training set
            shuffle: Whether to shuffle before splitting
            seed: Random seed

        Returns:
            Tuple of (train_samples, val_samples)
        """
        import random

        if shuffle:
            random.seed(seed)
            samples = samples.copy()
            random.shuffle(samples)

        split_idx = int(len(samples) * train_ratio)

        return samples[:split_idx], samples[split_idx:]


def load_and_format_dataset(
    file_path: str,
    template: str = "alpaca",
    output_format: str = "text"
) -> List[Dict[str, str]]:
    """
    Load and format dataset from file.

    Args:
        file_path: Path to data file (JSON or CSV)
        template: Prompt template to use
        output_format: Output format ('text' or 'dict')

    Returns:
        List of formatted samples
    """
    # Load data
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Format data
    formatter = DataFormatter(template=template)
    preprocessor = DatasetPreprocessor(formatter)

    processed = preprocessor.preprocess(data)

    if output_format == "text":
        return [s['text'] for s in processed]
    else:
        return processed


def save_processed_dataset(
    samples: List[Dict],
    output_path: str,
    format: str = "json"
):
    """
    Save processed dataset to file.

    Args:
        samples: List of processed samples
        output_path: Output file path
        format: Output format ('json' or 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    elif format == "csv":
        df = pd.DataFrame(samples)
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved {len(samples)} samples to {output_path}")