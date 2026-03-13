"""
Data augmentation utilities for training data.
Provides various augmentation strategies for instruction tuning.
"""

import random
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    enabled: bool = True
    paraphrase_prob: float = 0.0
    noise_prob: float = 0.0
    shuffle_prob: float = 0.0
    min_augmentations: int = 0
    max_augmentations: int = 2


class DataAugmenter:
    """
    Augment training data with various techniques.

    Note: This provides template-based augmentation.
    For more sophisticated augmentation, consider using external NLP libraries.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmenter.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()

        # Instruction templates for augmentation
        self.instruction_prefixes = [
            "Please ",
            "Can you ",
            "Could you ",
            "Help me ",
            "I need you to ",
            "",
        ]

        self.instruction_suffixes = [
            "",
            " Please be detailed.",
            " Be concise.",
            " Explain step by step.",
        ]

        # Common paraphrase patterns
        self.paraphrase_patterns = {
            "write": ["create", "develop", "compose", "draft"],
            "create": ["write", "build", "develop", "make"],
            "explain": ["describe", "clarify", "elaborate on", "detail"],
            "find": ["locate", "identify", "discover", "search for"],
            "fix": ["repair", "correct", "resolve", "debug"],
            "implement": ["code", "develop", "build", "create"],
            "optimize": ["improve", "enhance", "refine", "speed up"],
            "analyze": ["examine", "study", "investigate", "evaluate"],
        }

    def augment_instruction(self, instruction: str) -> str:
        """
        Augment instruction text.

        Args:
            instruction: Original instruction

        Returns:
            Augmented instruction
        """
        if not self.config.enabled:
            return instruction

        # Add random prefix
        if random.random() < 0.3:
            prefix = random.choice(self.instruction_prefixes)
            instruction = prefix + instruction.lower()

        # Add random suffix
        if random.random() < 0.2:
            suffix = random.choice(self.instruction_suffixes)
            instruction = instruction.rstrip('.') + suffix

        # Apply paraphrase
        if random.random() < self.config.paraphrase_prob:
            instruction = self._paraphrase(instruction)

        return instruction

    def _paraphrase(self, text: str) -> str:
        """Apply simple paraphrase transformations."""
        for word, synonyms in self.paraphrase_patterns.items():
            if word in text.lower():
                # Randomly replace with synonym
                if random.random() < 0.3:
                    synonym = random.choice(synonyms)
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    text = pattern.sub(synonym, text, count=1)
                    break
        return text

    def augment_sample(
        self,
        sample: Dict[str, str],
        num_variations: int = 1
    ) -> List[Dict[str, str]]:
        """
        Create augmented variations of a sample.

        Args:
            sample: Original sample with 'instruction' and 'response'
            num_variations: Number of variations to create

        Returns:
            List of augmented samples (includes original)
        """
        if not self.config.enabled:
            return [sample]

        variations = [sample]

        for _ in range(num_variations):
            aug_instruction = self.augment_instruction(sample['instruction'])
            variations.append({
                'instruction': aug_instruction,
                'response': sample['response'],
                'augmented': True
            })

        return variations

    def augment_dataset(
        self,
        samples: List[Dict[str, str]],
        augmentation_factor: float = 1.5
    ) -> List[Dict[str, str]]:
        """
        Augment entire dataset.

        Args:
            samples: Original samples
            augmentation_factor: Factor to increase dataset size

        Returns:
            Augmented dataset
        """
        if not self.config.enabled:
            return samples

        target_size = int(len(samples) * augmentation_factor)
        augmented = list(samples)

        while len(augmented) < target_size:
            # Pick random sample to augment
            sample = random.choice(samples)

            # Create one augmentation
            variations = self.augment_sample(sample, num_variations=1)
            augmented.extend(variations[1:])  # Skip original

        # Shuffle
        random.shuffle(augmented)

        logger.info(f"Augmented dataset from {len(samples)} to {len(augmented)} samples")
        return augmented


class InstructionTemplateGenerator:
    """
    Generate instruction variations from templates.
    """

    def __init__(self):
        """Initialize template generator."""
        self.templates = {
            "code_generation": [
                "Write a {language} function to {task}",
                "Create a {language} program that {task}",
                "Implement {language} code for: {task}",
                "Code a {language} solution for {task}",
            ],
            "explanation": [
                "Explain {topic}",
                "What is {topic}?",
                "Describe {topic} in detail",
                "Provide an explanation of {topic}",
            ],
            "debugging": [
                "Fix the bug in this {language} code: {code}",
                "Debug and correct this {language} snippet: {code}",
                "Find and fix errors in: {code}",
            ],
            "transformation": [
                "Convert this {from_format} to {to_format}: {input}",
                "Transform {from_format} into {to_format}: {input}",
                "Change the format from {from_format} to {to_format}: {input}",
            ]
        }

    def generate(
        self,
        template_type: str,
        **kwargs
    ) -> str:
        """
        Generate instruction from template.

        Args:
            template_type: Type of template
            **kwargs: Template variables

        Returns:
            Generated instruction
        """
        if template_type not in self.templates:
            logger.warning(f"Unknown template type: {template_type}")
            return ""

        template = random.choice(self.templates[template_type])
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return template

    def add_template(self, template_type: str, templates: List[str]):
        """
        Add custom templates.

        Args:
            template_type: Type identifier
            templates: List of template strings
        """
        if template_type in self.templates:
            self.templates[template_type].extend(templates)
        else:
            self.templates[template_type] = templates


class QualityFilter:
    """
    Filter training data based on quality criteria.
    """

    def __init__(
        self,
        min_instruction_length: int = 5,
        min_response_length: int = 10,
        max_response_length: int = 4096,
        remove_duplicates: bool = True,
        remove_empty: bool = True
    ):
        """
        Initialize quality filter.

        Args:
            min_instruction_length: Minimum instruction length
            min_response_length: Minimum response length
            max_response_length: Maximum response length
            remove_duplicates: Remove duplicate samples
            remove_empty: Remove empty samples
        """
        self.min_instruction_length = min_instruction_length
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.remove_duplicates = remove_duplicates
        self.remove_empty = remove_empty

    def filter(self, samples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter samples based on quality criteria.

        Args:
            samples: Input samples

        Returns:
            Filtered samples
        """
        filtered = []
        seen = set()

        stats = {
            'total': len(samples),
            'empty': 0,
            'too_short': 0,
            'too_long': 0,
            'duplicates': 0,
            'passed': 0
        }

        for sample in samples:
            instruction = sample.get('instruction', '')
            response = sample.get('response', '')

            # Remove empty
            if self.remove_empty and (not instruction or not response):
                stats['empty'] += 1
                continue

            # Check lengths
            if len(instruction) < self.min_instruction_length:
                stats['too_short'] += 1
                continue

            if len(response) < self.min_response_length:
                stats['too_short'] += 1
                continue

            if len(response) > self.max_response_length:
                stats['too_long'] += 1
                continue

            # Check duplicates
            if self.remove_duplicates:
                key = instruction + response
                if key in seen:
                    stats['duplicates'] += 1
                    continue
                seen.add(key)

            filtered.append(sample)
            stats['passed'] += 1

        # Log statistics
        logger.info(f"Quality filter results: {stats}")

        return filtered


def create_augmented_dataset(
    input_file: str,
    output_file: str,
    config: Optional[AugmentationConfig] = None
) -> int:
    """
    Create augmented dataset from input file.

    Args:
        input_file: Input JSON file
        output_file: Output JSON file
        config: Augmentation configuration

    Returns:
        Number of samples in output
    """
    import json

    # Load input
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Augment
    augmenter = DataAugmenter(config)
    augmented = augmenter.augment_dataset(samples, augmentation_factor=2.0)

    # Apply quality filter
    quality_filter = QualityFilter()
    filtered = quality_filter.filter(augmented)

    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    logger.info(f"Created augmented dataset: {len(filtered)} samples")
    return len(filtered)