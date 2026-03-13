import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    pipeline, TextIteratorStreamer
)
from threading import Thread
from typing import Generator, Dict, Any

class ModelDeployer:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load_model(self, quantize: bool = True):
        """Load model for inference with optional quantization"""
        print(f"Loading model from {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        if self.device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if quantize else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device != "cpu" else -1
        )

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text with configurable parameters"""
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        default_params = {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        default_params.update(kwargs)

        result = self.pipeline(prompt, **default_params)
        return result[0]['generated_text']

    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream generation for real-time response"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            do_sample=True
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text
