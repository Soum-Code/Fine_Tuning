
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

def test_aot_baseline():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading baseline model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    prompt = """<thought>
The user wants to find the sum of 15 and 27.
I need to split this into atomic steps.
</thought>
<atom>
Step 1: Identify the numbers. The numbers are 15 and 27.
</atom>
<atom>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating completion...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print("\n--- BASELINE GENERATION ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("---------------------------\n")

if __name__ == "__main__":
    test_aot_baseline()
