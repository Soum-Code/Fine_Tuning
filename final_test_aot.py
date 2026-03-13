
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def final_test():
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Dynamically find the adapter path
    potential_dirs = ["checkpoints/aot_smoke/checkpoint-1", "checkpoints/aot_smoke"]
    adapter_path = None
    
    for d in potential_dirs:
        if os.path.exists(d):
            adapter_path = d
            break
            
    if not adapter_path:
        print("Listing checkpoints directory to find the right path:")
        if os.path.exists("checkpoints"):
            print(os.listdir("checkpoints"))
            # Search recursively for adapter_config.json
            for root, dirs, files in os.walk("checkpoints"):
                if "adapter_config.json" in files:
                    adapter_path = root
                    print(f"Found adapter at: {adapter_path}")
                    break
    
    if not adapter_path:
        print("Error: Could not find any adapter checkpoint.")
        return
    
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    prompt = """<|user|>
What is 15 + 27?
<|assistant|>
<thought>
To solve 15 + 27, I will split it into atomic steps: adding the tens and then the ones.
</thought>
<atom>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating AoT response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print("\n--- FINE-TUNED AoT GENERATION ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("----------------------------------\n")

if __name__ == "__main__":
    final_test()
