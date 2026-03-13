
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def poll_and_test():
    checkpoint_dir = "checkpoints/20260313_155344"
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Watching {checkpoint_dir} for new checkpoints...")
    
    while True:
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Found checkpoint: {checkpoint_path}. Running test...")
            
            # Load
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(model, checkpoint_path)
            
            prompt = """<thought>
The user wants to find the sum of 15 and 27.
I need to split this into atomic steps.
</thought>
<atom>
Step 1: Identify the numbers. The numbers are 15 and 27.
</atom>
<atom>
"""
            
            print("Generating fine-tuned response...")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            print("\n--- FINE-TUNED GENERATION (AoT) ---")
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print("----------------------------------\n")
            break
            
        time.sleep(10)

if __name__ == "__main__":
    poll_and_test()
