# test_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_fine_tuned_model():
    # Load the fine-tuned model
    print("Loading fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

    # Test prompt
    prompt = "<|user|>\nWrite a Python function to multiply two numbers\n<|assistant|>"

    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    print("Generating response...")
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated response:")
    print("-" * 20)
    print(response)
    print("-" * 20)

if __name__ == "__main__":
    test_fine_tuned_model()
