import json
import random
import os
from typing import List, Dict, Any

def generate_math_aot_trace(a: int, b: int, operator: str = "*") -> Dict[str, str]:
    """Generates a synthetic AoT trace for simple math."""
    if operator == "*":
        instruction = f"What is {a} * {b}?"
        # Decompose multiplication: (a * 10) + (a * (b-10))
        b_ten = 10
        b_rem = b - 10
        
        thought = f"To solve {a} * {b}, I will decompose the multiplication into {a} * {b_ten} and {a} * {b_rem} and then add the results."
        atom1 = f"{a} * {b_ten} = {a * b_ten}"
        atom2 = f"{a} * {b_rem} = {a * b_rem}"
        atom3 = f"{a * b_ten} + {a * b_rem} = {a * b}"
        final = str(a * b)
        
        response = f"<thought> {thought} </thought> <atom> {atom1} </atom> <atom> {atom2} </atom> <atom> {atom3} </atom> <final_answer> {final} </final_answer>"
        
        return {"instruction": instruction, "response": response}
    return {}

def main():
    dataset = []
    print("Generating synthetic AoT math dataset...")
    
    # Generate 100 multiplication samples
    for _ in range(100):
        a = random.randint(11, 20)
        b = random.randint(11, 25)
        sample = generate_math_aot_trace(a, b)
        if sample:
            dataset.append(sample)
            
    os.makedirs("data", exist_ok=True)
    output_path = "data/aot_research_data.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Generated {len(dataset)} AoT samples in {output_path}")

if __name__ == "__main__":
    main()
