"""
Evaluate a model against the JailbreakBench dataset.
Uses transformers to load the model.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import jailbreakbench as jbb
from jailbreakbench.config import TEMPERATURE, TOP_P, MAX_GENERATION_LENGTH

# Model name
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"

# Load model with transformers
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# Load JailbreakBench dataset
dataset = jbb.read_dataset()
goals = dataset.goals
behaviors = dataset.behaviors

def query_model(prompts: list[str], batch_size: int = 4) -> list[str]:
    """Query the model using transformers."""
    responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        formatted_prompts = []
        
        for prompt in batch_prompts:
            messages = [
                {"role": "system", "content": 'Solve the following problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the final response to the problem.'},
                {"role": "user", "content": prompt}
            ]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)
        
        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            if TEMPERATURE > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            responses.append(response.strip())
    
    return responses

# Query model
print(f"Starting query for {len(goals)} prompts...")
responses = query_model(goals)
print("Query complete!")

# Save model responses
print("\n=== Saving Model Responses ===")
responses_output_path = "./outputs/eval/jailbreakbench/jbb_qwen_base_responses.json"
os.makedirs(os.path.dirname(responses_output_path), exist_ok=True)
with open(responses_output_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "total": len(goals),
        "data": [{"behavior": b, "goal": g, "response": r} for b, g, r in zip(behaviors, goals, responses)]
    }, f, ensure_ascii=False, indent=2)
print(f"Model responses saved to: {responses_output_path}")
