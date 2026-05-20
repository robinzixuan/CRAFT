"""
Evaluate a model against the JailbreakBench dataset.
Supports local checkpoints or HuggingFace model IDs.
Uses transformers to load the model (avoids vLLM compatibility issues).
"""

import os
import json
import glob
import pathlib
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import jailbreakbench as jbb
from jailbreakbench.classifier import Llama3JailbreakJudge, Llama3RefusalJudge, StringClassifier
from jailbreakbench.config import TEMPERATURE, TOP_P, MAX_GENERATION_LENGTH

# Load paths/credentials from config.yaml at the repo root (two levels up from this file)
_CFG_PATH = pathlib.Path(__file__).parent.parent.parent / "config.yaml"
if _CFG_PATH.exists():
    _cfg = yaml.safe_load(_CFG_PATH.read_text())
    MODEL_PATH = _cfg.get("paths", {}).get("craft_model", "./checkpoints/qwen3_4b_thinking_craft")
    EVAL_OUTPUT_DIR = _cfg.get("paths", {}).get("eval_output", "./outputs/eval")
    _OPENAI_KEY = _cfg.get("credentials", {}).get("openai_api_key", "")
    if _OPENAI_KEY:
        os.environ.setdefault("OPENAI_API_KEY", _OPENAI_KEY)
else:
    MODEL_PATH = os.environ.get("CRAFT_MODEL_PATH", "./checkpoints/qwen3_4b_thinking_craft")
    EVAL_OUTPUT_DIR = os.environ.get("CRAFT_EVAL_OUTPUT", "./outputs/eval")

# API key (used for judge evaluation)
API_KEY = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Detect if this is a HuggingFace model ID (contains '/' and is not a local path)
is_hf_model_id = "/" in MODEL_PATH and not os.path.exists(MODEL_PATH)

# Basic validation: if it's a local path, check that required files exist
if not is_hf_model_id:
    if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        raise RuntimeError(f"config.json not found; please verify the HF-format directory: {MODEL_PATH}")
    weight_files = []
    for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.safetensors"):
        weight_files.extend(glob.glob(os.path.join(MODEL_PATH, pattern)))
    if not weight_files:
        raise RuntimeError(f"No HF weight files found; please export the checkpoint to HF format: {MODEL_PATH}")
else:
    print(f"Detected HuggingFace model ID: {MODEL_PATH}")
    print("Model files will be downloaded automatically from HuggingFace Hub")

# Load model with transformers
print(f"Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# Load chat_template if the tokenizer does not have one
if tokenizer.chat_template is None and not is_hf_model_id:
    template_path = os.path.join(MODEL_PATH, "chat_template.jinja")
    if os.path.isfile(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()

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

# Query model in batch
print(f"Starting query for {len(goals)} prompts...")
responses = query_model(goals)
print("Query complete!")

# Save model responses first (so results are accessible even if evaluation fails)
print("\n=== Saving Model Responses ===")
responses_output_path = os.path.join(EVAL_OUTPUT_DIR, "jailbreakbench", "responses.json")
os.makedirs(os.path.dirname(responses_output_path), exist_ok=True)
with open(responses_output_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_path": MODEL_PATH,
        "total": len(goals),
        "data": [{"behavior": b, "goal": g, "response": r} for b, g, r in zip(behaviors, goals, responses)]
    }, f, ensure_ascii=False, indent=2)
print(f"Model responses saved to: {responses_output_path}")
