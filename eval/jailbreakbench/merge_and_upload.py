"""
Merge LoRA adapter into the base model and upload to HuggingFace Hub.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi, login

# Configuration
BASE_MODEL_PATH = "99sweetcookie/reasoningshield-stage1-sft"
LORA_PATH = "99sweetcookie/reasoningshield-stage2-dpo"
MERGED_MODEL_PATH = "./merged_reasoningshield_dpo"  # local save path
HF_REPO_NAME = "YOUR_USERNAME/reasoningshield-stage2-dpo-merged"  # replace with your HF username

def merge_model():
    """Merge LoRA adapter into base model."""
    print(f"Loading base model: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA adapter: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    print("Merging model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {MERGED_MODEL_PATH}")
    merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    print("Model merge complete!")
    return MERGED_MODEL_PATH

def upload_to_hf(model_path, repo_name):
    """Upload model to HuggingFace Hub."""
    # Check for authentication token
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("Please set the HF_TOKEN environment variable or run huggingface-cli login")
        return
    
    api = HfApi()
    
    print(f"Uploading model to: {repo_name}")
    api.create_repo(repo_name, exist_ok=True)
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        commit_message="Upload merged reasoningshield-stage2-dpo model"
    )
    print(f"Upload complete! Model URL: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-only", action="store_true", help="Only merge model, skip upload")
    parser.add_argument("--upload-only", action="store_true", help="Only upload already-merged model")
    parser.add_argument("--repo", type=str, default=HF_REPO_NAME, help="HuggingFace repository name")
    args = parser.parse_args()
    
    if args.upload_only:
        upload_to_hf(MERGED_MODEL_PATH, args.repo)
    elif args.merge_only:
        merge_model()
    else:
        merge_model()
        upload_to_hf(MERGED_MODEL_PATH, args.repo)
