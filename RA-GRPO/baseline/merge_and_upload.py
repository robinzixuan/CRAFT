"""
合并 LoRA 适配器到基础模型，并上传到 HuggingFace Hub
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import HfApi, login

# 配置
BASE_MODEL_PATH = "99sweetcookie/reasoningshield-stage1-sft"
LORA_PATH = "99sweetcookie/reasoningshield-stage2-dpo"
MERGED_MODEL_PATH = "./merged_reasoningshield_dpo"  # 本地保存路径
HF_REPO_NAME = "YOUR_USERNAME/reasoningshield-stage2-dpo-merged"  # 修改为你的 HF 用户名

def merge_model():
    """合并 LoRA 到基础模型"""
    print(f"正在加载基础模型: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"正在加载 LoRA 适配器: {LORA_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    print("正在合并模型...")
    merged_model = model.merge_and_unload()
    
    print(f"正在保存合并后的模型到: {MERGED_MODEL_PATH}")
    merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    print("模型合并完成!")
    return MERGED_MODEL_PATH

def upload_to_hf(model_path, repo_name):
    """上传模型到 HuggingFace Hub"""
    # 检查是否登录
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("请先设置 HF_TOKEN 环境变量或运行 huggingface-cli login")
        return
    
    api = HfApi()
    
    print(f"正在上传模型到: {repo_name}")
    api.create_repo(repo_name, exist_ok=True)
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        commit_message="Upload merged reasoningshield-stage2-dpo model"
    )
    print(f"上传完成! 模型地址: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-only", action="store_true", help="只合并模型，不上传")
    parser.add_argument("--upload-only", action="store_true", help="只上传已合并的模型")
    parser.add_argument("--repo", type=str, default=HF_REPO_NAME, help="HuggingFace 仓库名")
    args = parser.parse_args()
    
    if args.upload_only:
        upload_to_hf(MERGED_MODEL_PATH, args.repo)
    elif args.merge_only:
        merge_model()
    else:
        merge_model()
        upload_to_hf(MERGED_MODEL_PATH, args.repo)
