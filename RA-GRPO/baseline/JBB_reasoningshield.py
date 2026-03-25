"""
使用 JailbreakBench 数据集对 reasoningshield 模型进行越狱测试评估
该模型是基于 99sweetcookie/reasoningshield-stage1-sft 的 DPO 微调版本
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from tqdm import tqdm
import jailbreakbench as jbb
from jailbreakbench.config import TEMPERATURE, TOP_P

# 基础模型和 LoRA 适配器路径
BASE_MODEL_PATH = "99sweetcookie/reasoningshield-stage1-sft"
LORA_PATH = "99sweetcookie/reasoningshield-stage2-dpo"

# API key (用于 judge 评估)
API_KEY = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")

# 加载基础模型
print(f"正在加载基础模型: {BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载 LoRA 适配器
print(f"正在加载 LoRA 适配器: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# 加载 JailbreakBench 数据集
dataset = jbb.read_dataset()
goals = dataset.goals
behaviors = dataset.behaviors

def query_model(prompts: list[str], batch_size: int = 4) -> list[str]:
    """使用 transformers 查询模型"""
    responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="生成中"):
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

# 批量查询模型
print(f"开始查询 {len(goals)} 个 prompts...")
responses = query_model(goals)
print("查询完成!")

# 保存模型响应
print("\n=== 保存模型响应 ===")
responses_output_path = "baseline/response/jbb_reasoningshield_llama_responses.json"
with open(responses_output_path, "w", encoding="utf-8") as f:
    json.dump({
        "base_model": BASE_MODEL_PATH,
        "lora_adapter": LORA_PATH,
        "total": len(goals),
        "data": [{"behavior": b, "goal": g, "response": r} for b, g, r in zip(behaviors, goals, responses)]
    }, f, ensure_ascii=False, indent=2)
print(f"模型响应已保存到: {responses_output_path}")
