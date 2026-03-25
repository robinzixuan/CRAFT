"""
使用 JailbreakBench 数据集对模型进行越狱测试评估
使用 transformers 加载模型
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import jailbreakbench as jbb
from jailbreakbench.config import TEMPERATURE, TOP_P, MAX_GENERATION_LENGTH

# 模型名称
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"

# 使用 transformers 加载模型
print(f"正在加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
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

# 查询模型
print(f"开始查询 {len(goals)} 个 prompts...")
responses = query_model(goals)
print("查询完成!")

# 保存模型响应
print("\n=== 保存模型响应 ===")
responses_output_path = "baseline/response/jbb_qwen_base_responses.json"
with open(responses_output_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "total": len(goals),
        "data": [{"behavior": b, "goal": g, "response": r} for b, g, r in zip(behaviors, goals, responses)]
    }, f, ensure_ascii=False, indent=2)
print(f"模型响应已保存到: {responses_output_path}")
