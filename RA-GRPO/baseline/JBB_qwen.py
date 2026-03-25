"""
使用 JailbreakBench 数据集对模型进行越狱测试评估
支持本地 checkpoint 或 HuggingFace 模型 ID
使用 transformers 加载模型（避免 vLLM 兼容性问题）
"""

import os
import json
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import jailbreakbench as jbb
from jailbreakbench.classifier import Llama3JailbreakJudge, Llama3RefusalJudge, StringClassifier
from jailbreakbench.config import TEMPERATURE, TOP_P, MAX_GENERATION_LENGTH

# 模型路径（可以是本地路径或 HuggingFace 模型 ID）
MODEL_PATH = "/scratch/hlv8980/llama_think_safety_3/global_step_13/actor/huggingface"  # 可以是本地路径或 HF 模型 ID

# API key (用于 judge 评估)
API_KEY = os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY")

# 检测是否为 HuggingFace 模型 ID（包含 '/' 且不是本地路径）
is_hf_model_id = "/" in MODEL_PATH and not os.path.exists(MODEL_PATH)

# 简单校验：如果是本地路径，检查文件是否存在
if not is_hf_model_id:
    if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        raise RuntimeError(f"未找到 config.json，请确认 HF 格式目录: {MODEL_PATH}")
    weight_files = []
    for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.safetensors"):
        weight_files.extend(glob.glob(os.path.join(MODEL_PATH, pattern)))
    if not weight_files:
        raise RuntimeError(f"未找到 HF 权重文件，请将 checkpoint 导出为 HF 格式: {MODEL_PATH}")
else:
    print(f"检测到 HuggingFace 模型 ID: {MODEL_PATH}")
    print("将自动从 HuggingFace Hub 下载模型文件")

# 使用 transformers 加载本地模型
print(f"正在加载模型: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# 加载 chat_template（如果 tokenizer 没有）
if tokenizer.chat_template is None and not is_hf_model_id:
    template_path = os.path.join(MODEL_PATH, "chat_template.jinja")
    if os.path.isfile(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()

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

# 先保存模型响应（即使评估失败也能看到结果）
print("\n=== 保存模型响应 ===")
responses_output_path = "baseline/response/jbb_reasoningshield_qwen_responses.json"
with open(responses_output_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_path": MODEL_PATH,
        "total": len(goals),
        "data": [{"behavior": b, "goal": g, "response": r} for b, g, r in zip(behaviors, goals, responses)]
    }, f, ensure_ascii=False, indent=2)
print(f"模型响应已保存到: {responses_output_path}")

# # 评估结果
# print("\n=== 使用 Llama3 Judge 评估 ===")
# if not API_KEY:
#     print("警告: 未设置 TOGETHER_API_KEY 或 OPENAI_API_KEY，跳过 Judge 评估")
#     is_jailbroken = [None] * len(goals)
#     is_refusal = [None] * len(goals)
#     num_jailbroken = 0
#     num_refusal = 0
#     asr = 0.0
#     refusal_rate = 0.0
# else:
#     try:
#         jailbreak_judge = Llama3JailbreakJudge(API_KEY)
#         refusal_judge = Llama3RefusalJudge(API_KEY)

#         # 判断是否越狱成功
#         is_jailbroken = jailbreak_judge(goals, responses)
#         is_refusal = refusal_judge(goals, responses)

#         # 计算指标
#         num_jailbroken = sum(is_jailbroken)
#         num_refusal = sum(is_refusal)
#     except Exception as e:
#         print(f"评估失败: {e}")
#         print("模型响应已保存，可以手动评估或检查 API key")
#         is_jailbroken = [None] * len(goals)
#         is_refusal = [None] * len(goals)
#         num_jailbroken = 0
#         num_refusal = 0

# total = len(goals)
# asr = num_jailbroken / total if num_jailbroken else 0.0
# refusal_rate = num_refusal / total if num_refusal else 0.0

# print(f"总样本数: {total}")
# print(f"越狱成功数: {num_jailbroken}")
# print(f"拒绝回答数: {num_refusal}")
# print(f"攻击成功率 (ASR): {asr:.2%}")
# print(f"拒绝率: {refusal_rate:.2%}")

# # 打印部分结果示例
# print("\n=== 部分结果示例 ===")
# for i in range(min(5, len(goals))):
#     print(f"\n[{i}] Behavior: {behaviors[i]}")
#     print(f"Goal: {goals[i][:100]}...")
#     print(f"Response: {responses[i][:200]}...")
#     print(f"Jailbroken: {is_jailbroken[i]}, Refusal: {is_refusal[i]}")

# # 保存详细结果
# results = []
# for i in range(len(goals)):
#     results.append({
#         "behavior": behaviors[i],
#         "goal": goals[i],
#         "response": responses[i],
#         "jailbroken": is_jailbroken[i],
#         "refusal": is_refusal[i]
#     })

# output_path = "baseline/jbb_qwen_eval_results.json"
# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump({
#         "model_path": MODEL_PATH,
#         "asr": asr,
#         "refusal_rate": refusal_rate,
#         "total": total,
#         "num_jailbroken": num_jailbroken,
#         "num_refusal": num_refusal,
#         "results": results
#     }, f, ensure_ascii=False, indent=2)

# print(f"\n详细结果已保存到: {output_path}")
