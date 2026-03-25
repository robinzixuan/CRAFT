"""
Step 2: 使用LRM生成推理轨迹
对每个有害提示，让模型生成推理过程（包含<think>标签）
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning rollouts from LRM")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="LRM模型名称"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/star1.json",
        help="STAR-1数据集文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/raw_rollouts.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="每个提示的生成数量"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="生成温度"
    )
    return parser.parse_args()


def load_model(model_name: str):
    """加载模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading model: {model_name}")
    
    # 4bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_rollouts(
    model,
    tokenizer,
    prompts: list,
    num_rollouts: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
):
    """生成推理轨迹"""
    import torch
    from tqdm import tqdm
    import re
    
    all_samples = []
    total_tasks = len(prompts) * num_rollouts
    
    # 外层进度条：显示提示处理进度
    pbar = tqdm(
        prompts,
        desc="生成推理轨迹",
        unit="提示",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for prompt_idx, prompt in enumerate(pbar):
        # 构建输入
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # 内层进度条：显示每个提示的rollout生成进度（仅在num_rollouts > 1时显示）
        rollout_range = range(num_rollouts)
        if num_rollouts > 1:
            rollout_range = tqdm(
                rollout_range,
                desc=f"  提示 {prompt_idx+1}/{len(prompts)}",
                leave=False,
                unit="rollout"
            )
        
        for rollout_idx in rollout_range:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 移除输入部分
            if input_text in generated:
                generated = generated.replace(input_text, "")
            
            # 解析推理和回复
            think_match = re.search(r'<think>(.*?)</think>', generated, re.DOTALL)
            
            if think_match:
                reasoning = think_match.group(1).strip()
                response = generated[think_match.end():].strip()
            else:
                reasoning = ""
                response = generated.strip()
            
            if reasoning:  # 只保存有推理的样本
                all_samples.append({
                    "prompt": prompt,
                    "reasoning": reasoning,
                    "response": response,
                })
        
        # 更新外层进度条信息
        pbar.set_postfix({
            "已生成样本": len(all_samples),
            "成功率": f"{len(all_samples)/((prompt_idx+1)*num_rollouts)*100:.1f}%"
        })

    return all_samples


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Step 2: Generate Reasoning Rollouts")
    print("=" * 60)
    
    # 加载STAR-1数据集
    prompts_path = Path(__file__).parent.parent / args.prompts_file
    with open(prompts_path, "r", encoding="utf-8") as f:
        star1_data = json.load(f)
    
    # 从STAR-1数据中提取question字段作为提示
    prompts = [item["question"] for item in star1_data if "question" in item]
    print(f"Loaded {len(prompts)} prompts from STAR-1 dataset")
    
    # 加载模型
    model, tokenizer = load_model(args.model_name)
    
    # 生成rollouts
    samples = generate_rollouts(
        model,
        tokenizer,
        prompts,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # 保存
    output_path = Path(__file__).parent.parent / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(samples)} rollout samples")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
