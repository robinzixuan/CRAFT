"""
构建第二阶段训练数据（防止过度拒绝）

论文Section 4.1:
"We notice that models trained on these safety-only datasets are inclined to over-refuse.
To address this, we additionally use the 915 benign prompts from STAR-1 to construct
a supplementary preference dataset, contrasting normal responses from base LRMs with
refusal responses from trained LRMs, and apply DPO training as an additional stage."

第二阶段数据格式:
- chosen: base LRM对良性提示的正常回复
- rejected: IPO训练后模型对良性提示的（错误的）拒绝回复
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage 2 Data (Anti Over-Refusal)")
    
    parser.add_argument(
        "--benign_prompts",
        type=str,
        default="data/star1_benign_prompts.json",
        help="良性提示文件路径"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Base LRM模型名称"
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        default="./ipo_stage1_output",
        help="第一阶段训练后的模型路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stage2_train.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="最大生成token数"
    )
    
    return parser.parse_args()


def load_model(model_path: str):
    """加载模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading model: {model_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    """生成回复"""
    import torch
    import re
    
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 移除输入部分
    if input_text in generated:
        generated = generated.replace(input_text, "")
    
    return generated.strip()


def is_refusal(response: str) -> bool:
    """检测是否是拒绝回复"""
    import re
    
    refusal_patterns = [
        r"i cannot",
        r"i can't",
        r"i won't",
        r"i'm unable to",
        r"i'm sorry.*but",
        r"i apologize.*but",
        r"not something i can",
        r"cannot assist",
        r"can't help with",
        r"i must decline",
        r"i need to refuse",
        r"is harmful",
        r"is dangerous",
        r"is illegal",
    ]
    
    response_lower = response.lower()
    for pattern in refusal_patterns:
        if re.search(pattern, response_lower):
            return True
    
    return False


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Building Stage 2 Data (Anti Over-Refusal)")
    print("=" * 60)
    print()
    print("论文说明:")
    print("  模型在safety-only数据集上训练后容易过度拒绝")
    print("  使用915个良性提示构建补充偏好数据集:")
    print("  - chosen: base LRM的正常回复")
    print("  - rejected: 训练后模型的拒绝回复")
    print()
    
    # 加载良性提示
    prompts_path = Path(__file__).parent.parent / args.benign_prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        benign_data = json.load(f)
    
    prompts = [item["prompt"] if isinstance(item, dict) else item for item in benign_data]
    print(f"Loaded {len(prompts)} benign prompts")
    
    # 加载模型
    print("\n加载Base模型...")
    base_model, base_tokenizer = load_model(args.base_model)
    
    print("\n加载训练后模型...")
    trained_model, trained_tokenizer = load_model(args.trained_model)
    
    # 生成回复并构建数据
    from tqdm import tqdm
    
    stage2_data = []
    refusal_count = 0
    
    for prompt in tqdm(prompts, desc="Generating responses"):
        # Base模型的正常回复
        base_response = generate_response(
            base_model, base_tokenizer, prompt, args.max_new_tokens
        )
        
        # 训练后模型的回复
        trained_response = generate_response(
            trained_model, trained_tokenizer, prompt, args.max_new_tokens
        )
        
        # 检查训练后模型是否错误地拒绝了
        if is_refusal(trained_response):
            refusal_count += 1
            stage2_data.append({
                "prompt": prompt,
                "chosen": base_response,  # 正常回复是preferred
                "rejected": trained_response,  # 错误的拒绝是rejected
            })
    
    print(f"\n检测到 {refusal_count}/{len(prompts)} 个过度拒绝样本")
    print(f"构建了 {len(stage2_data)} 个第二阶段训练样本")
    
    # 保存
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stage2_data, f, indent=2, ensure_ascii=False)
    
    print(f"保存到 {output_path}")


if __name__ == "__main__":
    main()
