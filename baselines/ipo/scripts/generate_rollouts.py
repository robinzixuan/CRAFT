"""
Step 2: Generate reasoning rollouts using an LRM.
For each harmful prompt, the model generates its reasoning process (including <think> tags).
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
        help="LRM model name"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/star1.json",
        help="Path to the STAR-1 dataset file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/raw_rollouts.json",
        help="Output file path"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="Number of rollouts per prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    return parser.parse_args()


def load_model(model_name: str):
    """Load model and tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading model: {model_name}")
    
    # 4-bit quantization configuration
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
    """Generate reasoning rollouts."""
    import torch
    from tqdm import tqdm
    import re
    
    all_samples = []
    total_tasks = len(prompts) * num_rollouts
    
    # Outer progress bar: prompt processing progress
    pbar = tqdm(
        prompts,
        desc="Generating reasoning rollouts",
        unit="prompt",
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
        
        # Inner progress bar: rollout progress per prompt (only shown when num_rollouts > 1)
        rollout_range = range(num_rollouts)
        if num_rollouts > 1:
            rollout_range = tqdm(
                rollout_range,
                desc=f"  Prompt {prompt_idx+1}/{len(prompts)}",
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
            
            # Remove input portion
            if input_text in generated:
                generated = generated.replace(input_text, "")
            
            # Parse reasoning and response
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
        
        # Update outer progress bar
        pbar.set_postfix({
            "samples_generated": len(all_samples),
            "success_rate": f"{len(all_samples)/((prompt_idx+1)*num_rollouts)*100:.1f}%"
        })

    return all_samples


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Step 2: Generate Reasoning Rollouts")
    print("=" * 60)
    
    # Load STAR-1 dataset
    prompts_path = Path(__file__).parent.parent / args.prompts_file
    with open(prompts_path, "r", encoding="utf-8") as f:
        star1_data = json.load(f)
    
    # Extract the 'question' field from STAR-1 data as prompts
    prompts = [item["question"] for item in star1_data if "question" in item]
    print(f"Loaded {len(prompts)} prompts from STAR-1 dataset")
    
    # Load model
    model, tokenizer = load_model(args.model_name)
    
    # Generate rollouts
    samples = generate_rollouts(
        model,
        tokenizer,
        prompts,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    
    # Save
    output_path = Path(__file__).parent.parent / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(samples)} rollout samples")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
