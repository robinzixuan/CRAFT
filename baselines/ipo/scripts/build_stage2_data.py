"""
Build Stage 2 training data (anti over-refusal).

Paper Section 4.1:
"We notice that models trained on these safety-only datasets are inclined to over-refuse.
To address this, we additionally use the 915 benign prompts from STAR-1 to construct
a supplementary preference dataset, contrasting normal responses from base LRMs with
refusal responses from trained LRMs, and apply DPO training as an additional stage."

Stage 2 data format:
- chosen: normal response from base LRM to a benign prompt
- rejected: (erroneous) refusal response from the IPO-trained model to the same benign prompt
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
        help="Path to benign prompts file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Base LRM model name"
    )
    parser.add_argument(
        "--trained_model",
        type=str,
        default="./ipo_stage1_output",
        help="Path to the Stage 1 trained model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stage2_train.json",
        help="Output file path"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate"
    )
    
    return parser.parse_args()


def load_model(model_path: str):
    """Load model and tokenizer."""
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
    """Generate a response for the given prompt."""
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
    
    # Remove input portion
    if input_text in generated:
        generated = generated.replace(input_text, "")
    
    return generated.strip()


def is_refusal(response: str) -> bool:
    """Detect whether the response is a refusal."""
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
    print("Paper note:")
    print("  Models trained on safety-only datasets tend to over-refuse.")
    print("  We build a supplementary preference dataset from 915 benign prompts:")
    print("  - chosen: normal response from base LRM")
    print("  - rejected: refusal response from the trained model")
    print()
    
    # Load benign prompts
    prompts_path = Path(__file__).parent.parent / args.benign_prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        benign_data = json.load(f)
    
    prompts = [item["prompt"] if isinstance(item, dict) else item for item in benign_data]
    print(f"Loaded {len(prompts)} benign prompts")
    
    # Load models
    print("\nLoading base model...")
    base_model, base_tokenizer = load_model(args.base_model)

    print("\nLoading trained model...")
    trained_model, trained_tokenizer = load_model(args.trained_model)

    # Generate responses and build dataset
    from tqdm import tqdm
    
    stage2_data = []
    refusal_count = 0
    
    for prompt in tqdm(prompts, desc="Generating responses"):
        # Normal response from base model
        base_response = generate_response(
            base_model, base_tokenizer, prompt, args.max_new_tokens
        )
        
        # Response from trained model
        trained_response = generate_response(
            trained_model, trained_tokenizer, prompt, args.max_new_tokens
        )
        
        # Check whether the trained model incorrectly refuses
        if is_refusal(trained_response):
            refusal_count += 1
            stage2_data.append({
                "prompt": prompt,
                "chosen": base_response,  # normal response is preferred
                "rejected": trained_response,  # erroneous refusal is rejected
            })
    
    print(f"\nDetected {refusal_count}/{len(prompts)} over-refusal samples")
    print(f"Built {len(stage2_data)} Stage 2 training samples")
    
    # 保存
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stage2_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
