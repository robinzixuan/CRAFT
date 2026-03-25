import sys
import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

# Add the path to the BOOST folder to sys.path
sys.path.append(os.path.abspath('../BOOST'))
from BOOST.Attack_GCG.run_gcg import gcg_attack
import argparse
import random
import numpy as np
import torch


def set_random_seed(seed=42):
    """Set random seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_openai_key():
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=10,
        )
        print("API key works! Response:", response.choices[0].message.content)
    except Exception as e:
        print("API key test failed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCG attack on harmful dataset')
    parser.add_argument('--index', type=int, default=0, help='The index of the question')
    parser.add_argument('--model_path', type=str, default='allenai/tulu-2-dpo-7b',
                        help='target model path')
    parser.add_argument("--control_string_length", type=int, default=20)
    parser.add_argument("--max_attack_steps", type=int, default=500)
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_attack_attempts", type=int, default=1)
    parser.add_argument("--max_prompts_in_single_attack", type=int, default=1)
    parser.add_argument("--max_successful_prompt", type=int, default=1)
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument("--eos_num", type=int, default=10)
    parser.add_argument("--run_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--evaluation', type=str, default='default', choices=['default', 'strongreject'],
                        help='Evaluation method for attack success: "default" (original) or "strongreject" (use strongreject autograder)')
    parser.add_argument('--harmful_dataset', type=str, default='Dataset/harmful.csv',
                        help='Path to the harmful questions dataset')
    parser.add_argument('--targets_dataset', type=str, default='Dataset/harmful_targets.csv',
                        help='Path to the harmful targets dataset')

    args = parser.parse_args()
    set_random_seed(args.seed)  # Set the random seed
    test_openai_key()
    gcg_attack(args)
