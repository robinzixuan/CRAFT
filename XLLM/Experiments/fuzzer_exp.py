import sys
import os
from dotenv import load_dotenv
load_dotenv()
openai_key = os.getenv('OPENAI_API_KEY')

# Add the path to the BOOST folder to sys.path
sys.path.append(os.path.abspath('../BOOST'))
import argparse
import random
import numpy as np
import torch
from BOOST.Attack_GPTFuzzer.gptfuzz import fuzzer_attack
from fastchat.model import add_model_args
from BOOST.utils.constants import claude_key, gemini_key
from BOOST.utils.templates import get_eos


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--index', type=int, default=10, help='The index of the question')
    parser.add_argument('--model_path', type=str, default='gpt-4-1106-preview',
                        help='mutate model path')
    parser.add_argument('--target_model', type=str, default='google/gemma-7b-it',
                        help='The target model, openai model or open-sourced LLMs')
    parser.add_argument('--max_query', type=int, default=10,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=1, help='The maximum jailbreak number')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument('--seed_selection_strategy', type=str,
                        default='round_robin', help='The seed selection strategy')
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed_path", type=str,
                        default="./Dataset/fuzzer_seed.csv", help="The seed path")
    parser.add_argument("--add_eos", action='store_true')
    parser.add_argument("--eos_num", type=int, default=10, help="The number of eos tokens")
    parser.add_argument("--run_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--harmful_dataset', type=str, default='Dataset/harmful.csv',
                        help='Path to the harmful questions dataset')
    parser.add_argument('--targets_dataset', type=str, default='Dataset/harmful_targets.csv',
                        help='Path to the harmful targets dataset')
    parser.add_argument('--evaluation', type=str, default='default', choices=['default', 'strongreject'], help='Evaluation method for attack success: "default" (original) or "strongreject" (use strongreject autograder)')
    add_model_args(parser)

    args = parser.parse_args()
    set_random_seed(args.seed)  # Set the random seed
    args.openai_key = openai_key
    args.claude_key = claude_key
    args.gemini_key = gemini_key
    fuzzer_attack(args)