import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from typing import Tuple, Union
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# 配置 Huggingface Token（建议用环境变量）
os.environ["HUGGINGFACE_HUB_TOKEN"] = "REDACTED_HF_TOKEN"
output_prefix = "advbench_instruct_vs_base_diff"
# from huggingface_hub import hf_hub_download
# print(hf_hub_download(repo_id="NousResearch/Llama-2-7b-chat-hf", filename="config.json", cache_dir=None))


def compute_entropy(logits: torch.Tensor):
    """
    Calculate entropy of the prediction distribution.
    logits: [seq_len, vocab_size] tensor of logits for each token position.
    Returns: [seq_len] tensor of entropy per token.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def process_and_compare_series(series: pd.Series, model_name_1: str, model_name_2: str, max_length: int = 1024):
    """
    Process a series of texts with two models and compute the difference in token entropies.
    """
    # Use tokenizer from the first model for both
    tokenizer = AutoTokenizer.from_pretrained(model_name_1, token=os.environ["HUGGINGFACE_HUB_TOKEN"])

    print(f"Loading model 1: {model_name_1}")
    model_1 = AutoModelForCausalLM.from_pretrained(model_name_1, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    print(f"Loading model 2: {model_name_2}")
    model_2 = AutoModelForCausalLM.from_pretrained(model_name_2, token=os.environ["HUGGINGFACE_HUB_TOKEN"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_1.to(device)
    model_2.to(device)
    model_1.eval()
    model_2.eval()

    all_tokens = []
    all_entropy_diffs = []
    prompt_avg_entropy_diffs = []

    for text in tqdm(series, desc="Processing and comparing texts"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs_1 = model_1(**inputs)
            token_entropies_1 = compute_entropy(outputs_1.logits[0])

            outputs_2 = model_2(**inputs)
            token_entropies_2 = compute_entropy(outputs_2.logits[0])

        if token_entropies_1.shape[0] != token_entropies_2.shape[0]:
            print(f"Warning: Mismatch in sequence length for text: {text[:50]}...")
            min_len = min(token_entropies_1.shape[0], token_entropies_2.shape[0])
            token_entropies_1 = token_entropies_1[:min_len]
            token_entropies_2 = token_entropies_2[:min_len]
            current_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][:min_len])
        else:
            current_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Entropy difference: model_1 (instruct) - model_2 (base)
        entropy_diff = token_entropies_1 - token_entropies_2

        prompt_avg_entropy_diffs.append(entropy_diff.mean().item())

        all_tokens.extend(current_tokens)
        all_entropy_diffs.extend(entropy_diff.cpu().numpy().tolist())

    return {
        'all_tokens': np.array(all_tokens),
        'all_entropy_diffs': np.array(all_entropy_diffs),
        'prompt_avg_entropy_diffs': prompt_avg_entropy_diffs
    }

# ====== 数据读取与模型设定 ======
data = pd.read_csv('/projects/p32013/neurons/XLLM-main/Dataset/Advbench.csv')
model_name_1 = "meta-llama/Llama-3.2-3B-Instruct"
model_name_2 = "meta-llama/Llama-3.2-3B"


print(f"\nComparing models: \n1: {model_name_1} (Instruct)\n2: {model_name_2} (Base)")
print("\nAnalyzing token entropy differences for the first 100 prompts...")

stats = process_and_compare_series(data['target'].head(100), model_name_1, model_name_2)

mean_prompt_entropy_diff = np.mean(stats['prompt_avg_entropy_diffs'])
std_prompt_entropy_diff = np.std(stats['prompt_avg_entropy_diffs'])

print(f"\nOverall Statistics (Entropy Diff = Instruct - Base):")
print(f"Total number of prompts analyzed: {len(stats['prompt_avg_entropy_diffs'])}")
print(f"Total number of tokens analyzed: {len(stats['all_tokens'])}")
print(f"Mean of prompt-average-entropy-differences: {mean_prompt_entropy_diff:.4f}")
print(f"Standard deviation of prompt-average-entropy-differences: {std_prompt_entropy_diff:.4f}")


# Get top 100 tokens with largest and smallest entropy differences
df_token_entropy_diff = pd.DataFrame({
    'token': stats['all_tokens'],
    'entropy_diff': stats['all_entropy_diffs']
})
df_sorted = df_token_entropy_diff.sort_values(by='entropy_diff', ascending=True)

# Low diff: Instruct model is more certain than base model
low_diff_tokens = df_sorted.head(100)
# High diff: Instruct model is more uncertain than base model
high_diff_tokens = df_sorted.tail(100).sort_values(by='entropy_diff', ascending=False)


summary_path = f"results/{output_prefix}_summary.txt"
with open(summary_path, "w", encoding='utf-8') as f:
    f.write(f"Comparison between {model_name_1} and {model_name_2}\n")
    f.write("Overall Statistics (Entropy Diff = Instruct - Base):\n")
    f.write(f"Total number of prompts analyzed: {len(stats['prompt_avg_entropy_diffs'])}\n")
    f.write(f"Total number of tokens analyzed: {len(stats['all_tokens'])}\n")
    f.write(f"Mean of prompt-average-entropy-differences: {mean_prompt_entropy_diff:.4f}\n")
    f.write(f"Standard deviation of prompt-average-entropy-differences: {std_prompt_entropy_diff:.4f}\n")
    f.write(f"List of prompt-average-entropy-differences:\n{stats['prompt_avg_entropy_diffs']}\n")
    f.write("\n\n--- Top 100 Tokens with Largest Entropy Difference (Instruct more uncertain) ---\n")
    f.write(high_diff_tokens.to_string())
    f.write("\n\n--- Top 100 Tokens with Smallest Entropy Difference (Instruct more certain) ---\n")
    f.write(low_diff_tokens.to_string())