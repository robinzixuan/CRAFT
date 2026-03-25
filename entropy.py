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
output_prefix = "advbench"
# from huggingface_hub import hf_hub_download
# print(hf_hub_download(repo_id="NousResearch/Llama-2-7b-chat-hf", filename="config.json", cache_dir=None))


def compute_entropy(logits: torch.Tensor):
    """
    Calculate entropy of the prediction distribution.
    """
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item() if is_single_input else entropy


def process_text_series(series: pd.Series, model_name: str, max_length: int = 1024):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.environ["HUGGINGFACE_HUB_TOKEN"])

    all_tokens = []
    all_entropies = []

    for text in tqdm(series, desc="Processing texts"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        outputs = model(**inputs)
        token_entropies = compute_entropy(outputs.logits)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        all_tokens.extend(tokens)
        all_entropies.extend(token_entropies[0].detach().cpu().tolist())

    all_tokens = np.array(all_tokens)
    all_entropies = np.array(all_entropies)

    k = int(len(all_tokens) * 0.20)
    top_indices = np.argsort(all_entropies)[-k:]
    bottom_indices = np.argsort(all_entropies)[:k]

    top_tokens = all_tokens[top_indices]
    bottom_tokens = all_tokens[bottom_indices]

    top_token_freq = {}
    bottom_token_freq = {}

    for token in top_tokens:
        top_token_freq[token] = top_token_freq.get(token, 0) + 1
    for token in bottom_tokens:
        bottom_token_freq[token] = bottom_token_freq.get(token, 0) + 1

    top_token_freq = dict(sorted(top_token_freq.items(), key=lambda x: x[1], reverse=True))
    bottom_token_freq = dict(sorted(bottom_token_freq.items(), key=lambda x: x[1], reverse=True))

    print("\nTop 100 High Entropy Tokens and their frequencies:")
    for i, (token, freq) in enumerate(list(top_token_freq.items())[:100]):
        print(f"{i+1}. Token: '{token}', Frequency: {freq}")

    print("\nTop 100 Low Entropy Tokens and their frequencies:")
    for i, (token, freq) in enumerate(list(bottom_token_freq.items())[:100]):
        print(f"{i+1}. Token: '{token}', Frequency: {freq}")

    with open(f"{output_prefix}_top_tokens.txt", "w", encoding='utf-8') as f_top, \
         open(f"{output_prefix}_bottom_tokens.txt", "w", encoding='utf-8') as f_bottom:

        f_top.write("Top 100 High Entropy Tokens and their Frequencies:\n")
        for i, (token, freq) in enumerate(list(top_token_freq.items())[:100]):
            f_top.write(f"{i+1}. Token: '{token}', Frequency: {freq}\n")

        f_bottom.write("Top 100 Low Entropy Tokens and their Frequencies:\n")
        for i, (token, freq) in enumerate(list(bottom_token_freq.items())[:100]):
            f_bottom.write(f"{i+1}. Token: '{token}', Frequency: {freq}\n")

    return {
        'top_token_freq': top_token_freq,
        'bottom_token_freq': bottom_token_freq,
        'all_entropies': all_entropies,
        'all_tokens': all_tokens
    }

# ====== 数据读取与模型设定 ======
data = pd.read_csv('/projects/p32013/XLLM/Dataset/Advbench.csv')
model_name = "meta-llama/Llama-3.2-3B-Instruct"

print(f"\nLogin successful.\nLoading model: {model_name}...")
print("\nAnalyzing token entropies across all data...")

stats = process_text_series(data['goal'], model_name)

mean_entropy = np.mean(stats['all_entropies'])
std_entropy = np.std(stats['all_entropies'])

print(f"\nOverall Statistics:")
print(f"Total number of tokens analyzed: {len(stats['all_tokens'])}")
print(f"Mean entropy: {mean_entropy:.4f}")
print(f"Standard deviation of entropy: {std_entropy:.4f}")
print(f"Unique high entropy tokens: {len(stats['top_token_freq'])}")
print(f"Unique low entropy tokens: {len(stats['bottom_token_freq'])}")

summary_path = f"{output_prefix}_entropy_summary.txt"
with open(summary_path, "w", encoding='utf-8') as f:
    f.write("Overall Statistics:\n")
    f.write(f"Total number of tokens analyzed: {len(stats['all_tokens'])}\n")
    f.write(f"Mean entropy: {mean_entropy:.4f}\n")
    f.write(f"Standard deviation of entropy: {std_entropy:.4f}\n")
    f.write(f"Unique high entropy tokens: {len(stats['top_token_freq'])}\n")
    f.write(f"Unique low entropy tokens: {len(stats['bottom_token_freq'])}\n")

# ====== 可视化部分：保存 entropy 分布图 ======
plt.figure(figsize=(10, 6))
plt.hist(stats['all_entropies'], bins=100, color='skyblue', edgecolor='black')
plt.title("Token Entropy Distribution")
plt.xlabel("Entropy")
plt.ylabel("Token Count")
plt.grid(True)

plot_path = f"{output_prefix}_entropy_histogram.png"
plt.savefig(plot_path)
plt.close()

print(f"\nHistogram saved to: {plot_path}")
print(f"\nResults saved to: {summary_path}, top and bottom token files.")
