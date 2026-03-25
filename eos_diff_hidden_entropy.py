import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
os.environ["TRANSFORMERS_CACHE"] = "/projects/p32013/neurons/tmp/hf_cache"
os.environ["HF_HOME"] = "/projects/p32013/neurons/tmp/hf_cache"
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["HUGGINGFACE_HUB_TOKEN"] = "REDACTED_HF_TOKEN"
output_prefix = "advbench_eosdiff"

def compute_tokenwise_hidden_entropy(hidden_state: torch.Tensor):
    """
    hidden_state: [seq_len, hidden_dim]
    Returns: [seq_len] tensor of entropy per token.
    """
    probs = F.softmax(hidden_state, dim=-1)
    log_probs = F.log_softmax(hidden_state, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy  # [seq_len]

def process_entropy_diff_with_eos(data: pd.Series, model_name: str, max_length: int = 1024):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    model.eval()

    eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")
    if eos_token_id is None:
        raise ValueError("Model/tokenizer does not define eos_token_id.")

    records = []

    for text in tqdm(data, desc="Processing entropy per token vs EOS"):
        # ✅ 强制在文本末尾添加 <eos>
        text_with_eos = text + tokenizer.eos_token
        inputs = tokenizer(text_with_eos, return_tensors="pt", truncation=True, max_length=max_length)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
        input_ids = inputs.input_ids[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # ✅ 获取 <eos> 的位置（找不到就 fallback 为最后一个）
        eos_indices = (input_ids == eos_token_id).nonzero(as_tuple=True)
        if len(eos_indices[0]) == 0:
            eos_idx = len(input_ids) - 1
        else:
            eos_idx = eos_indices[0].item()

        eos_hidden = hidden_states[eos_idx]
        eos_entropy = compute_tokenwise_hidden_entropy(eos_hidden.unsqueeze(0)).item()

        token_entropies = compute_tokenwise_hidden_entropy(hidden_states)  # [seq_len]
        entropy_diffs = token_entropies - eos_entropy

        for i in range(len(tokens)):
            records.append({
                "token": tokens[i],
                "token_entropy": token_entropies[i].item(),
                "eos_entropy": eos_entropy,
                "entropy_diff": entropy_diffs[i].item()
            })

    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip()  # 防止 invisible character
    raw_csv_path = f"{output_prefix}_tokenwise_entropy_diff.csv"
    df.to_csv(raw_csv_path, index=False, encoding='utf-8')
    print(f"✓ Saved raw token-wise entropy to: {raw_csv_path}")

    df_token_avg = df.groupby("token").agg({
        "token_entropy": "mean",
        "eos_entropy": "mean",
        "entropy_diff": "mean"
    }).reset_index()
    df_token_avg["count"] = df.groupby("token").size().values

    df_token_avg = df_token_avg.sort_values(by="entropy_diff", ascending=False)
    avg_csv_path = f"{output_prefix}_tokenwise_entropy_diff_avg.csv"
    df_token_avg.to_csv(avg_csv_path, index=False, encoding='utf-8')
    print(f"✓ Saved averaged token entropy to: {avg_csv_path}")

    return raw_csv_path, avg_csv_path


def extract_top_bottom_tokens(csv_path: str, output_prefix: str, frac: float = 0.2):
    df = pd.read_csv(csv_path)
    sorted_df = df.sort_values(by="entropy_diff", ascending=False)
    k = int(len(sorted_df) * frac)

    top_df = sorted_df.head(k)
    bottom_df = sorted_df.tail(k)

    top_path = f"{output_prefix}_top_{int(frac * 100)}pct.txt"
    bottom_path = f"{output_prefix}_bottom_{int(frac * 100)}pct.txt"

    top_df["token"].to_csv(top_path, index=False, header=False)
    bottom_df["token"].to_csv(bottom_path, index=False, header=False)

    print(f"✓ Top {frac*100:.0f}% tokens → {top_path}")
    print(f"✓ Bottom {frac*100:.0f}% tokens → {bottom_path}")


if __name__ == "__main__":
    data = pd.read_csv("/projects/p32013/XLLM/Dataset/Advbench.csv")
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    print(f"\n🚀 Loading model: {model_name}")
    print("🔍 Analyzing hidden entropy vs. EOS...")

    raw_csv, avg_csv = process_entropy_diff_with_eos(data["target"], model_name)

    extract_top_bottom_tokens(raw_csv, output_prefix="advbench_rawtokens")
    extract_top_bottom_tokens(avg_csv, output_prefix="advbench_avgtokens")
