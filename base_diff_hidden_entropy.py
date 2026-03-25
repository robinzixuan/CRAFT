import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TRANSFORMERS_CACHE"] = "/projects/p32013/.cache"
os.environ["HF_HOME"] = "/projects/p32013/.cache"
os.environ["HUGGINGFACE_HUB_TOKEN"] = "REDACTED_HF_TOKEN"

output_prefix = "advbench_instruct_vs_base"

def compute_tokenwise_hidden_entropy(hidden_state: torch.Tensor):
    probs = F.softmax(hidden_state, dim=-1)
    log_probs = F.log_softmax(hidden_state, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy  # [seq_len]

def process_entropy_diff_instruct_vs_base(data: pd.Series, model_instruct: str, model_base: str, max_length: int = 1024):
    tokenizer = AutoTokenizer.from_pretrained(model_instruct, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    model1 = AutoModelForCausalLM.from_pretrained(model_instruct, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    model2 = AutoModelForCausalLM.from_pretrained(model_base, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    model1.eval()
    model2.eval()

    records = []

    for text in tqdm(data, desc="Comparing hidden entropy: instruct vs base"):
        text_with_eos = text + tokenizer.eos_token
        inputs = tokenizer(text_with_eos, return_tensors="pt", truncation=True, max_length=max_length)

        with torch.no_grad():
            h1 = model1(**inputs, output_hidden_states=True).hidden_states[-1][0]  # [seq_len, hidden_dim]
            h2 = model2(**inputs, output_hidden_states=True).hidden_states[-1][0]  # [seq_len, hidden_dim]

        input_ids = inputs.input_ids[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        ent1 = compute_tokenwise_hidden_entropy(h1)  # instruct
        ent2 = compute_tokenwise_hidden_entropy(h2)  # base
        ent_diff = ent1 - ent2

        for i in range(len(tokens)):
            records.append({
                "token": tokens[i],
                "entropy_instruct": ent1[i].item(),
                "entropy_base": ent2[i].item(),
                "entropy_diff": ent_diff[i].item()
            })

    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip()
    raw_csv_path = f"{output_prefix}_tokenwise_entropy_diff.csv"
    df.to_csv(raw_csv_path, index=False, encoding='utf-8')
    print(f"✓ Saved raw token-wise entropy to: {raw_csv_path}")

    df_token_avg = df.groupby("token").agg({
        "entropy_instruct": "mean",
        "entropy_base": "mean",
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
    df.columns = df.columns.str.strip()
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

    model_name_instruct = "meta-llama/Llama-3.2-3B-Instruct"
    model_name_base     = "meta-llama/Llama-3.2-3B"

    print(f"\nComparing: {model_name_instruct} vs {model_name_base}")
    raw_csv, avg_csv = process_entropy_diff_instruct_vs_base(data["goal"], model_name_instruct, model_name_base)

    extract_top_bottom_tokens(raw_csv, output_prefix="advbench_rawtokens")
    extract_top_bottom_tokens(avg_csv, output_prefix="advbench_avgtokens")
