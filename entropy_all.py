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

def compute_entropy(logits: torch.Tensor):
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item() if is_single_input else entropy

def process_text_series(series: pd.Series, model_name: str, output_dir: str, max_length: int = 1024):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(model_name, token=os.environ["HUGGINGFACE_HUB_TOKEN"])

    all_tokens = []
    all_entropies = []
    token_records = []

    for text_index, text in enumerate(tqdm(series, desc=f"Processing {os.path.basename(output_dir)}")):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        outputs = model(**inputs)
        token_entropies = compute_entropy(outputs.logits)
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        entropies = token_entropies[0].detach().cpu().tolist()
        all_tokens.extend(tokens)
        all_entropies.extend(entropies)

        for pos, (tok, ent) in enumerate(zip(tokens, entropies)):
            token_records.append({
                "token": tok,
                "entropy": ent,
                "dataset": os.path.basename(output_dir),
                "text_index": text_index,
                "position_in_text": pos
            })

    # Save full token-level CSV
    df_tokens = pd.DataFrame(token_records)
    df_tokens.to_csv(os.path.join(output_dir, "token_entropy.csv"), index=False)

    all_tokens = np.array(all_tokens)
    all_entropies = np.array(all_entropies)

    # High/low entropy token freq
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

    with open(os.path.join(output_dir, "top_tokens.txt"), "w", encoding='utf-8') as f_top, \
         open(os.path.join(output_dir, "bottom_tokens.txt"), "w", encoding='utf-8') as f_bottom:
        f_top.write("Top 100 High Entropy Tokens and their Frequencies:\n")
        for i, (token, freq) in enumerate(list(top_token_freq.items())[:100]):
            f_top.write(f"{i+1}. Token: '{token}', Frequency: {freq}\n")
        f_bottom.write("Top 100 Low Entropy Tokens and their Frequencies:\n")
        for i, (token, freq) in enumerate(list(bottom_token_freq.items())[:100]):
            f_bottom.write(f"{i+1}. Token: '{token}', Frequency: {freq}\n")

    # Write summary file
    mean_entropy = np.mean(all_entropies)
    std_entropy = np.std(all_entropies)
    summary_path = os.path.join(output_dir, "entropy_summary.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(f"Total tokens analyzed: {len(all_tokens)}\n")
        f.write(f"Mean entropy: {mean_entropy:.4f}\n")
        f.write(f"Standard deviation: {std_entropy:.4f}\n")
        f.write(f"Unique high entropy tokens: {len(top_token_freq)}\n")
        f.write(f"Unique low entropy tokens: {len(bottom_token_freq)}\n")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_entropies, bins=100, color='skyblue', edgecolor='black')
    plt.title("Token Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Token Count")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "entropy_histogram.png"))
    plt.close()

    print(f"[{os.path.basename(output_dir)}] Output saved in {output_dir}")

def main():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    base_output_dir = "./entropy_results_instruct"
    os.makedirs(base_output_dir, exist_ok=True)

    datasets = {
        "advbench": "/projects/p32013/XLLM/Dataset/Advbench.csv",
        # "strongreject": "/projects/p32013/neurons/strongreject/strongreject_dataset/strongreject_dataset.csv",
        # "safety-neuron": "/projects/p32013/neurons/Safety-Neuron/neuron_detection/corpus_all/english.txt"
    }

    for dataset_name, csv_path in datasets.items():
        print(f"\n=== Processing Dataset: {dataset_name} ===")
        output_dir = os.path.join(base_output_dir, dataset_name)
        if dataset_name == "advbench":
            df = pd.read_csv(csv_path)
            process_text_series(df['target'], model_name=model_name, output_dir=output_dir)
        elif dataset_name == "strongreject":
            df = pd.read_csv(csv_path)
            process_text_series(df['forbidden_prompt'], model_name=model_name, output_dir=output_dir)
        elif dataset_name == "safety-neuron":
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            process_text_series(pd.Series(lines), model_name=model_name, output_dir=output_dir)
        else:
            print(f"Unknown dataset: {dataset_name}. Skipping...")
        print(f"Finished processing {dataset_name} dataset.")
    print("\nAll datasets processed successfully.")

if __name__ == "__main__":
    main()
