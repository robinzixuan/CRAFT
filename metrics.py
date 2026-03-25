import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "REDACTED_HF_TOKEN"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-3.2-3B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HUGGINGFACE_HUB_TOKEN"])
model = AutoModelForCausalLM.from_pretrained(model_name, token=os.environ["HUGGINGFACE_HUB_TOKEN"]).to(device)
model.eval()

df = pd.read_csv("/projects/p32013/neurons/XLLM-main/Dataset/Advbench_391_harmless.csv")  
prompts = df["text"].tolist()[:100]  

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def compute_hidden_entropy(hidden_states):
    probs = F.softmax(hidden_states, dim=-1)
    log_probs = F.log_softmax(hidden_states, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

if tokenizer.eos_token is None:
    raise ValueError("Tokenizer does not define an eos token.")
eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</s>")
if eos_token_id is None:
    raise ValueError("Tokenizer does not define eos_token_id or </s>.")

results = []

for text in tqdm(prompts):
    # 强制加入 <eos> 确保能找到
    text_with_eos = text + tokenizer.eos_token
    inputs = tokenizer(text_with_eos, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    logits = outputs.logits[0]                # [seq_len, vocab_size]
    hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
    input_ids = inputs.input_ids[0]

    # Token-level entropy
    token_entropy = compute_entropy(logits)             # [seq_len]
    hidden_entropy = compute_hidden_entropy(hidden_states)  # [seq_len]

    # <eos> 
    eos_indices = (input_ids == eos_token_id).nonzero(as_tuple=True)
    eos_idx = eos_indices[0].item() if len(eos_indices[0]) > 0 else len(input_ids) - 1
    eos_entropy = hidden_entropy[eos_idx]  # scalar

    hidden_entropy_diff = hidden_entropy - eos_entropy  # [seq_len]

    results.append({
        "prompt": text,
        "avg_normal_entropy": token_entropy.mean().item(),
        "avg_hidden_entropy": hidden_entropy.mean().item(),
        "avg_hidden_entropy_diff": hidden_entropy_diff.mean().item()
    })

df_out = pd.DataFrame(results)
df_out.to_csv("entropy_analysis_harmless.csv", index=False)
print("✓ Saved to entropy_analysis_harmless.csv")