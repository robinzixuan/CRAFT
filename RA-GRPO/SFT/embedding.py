from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, numpy as np, json
from tqdm import tqdm
import argparse





def embed_batch(texts, batch_size=8, tok=None, mdl=None):
    """Embed a batch of texts efficiently"""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        enc = tok(batch_texts, return_tensors="pt", truncation=True, 
                 max_length=2048, padding=True).to(mdl.device)
        
        with torch.no_grad():
            outputs = mdl(**enc, output_hidden_states=True)
            h_last = outputs.hidden_states[-10]  # [batch_size, T, d]
        
        # Use last token pooling (instead of mean pooling)
        attention_mask = enc.attention_mask  # [batch_size, seq_len]
        last_indices = attention_mask.sum(dim=1) - 1  # [batch_size] - indices of last valid tokens
        B, T, H = h_last.shape
        rows = torch.arange(B, device=h_last.device)
        h_last_tokens = h_last[rows, last_indices, :]  # [batch_size, H] - last token embeddings
        
        # Convert to numpy and add to results
        batch_embeddings = h_last_tokens.cpu().float().numpy()
        embeddings.extend(batch_embeddings)
    
    return embeddings

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for reasoning traces')
    parser.add_argument('--input', default='dataset/data.json', help='Input data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--output_dir', default='dataset', help='Output directory for embeddings')
    parser.add_argument('--model_name', default='meta-llama/Llama-3.1-8B-Instruct')
    args = parser.parse_args()
    

    model_name = args.model_name
    tok = AutoTokenizer.from_pretrained(model_name)

    tok.pad_token = tok.eos_token
    
    mdl = AutoModelForCausalLM.from_pretrained(model_name,
                output_hidden_states=True, torch_dtype=torch.bfloat16).eval().cuda()
    # Load data
    print("Loading data...")
    with open(args.input, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Separate data by labels
    safe_data = [ex for ex in data if ex["label"] == "safe"]
    unsafe_data = [ex for ex in data if ex["label"] == "unsafe"]
    rethink_data = [ex for ex in data if ex["label"] == "rethink"]

    print(f"Safe: {len(safe_data)}, Unsafe: {len(unsafe_data)}, Rethink: {len(rethink_data)}")

    # Process each category with batch embedding
    batch_size = args.batch_size

    if safe_data:
        print("Processing safe examples...")
        safe_texts = [ex["reasoning_trace"] for ex in safe_data]
        safe_z = embed_batch(safe_texts, batch_size=batch_size, tok=tok, mdl=mdl)
        np.save(f"{args.output_dir}/safe_latents.npy", np.stack(safe_z))
        print(f"Saved {len(safe_z)} safe embeddings")

    if unsafe_data:
        print("Processing unsafe examples...")
        unsafe_texts = [ex["reasoning_trace"] for ex in unsafe_data]
        unsafe_z = embed_batch(unsafe_texts, batch_size=batch_size, tok=tok, mdl=mdl)
        np.save(f"{args.output_dir}/unsafe_latents.npy", np.stack(unsafe_z))
        print(f"Saved {len(unsafe_z)} unsafe embeddings")

    if rethink_data:
        print("Processing rethink examples...")
        rethink_texts = [ex["reasoning_trace"] for ex in rethink_data]
        rethink_z = embed_batch(rethink_texts, batch_size=batch_size, tok=tok, mdl=mdl)
        np.save(f"{args.output_dir}/rethink_latents.npy", np.stack(rethink_z))
        print(f"Saved {len(rethink_z)} rethink embeddings")

    print("All embeddings saved successfully!")

if __name__ == "__main__":
    main()
