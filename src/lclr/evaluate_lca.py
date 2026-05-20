#!/usr/bin/env python3
"""
Evaluation script for LCA model to verify contrastive learning is working.
Tests the trained projection head and safety head on test data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from lca import ProjectionHead, SafetyHead, pool
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

class LCAModel(nn.Module):
    """Complete LCA model for inference."""
    def __init__(self, model_name, proj_state, head_state, mu_safe, mu_unsafe, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_hidden_states=True, 
            torch_dtype=torch.bfloat16
        ).eval().to(device)
        
        # Load trained components
        self.proj = ProjectionHead(self.lm.config.hidden_size).to(device)
        self.proj.load_state_dict(proj_state)
        
        self.head = SafetyHead(512).to(device)
        self.head.load_state_dict(head_state)
        
        # Load prototype vectors
        self.mu_safe = torch.tensor(mu_safe, device=device, dtype=torch.float32)
        self.mu_unsafe = torch.tensor(mu_unsafe, device=device, dtype=torch.float32)
    
    def encode_text(self, texts):
        """Encode texts to embeddings."""
        enc = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            h = self.lm(**enc).hidden_states[-1]
        
        z_raw = pool(h, enc["attention_mask"])
        z = self.proj(z_raw)
        return z
    
    def predict_safety(self, texts):
        """Predict safety scores for texts."""
        z = self.encode_text(texts)
        
        # Get safety predictions
        logits = self.head(z)
        safety_probs = torch.sigmoid(logits)
        
        # Get prototype similarities
        z_norm = F.normalize(z, dim=-1)
        cos_safe = (z_norm * F.normalize(self.mu_safe, dim=-1)).sum(-1)
        cos_unsafe = (z_norm * F.normalize(self.mu_unsafe, dim=-1)).sum(-1)
        
        return {
            'safety_probs': safety_probs.cpu().numpy(),
            'cos_safe': cos_safe.cpu().numpy(),
            'cos_unsafe': cos_unsafe.cpu().numpy(),
            'embeddings': z.cpu().numpy()
        }

def evaluate_model(model, test_data, batch_size=32):
    """Evaluate the trained LCA model."""
    print("Evaluating LCA model...")
    
    all_predictions = []
    all_labels = []
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        texts = [item['reasoning_trace'] for item in batch]
        labels = [1 if item['label'] == 'safe' else 0 for item in batch]
        
        # Get predictions
        results = model.predict_safety(texts)
        
        all_predictions.extend(results['safety_probs'])
        all_labels.extend(labels)
        all_embeddings.extend(results['embeddings'])
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    embeddings = np.array(all_embeddings)
    
    # Binary predictions (threshold at 0.5)
    binary_preds = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, binary_preds, average='binary')
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Analyze contrastive learning effectiveness
    print(f"\nContrastive Learning Analysis:")
    print(f"Mean safety probability for safe examples: {predictions[labels==1].mean():.4f}")
    print(f"Mean safety probability for unsafe examples: {predictions[labels==0].mean():.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'labels': labels,
        'embeddings': embeddings
    }

def visualize_embeddings(embeddings, labels, output_path='embeddings_plot.png'):
    """Visualize embeddings using t-SNE."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        print("Creating t-SNE visualization...")
        
        # Reduce dimensionality
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['green' if label == 1 else 'red' for label in labels]
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.6)
        plt.title('LCA Embeddings Visualization (Green=Safe, Red=Unsafe)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Embeddings plot saved to {output_path}")
        
    except ImportError:
        print("sklearn not available for t-SNE visualization")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained LCA model')
    parser.add_argument('--model_name', type=str, default='microsoft/DialoGPT-medium',
                       help='Base model name')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory containing trained models')
    parser.add_argument('--test_data', type=str, default='outputs/test_data.json',
                       help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Load trained models
    print("Loading trained models...")
    proj_state = torch.load(f"{args.output_dir}/projection_head.pt", map_location=args.device)
    head_state = torch.load(f"{args.output_dir}/safety_head.pt", map_location=args.device)
    mu_safe = np.load(f"{args.output_dir}/mu_safe.npy")
    mu_unsafe = np.load(f"{args.output_dir}/mu_unsafe.npy")
    
    # Create model
    model = LCAModel(
        args.model_name, 
        proj_state, 
        head_state, 
        mu_safe, 
        mu_unsafe, 
        device=args.device
    )
    
    # Evaluate
    results = evaluate_model(model, test_data, args.batch_size)
    
    # Visualize embeddings
    visualize_embeddings(
        results['embeddings'], 
        results['labels'], 
        f"{args.output_dir}/embeddings_plot.png"
    )
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
