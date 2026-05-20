#!/usr/bin/env python3
"""
Complete LCA training script for safety alignment.
Loads data from dataset/data.json, splits train/test, and runs contrastive learning.
"""

import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from LCA import train_latent_contrastive
import argparse
import os
import wandb
from datetime import datetime

def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Load data from JSON file and preprocess for LCA training with three trace types.
    
    Args:
        data_path: Path to the JSON data file
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_data, test_data: Lists of dictionaries with 'reasoning_trace' and 'label' keys
        mu_safe, mu_unsafe, mu_rethink: Initial prototype vectors for three trace types
    """
    print(f"\n📂 DATA LOADING AND PREPROCESSING")
    print(f"   - Loading data from {data_path}...")
    
    # Load the JSON data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"   - Loaded {len(data)} examples from JSON file")
    
    # Filter and preprocess data
    print(f"   - Filtering and preprocessing data...")
    processed_data = []
    safe_count = 0
    unsafe_count = 0
    rethink_count = 0
    skipped_count = 0
    
    for item in data:
        # Use reasoning_trace as the main text and label for safety
        if 'reasoning_trace' in item and 'label' in item:
            label = item['label'].lower()
            
            # Support three trace types: safe, unsafe, rethink
            if label in ['safe', 'unsafe', 'rethink']:
                processed_data.append({
                    'reasoning_trace': item['reasoning_trace'],
                    'label': label
                })
                
                if label == 'safe':
                    safe_count += 1
                elif label == 'unsafe':
                    unsafe_count += 1
                elif label == 'rethink':
                    rethink_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    
    print(f"   - Processed {len(processed_data)} examples: {safe_count} safe, {unsafe_count} unsafe, {rethink_count} rethink")
    if skipped_count > 0:
        print(f"   - Skipped {skipped_count} examples (missing required fields or invalid labels)")
    
    # Split into train/test
    print(f"   - Splitting data into train/test (test_size={test_size})...")
    train_data, test_data = train_test_split(
        processed_data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[item['label'] for item in processed_data]
    )
    
    # Count labels in train/test splits
    train_safe = sum(1 for item in train_data if item['label'] == 'safe')
    train_unsafe = sum(1 for item in train_data if item['label'] == 'unsafe')
    train_rethink = sum(1 for item in train_data if item['label'] == 'rethink')
    test_safe = sum(1 for item in test_data if item['label'] == 'safe')
    test_unsafe = sum(1 for item in test_data if item['label'] == 'unsafe')
    test_rethink = sum(1 for item in test_data if item['label'] == 'rethink')
    
    print(f"   - Train: {len(train_data)} examples (safe={train_safe}, unsafe={train_unsafe}, rethink={train_rethink})")
    print(f"   - Test: {len(test_data)} examples (safe={test_safe}, unsafe={test_unsafe}, rethink={test_rethink})")
    
    # Initialize prototype vectors for three trace types (random for now, will be updated during training)
    print(f"   - Initializing prototype vectors (dimension=512)...")
    d_embed = 512  # embedding dimension
    mu_safe = np.random.randn(d_embed).astype(np.float32)
    mu_unsafe = np.random.randn(d_embed).astype(np.float32)
    mu_rethink = np.random.randn(d_embed).astype(np.float32)
    
    # Normalize prototypes
    mu_safe = mu_safe / np.linalg.norm(mu_safe)
    mu_unsafe = mu_unsafe / np.linalg.norm(mu_unsafe)
    mu_rethink = mu_rethink / np.linalg.norm(mu_rethink)
    
    print(f"   - Prototype norms: Safe={np.linalg.norm(mu_safe):.4f}, Unsafe={np.linalg.norm(mu_unsafe):.4f}, Rethink={np.linalg.norm(mu_rethink):.4f}")
    
    return train_data, test_data, mu_safe, mu_unsafe, mu_rethink

def setup_wandb(project_name="lca-safety-alignment", config=None):
    """
    Initialize Weights & Biases for experiment tracking.
    
    Args:
        project_name: Name of the wandb project
        config: Dictionary of configuration parameters to log
    
    Returns:
        wandb run object
    """
    print(f"\n📊 SETTING UP WANDB LOGGING")
    print(f"   - Project: {project_name}")
    
    # Initialize wandb run
    run = wandb.init(
        project=project_name,
        config=config,
        name=f"lca-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=["lca", "safety-alignment", "contrastive-learning"]
    )
    
    print(f"   - Run ID: {run.id}")
    print(f"   - Run URL: {run.url}")
    print(f"   - Wandb initialized successfully")
    
    return run

def save_checkpoint(epoch, proj_state, head_state, mu_safe, mu_unsafe, mu_rethink, 
                   optimizer_state, output_dir, is_best=False):
    """
    Save training checkpoint including model states and optimizer state.
    
    Args:
        epoch: Current epoch number
        proj_state: Projection head state dict
        head_state: Safety head state dict
        mu_safe, mu_unsafe, mu_rethink: Prototype vectors
        optimizer_state: Optimizer state dict
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best checkpoint so far
    """
    checkpoint = {
        'epoch': epoch,
        'projection_head_state': proj_state,
        'safety_head_state': head_state,
        'mu_safe': mu_safe,
        'mu_unsafe': mu_unsafe,
        'mu_rethink': mu_rethink,
        'optimizer_state': optimizer_state,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"   - Checkpoint saved: {checkpoint_path}")
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(output_dir, 'best_checkpoint.pt')
        torch.save(checkpoint, best_path)
        print(f"   - Best checkpoint saved: {best_path}")

def load_checkpoint(checkpoint_path, device='cuda'):
    """
    Load training checkpoint for resuming training.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load tensors on
    
    Returns:
        Dictionary containing checkpoint data
    """
    print(f"\n🔄 LOADING CHECKPOINT")
    print(f"   - Loading from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"   - Checkpoint loaded successfully")
    print(f"   - Epoch: {checkpoint['epoch']}")
    print(f"   - Timestamp: {checkpoint['timestamp']}")
    
    return checkpoint

def main():
    print(f"\n🚀 LCA TRAINING SCRIPT STARTING")
    print(f"=" * 60)
    
    parser = argparse.ArgumentParser(description='Train LCA model for safety alignment')
    parser.add_argument('--data_path', type=str, default='dataset/data.json',
                       help='Path to the dataset JSON file')
    parser.add_argument('--model_name', type=str, default='microsoft/DialoGPT-medium',
                       help='Base model name for training')
    parser.add_argument('--verifier_path', type=str, default=None,
                       help='Optional safety verifier model path')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--output_dir', type=str, default='outputs_llama',
                       help='Directory to save trained models')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='lca-safety-alignment',
                       help='Wandb project name')
    parser.add_argument('--resume_from', type=str, default="",
                       help='Path to checkpoint file to resume training from')
    parser.add_argument('--save_checkpoints', action='store_true',
                       help='Save checkpoints during training')
    parser.add_argument('--checkpoint_freq', type=int, default=1,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    print(f"\n⚙️ CONFIGURATION:")
    print(f"   - Data path: {args.data_path}")
    print(f"   - Model: {args.model_name}")
    print(f"   - Verifier: {args.verifier_path or 'None'}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Test size: {args.test_size}")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Wandb logging: {'Enabled' if args.use_wandb else 'Disabled'}")
    print(f"   - Resume from: {args.resume_from or 'None'}")
    print(f"   - Save checkpoints: {'Enabled' if args.save_checkpoints else 'Disabled'}")
    if args.save_checkpoints:
        print(f"   - Checkpoint frequency: Every {args.checkpoint_freq} epoch(s)")
    
    # Create output directory
    print(f"\n📁 SETTING UP OUTPUT DIRECTORY")
    print(f"   - Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"   - Output directory ready")
    
    # Initialize wandb if enabled
    wandb_run = None
    if args.use_wandb:
        config = {
            'model_name': args.model_name,
            'verifier_path': args.verifier_path,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'test_size': args.test_size,
            'output_dir': args.output_dir,
            'resume_from': args.resume_from,
            'save_checkpoints': args.save_checkpoints,
            'checkpoint_freq': args.checkpoint_freq
        }
        wandb_run = setup_wandb(args.wandb_project, config)
    
    # Load and preprocess data
    train_data, test_data, mu_safe, mu_unsafe, mu_rethink = load_and_preprocess_data(
        args.data_path, 
        test_size=args.test_size
    )
    
    print(f"\n🏃 STARTING LCA TRAINING")
    print(f"   - Model: {args.model_name}")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Train examples: {len(train_data)}")
    print(f"   - Test examples: {len(test_data)}")
    
    # Run LCA training
    proj_state, head_state, mu_safe_final, mu_unsafe_final, mu_rethink_final = train_latent_contrastive(
        model_name=args.model_name,
        data=train_data,
        mu_safe=mu_safe,
        mu_unsafe=mu_unsafe,
        mu_rethink=mu_rethink,
        epochs=args.epochs,
        bs=args.batch_size,
        verifier_name_or_path=args.verifier_path,
        wandb_run=wandb_run,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume_from
    )
    
    # Save trained models
    print(f"\n💾 SAVING TRAINED MODELS")
    print(f"   - Saving to directory: {args.output_dir}")
    
    # Save projection head
    proj_path = os.path.join(args.output_dir, 'projection_head.pt')
    print(f"   - Saving projection head to: {proj_path}")
    torch.save(proj_state, proj_path)
    
    # Save safety head
    head_path = os.path.join(args.output_dir, 'safety_head.pt')
    print(f"   - Saving safety head to: {head_path}")
    torch.save(head_state, head_path)
    
    # Save prototype vectors
    mu_safe_path = os.path.join(args.output_dir, 'mu_safe.npy')
    mu_unsafe_path = os.path.join(args.output_dir, 'mu_unsafe.npy')
    mu_rethink_path = os.path.join(args.output_dir, 'mu_rethink.npy')
    print(f"   - Saving prototype vectors...")
    np.save(mu_safe_path, mu_safe_final)
    np.save(mu_unsafe_path, mu_unsafe_final)
    np.save(mu_rethink_path, mu_rethink_final)
    
    # Save test data for evaluation
    test_data_path = os.path.join(args.output_dir, 'test_data.json')
    print(f"   - Saving test data to: {test_data_path}")
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"=" * 60)
    print(f"📁 Models saved to: {args.output_dir}")
    print(f"📊 Final Results:")
    print(f"   - Safe prototype norm: {np.linalg.norm(mu_safe_final):.4f}")
    print(f"   - Unsafe prototype norm: {np.linalg.norm(mu_unsafe_final):.4f}")
    print(f"   - Rethink prototype norm: {np.linalg.norm(mu_rethink_final):.4f}")
    print(f"   - Test examples saved: {len(test_data)}")
    print(f"=" * 60)
    
    # Finish wandb run if enabled
    if wandb_run:
        print(f"\n📊 FINISHING WANDB RUN")
        wandb.finish()
        print(f"   - Wandb run completed")

if __name__ == "__main__":
    main()
