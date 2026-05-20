"""
IPO main training script.
Example usage: python scripts/train_ipo.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ipo import (
    IPODataBuilder,
    IPOTrainer,
    ReasoningIntervention,
    SafetyTriggerGenerator,
)
from ipo.trainer import IPOConfig
from ipo.data_builder import SyntheticDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="IPO Training Script")
    
    # Data arguments
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to training data JSON file (if not provided, synthetic data is used)"
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data JSON file"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Base model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ipo_output",
        help="Output directory"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="使用合成数据进行测试"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("IPO (Intervened Preference Optimization) Training")
    print("=" * 60)
    
    # Create data builder
    intervention = ReasoningIntervention(use_rule_based=True)
    data_builder = IPODataBuilder(intervention=intervention)
    
    # Prepare training data
    if args.train_data:
        print(f"Loading training data from {args.train_data}")
        train_dataset = data_builder.prepare_dataset_from_json(args.train_data)
    elif args.use_synthetic:
        print("Using synthetic data for testing...")
        synth_gen = SyntheticDataGenerator()
        samples = synth_gen.generate_test_samples(n=5)
        ipo_samples = data_builder.build_from_unsafe_samples(
            samples, 
            num_triggers_per_sample=2,
            show_progress=True,
        )
        print(f"Built {len(ipo_samples)} IPO samples")
        
        # Save synthetic data
        data_path = Path(args.output_dir) / "synthetic_train_data.json"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_builder.save_to_json(ipo_samples, str(data_path), format="dpo")
        
        # Create dataset
        train_dataset = data_builder.prepare_dataset(ipo_samples)
    else:
        print("Error: Please provide --train_data or --use_synthetic")
        sys.exit(1)
    
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Prepare evaluation data
    eval_dataset = None
    if args.eval_data:
        print(f"Loading evaluation data from {args.eval_data}")
        eval_dataset = data_builder.prepare_dataset_from_json(args.eval_data)
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Create training config
    config = IPOConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_lora=not args.no_lora,
        use_4bit=not args.no_4bit,
        seed=args.seed,
    )
    
    print("\nTraining Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Output: {config.output_dir}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  DPO beta: {config.beta}")
    print(f"  LoRA: {config.use_lora} (r={config.lora_r}, alpha={config.lora_alpha})")
    print(f"  4-bit quantization: {config.use_4bit}")
    
    # Create trainer and train
    trainer = IPOTrainer(config)
    trainer.train(train_dataset, eval_dataset)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
