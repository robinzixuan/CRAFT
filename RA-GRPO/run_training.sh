#!/bin/bash

# LCA Training Script
# This script runs the complete LCA training pipeline
set -x

module load cuda/12.4.0-gcc-12.4.0
export CUDA_HOME=/software/cuda/cuda-12.1.0 
module load glibc/2.28-gcc-12.4.0
export HF_HOME="/projects/p32013/.cache/"
# Set up environment variables for wandb
export WANDB_PROJECT="grpo-training"
export WANDB_ENABLED="true"



# Run training
echo "Step 2: Starting LCA training..."
python SFT/train_lca.py \
    --data_path dataset/data.json \
    --model_name  deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --epochs 8 \
    --batch_size 32\
    --test_size 0.2 \
    --output_dir outputs \
    --use_wandb \
    --wandb_project easy-r1 \
    --save_checkpoints \
    --checkpoint_freq 2 

    
if [ $? -ne 0 ]; then
    echo "❌ Training failed."
    exit 1
fi

echo "✅ Training completed!"

# # Run evaluation
# echo "Step 3: Evaluating trained model..."
# python evaluate_lca.py \
#     --model_name microsoft/DialoGPT-medium \
#     --output_dir outputs \
#     --test_data outputs/test_data.json \
#     --batch_size 32
  

# if [ $? -ne 0 ]; then
#     echo "❌ Evaluation failed."
#     exit 1
# fi

# echo "✅ Evaluation completed!"
# echo "🎉 LCA training pipeline completed successfully!"
# echo "Check the outputs/ directory for trained models and results."
