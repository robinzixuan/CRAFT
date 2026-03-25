#!/bin/bash

set -x

module load cuda/12.4.0-gcc-12.4.0
export CUDA_HOME=/software/cuda/cuda-12.1.0 
module load glibc/2.28-gcc-12.4.0
export HF_HOME="/projects/p32013/.cache/"
export HF_TOKEN='REDACTED_HF_TOKEN'
# Set up environment variables for wandb
export WANDB_PROJECT="grpo-training"
export WANDB_ENABLED="true"


python SFT/embedding.py