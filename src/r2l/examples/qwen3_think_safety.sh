#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# export CUDA_VISIBLE_DEVICES=0
# export NVIDIA_VISIBLE_DEVICES=0
# export RAY_GPU_IDS=0
export OPENAI_API_KEY=REDACTED_OPENAI_KEY
module load cuda/12.4.0-gcc-12.4.0
module load gcc/12.4.0-gcc-8.5.0
module load cudnn/8.9.7.29-12-cuda-gcc-12.4.0
export CUDA_HOME=/software/cuda/cuda-12.1.0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH 
nvcc --version
module load glibc/2.28-gcc-12.4.0
module load vllm/0.10.1-gpt-oss
export HF_HOME="/projects/p32013/.cache/"
# Set up environment variables for wandb
export WANDB_PROJECT="lca-safety-grpo"
export WANDB_ENABLED="true"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH=/usr/bin:$PATH

export RAY_EXPERIMENTAL_GPU_ALLOCATOR="cuda"
export FLASH_ATTENTION_DISABLE=1



MODEL_PATH=Qwen/Qwen3-4B-Thinking-2507  # replace it with your local file path

if [ -n "$CONDA_PREFIX" ] && [ -f "${CONDA_PREFIX}/bin/python3" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python3"
elif [ -f "/projects/p32013/conda_envs/vllm/bin/python3" ]; then
    PYTHON_BIN="/projects/p32013/conda_envs/vllm/bin/python3"
    export PATH="/projects/p32013/conda_envs/vllm/bin:$PATH"
else
    echo "Warning: Using system python3, codetiming may not be available"
    PYTHON_BIN="python3"
fi

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 已修改 verl/workers/fsdp_workers.py 使用 sdpa 替代 flash_attention_2
$PYTHON_BIN -m verl.trainer.main \
    config=examples/config1.yaml \
    data.max_response_length=2048 \
    data.max_prompt_length=1024 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.padding_free=false \
    worker.reward.model_path=${MODEL_PATH} \
    worker.reward.gpu_memory_utilization=0.40 \
    trainer.n_gpus_per_node=2 \
    trainer.experiment_name=qwen3_4b_think_safety
