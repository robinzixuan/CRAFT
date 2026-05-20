#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# export CUDA_VISIBLE_DEVICES=0
# export NVIDIA_VISIBLE_DEVICES=0
# export RAY_GPU_IDS=0
export OPENAI_API_KEY="${OPENAI_API_KEY:?set this to your OpenAI key for GPT-based eval}"
# Cluster setup (uncomment / adapt for your environment):
# module load cuda/12.4 cudnn/8.9.7-cuda-12 vllm/0.10.1
# export CUDA_HOME=/path/to/cuda
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}/"
# Set up environment variables for wandb
export WANDB_PROJECT="lca-safety-grpo"
export WANDB_ENABLED="true"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH=/usr/bin:$PATH

export RAY_EXPERIMENTAL_GPU_ALLOCATOR="cuda"
export FLASH_ATTENTION_DISABLE=1
export CUDA_VISIBLE_DEVICES=2,3



MODEL_PATH=Qwen/Qwen3-4B-Thinking-2507  # replace it with your local file path

if [ -n "$CONDA_PREFIX" ] && [ -f "${CONDA_PREFIX}/bin/python3" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "Warning: Using system python3, codetiming may not be available"
    PYTHON_BIN="python3"
fi

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 已修改 verl/workers/fsdp_workers.py 使用 sdpa 替代 flash_attention_2
$PYTHON_BIN -m verl.trainer.main \
    config=examples/configs/config_qwen3_4b_thinking_2gpu.yaml \
    data.max_response_length=2048 \
    data.max_prompt_length=1024 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.padding_free=false \
    worker.actor.use_torch_compile=false \
    worker.reward.model_path=${MODEL_PATH} \
    worker.reward.gpu_memory_utilization=0.35 \
    trainer.n_gpus_per_node=2 \
    trainer.experiment_name=qwen3_4b_think_safety_2_2gpu
