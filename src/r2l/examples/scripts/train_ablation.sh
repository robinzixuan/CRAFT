#!/bin/bash
set -x

# 清理旧的 Ray 会话
unset RAY_ADDRESS
ray stop --force 2>/dev/null || true


# 使用 /tmp 作为 Ray 临时目录（有更多空间）
export RAY_TMPDIR="/tmp/ray_workspace"
mkdir -p $RAY_TMPDIR
echo "Ray 日志目录: $RAY_TMPDIR"

# 增加调试日志
# export RAY_LOG_TO_STDERR=1
# export VLLM_LOGGING_LEVEL=DEBUG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
export LOG_LEVEL=DEBUG
export OPENAI_API_KEY="${OPENAI_API_KEY:?set this to your OpenAI key for GPT-based eval}"
# Cluster setup (uncomment / adapt for your environment):
# module load cuda/12.4 cudnn/8.9.7-cuda-12 vllm/0.10.1
# export CUDA_HOME=/path/to/cuda
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}/"
export WANDB_PROJECT="lca-safety-grpo"
export WANDB_ENABLED="true"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export PATH=/usr/bin:$PATH
export RAY_DISABLE_IMPORT_WARNING=1
export CUDA_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_GPU_ALLOCATOR="cuda"
export FLASH_ATTENTION_DISABLE=1
export VLLM_USE_V1=0
export VLLM_DISABLE_RAY_INIT=1


# 3) 强制 Ray 使用 spawn（避免 fork + CUDA 死锁）
export RAY_FORCE_MULTIPROCESSING_START_METHOD=spawn

# 4) 禁止 placement group（否则必卡）
export VLLM_RAY_DISABLE_PLACEMENT_GROUP=1

MODEL_PATH=Qwen/Qwen3-4B-Thinking-2507

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

# 4 GPU 配置
$PYTHON_BIN -m verl.trainer.main \
    config=examples/configs/config_ablation.yaml \
    data.max_response_length=2048 \
    data.max_prompt_length=1024 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.padding_free=false \
    worker.actor.use_torch_compile=false \
    worker.reward.model_path=${MODEL_PATH} \
    worker.reward.gpu_memory_utilization=0.25 \
    trainer.n_gpus_per_node=2 \
    trainer.experiment_name=qwen3_4b_think_safety_4_ablsion


