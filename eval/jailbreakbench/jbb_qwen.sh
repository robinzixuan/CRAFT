#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

export OPENAI_API_KEY="${OPENAI_API_KEY:?set this to your OpenAI key for GPT-based eval}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}/"
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?set this to your Together API key}"
# Cluster setup (uncomment / adapt for your environment):
# module load cuda/12.4 cudnn/8.9.7-cuda-12 vllm/0.10.1
# export CUDA_HOME=/path/to/cuda

PYTHON_BIN="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 运行 JailbreakBench 评估脚本
$PYTHON_BIN baseline/JBB_qwen.py
