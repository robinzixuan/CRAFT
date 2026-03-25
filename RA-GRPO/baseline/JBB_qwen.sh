#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

export OPENAI_API_KEY=REDACTED_OPENAI_KEY
export HF_HOME="/projects/p32013/.cache/"
export TOGETHER_API_KEY=tgp_v1_8Gfeb1DZcPh6V3YqJlrm_N5xH1wXhogbDm1QRAMFVfo
# 只加载 CUDA 12（vLLM 需要 libcudart.so.12）
module load cuda/12.4.0-gcc-12.4.0

PYTHON_BIN="/projects/p32013/conda_envs/jailbreakbench/bin/python3"

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 运行 JailbreakBench 评估脚本
$PYTHON_BIN baseline/JBB_qwen.py
