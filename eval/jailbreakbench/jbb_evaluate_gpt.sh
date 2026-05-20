#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置 OpenAI API key（使用你在 JBB_qwen.sh 中设置的 key）
export OPENAI_API_KEY="${OPENAI_API_KEY:?set this to your OpenAI key for GPT-based eval}"

PYTHON_BIN="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 运行 GPT-4o 评估脚本
$PYTHON_BIN baseline/JBB_evaluate_gpt.py

