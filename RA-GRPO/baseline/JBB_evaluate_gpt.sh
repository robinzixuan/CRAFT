#!/bin/bash
set -x

cd /projects/p32013/neurons/RA-GRPO

# 设置 OpenAI API key（使用你在 JBB_qwen.sh 中设置的 key）
export OPENAI_API_KEY="REDACTED_OPENAI_KEY"

# 使用 jailbreakbench conda 环境
PYTHON_BIN="/projects/p32013/conda_envs/jailbreakbench/bin/python3"

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 运行 GPT-4o 评估脚本
$PYTHON_BIN baseline/JBB_evaluate_gpt.py

