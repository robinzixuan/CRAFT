#!/bin/bash
set -x

cd /projects/p32013/neurons/RA-GRPO

# 设置 Together API key
export TOGETHER_API_KEY="tgp_v1_8Gfeb1DZcPh6V3YqJlrm_N5xH1wXhogbDm1QRAMFVfo"

# 使用 jailbreakbench conda 环境
PYTHON_BIN="/projects/p32013/conda_envs/jailbreakbench/bin/python3"

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# 运行评估脚本
$PYTHON_BIN baseline/JBB_evaluate.py

