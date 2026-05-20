#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set OpenAI API key (use the same key as in jbb_qwen.sh)
export OPENAI_API_KEY="${OPENAI_API_KEY:?set this to your OpenAI key for GPT-based eval}"

PYTHON_BIN="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Run GPT-4o evaluation script
$PYTHON_BIN jbb_evaluate_gpt.py

