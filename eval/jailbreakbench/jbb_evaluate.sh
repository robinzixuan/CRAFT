#!/bin/bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set Together API key
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?set this to your Together API key}"

PYTHON_BIN="${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Run evaluation script
$PYTHON_BIN jbb_evaluate.py

