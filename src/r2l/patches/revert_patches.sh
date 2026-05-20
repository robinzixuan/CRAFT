#!/bin/bash
# 还原环境适配补丁（当 flash_attn 可用时使用）
# 用法: source patches/revert_patches.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Reverting patches..."

# Revert Patch 1: sdpa -> flash_attention_2
FSDP_FILE="$PROJECT_ROOT/verl/workers/fsdp_workers.py"
if grep -q 'attn_implementation="sdpa"' "$FSDP_FILE" 2>/dev/null; then
    sed -i 's/attn_implementation="sdpa"/attn_implementation="flash_attention_2"/g' "$FSDP_FILE"
    echo "  ✓ Reverted fsdp_workers.py: sdpa -> flash_attention_2"
else
    echo "  ✓ fsdp_workers.py already using flash_attention_2 or not found"
fi

echo "Patches reverted successfully!"

