#!/bin/bash
# Apply environment-compatibility patches
# Usage: source patches/apply_patches.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Applying patches for local environment..."

# Patch 1: flash_attention_2 -> sdpa (use when flash_attn is not installed)
FSDP_FILE="$PROJECT_ROOT/verl/workers/fsdp_workers.py"
if grep -q 'attn_implementation="flash_attention_2"' "$FSDP_FILE" 2>/dev/null; then
    sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/g' "$FSDP_FILE"
    echo "  ✓ Patched fsdp_workers.py: flash_attention_2 -> sdpa"
else
    echo "  ✓ fsdp_workers.py already patched or not found"
fi

echo "Patches applied successfully!"

