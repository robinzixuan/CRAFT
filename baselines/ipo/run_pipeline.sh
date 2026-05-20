#!/bin/bash
# IPO full training pipeline (based on paper Section 4.1 Training Settings)
# Usage: bash run_pipeline.sh

set -e

echo "============================================================"
echo "IPO (Intervened Preference Optimization) Training Pipeline"
echo "论文: Towards Safe Reasoning in Large Reasoning Models"
echo "       via Corrective Intervention"
echo "============================================================"

# Configuration (can be overridden via environment variables)
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"          # 论文使用N=1
NUM_TRIGGERS="${NUM_TRIGGERS:-6}"          # 论文使用6个safety triggers
EPOCHS="${EPOCHS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-./ipo_output}"
SKIP_STAGE2="${SKIP_STAGE2:-false}"

echo ""
echo "Configuration (paper settings):"
echo "  Model: $MODEL_NAME"
echo "  Rollouts per prompt (N): $NUM_ROLLOUTS"
echo "  Safety triggers: $NUM_TRIGGERS"
echo "  Training epochs: $EPOCHS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Skip Stage 2: $SKIP_STAGE2"
echo ""

# =============================================================
# Step 1: Download STAR-1 dataset
# Paper uses 1,000 harmful prompts + 915 benign prompts from STAR-1
# =============================================================
echo "============================================================"
echo "Step 1: Downloading STAR-1 dataset..."
echo "  - 1000 harmful prompts (for IPO training)"
echo "  - 915 benign prompts (for anti over-refusal)"
echo "============================================================"
python scripts/download_data.py

# =============================================================
# Step 2: Generate reasoning rollouts
# For each harmful prompt, generate reasoning using the base LRM
# =============================================================
echo ""
echo "============================================================"
echo "Step 2: Generating reasoning rollouts from base LRM..."
echo "============================================================"
python scripts/generate_rollouts.py \
    --model_name "$MODEL_NAME" \
    --prompts_file data/star1_harmful_prompts.json \
    --output_file data/raw_rollouts.json \
    --num_rollouts "$NUM_ROLLOUTS"

# =============================================================
# Step 3: Build IPO training data
# Execute Algorithm 1: detect compliance cues -> substitute with safety triggers
# Paper: run once per safety trigger (6 total), then merge datasets
# =============================================================
echo ""
echo "============================================================"
echo "Step 3: Building IPO training data (Algorithm 1)..."
echo "  - Detecting compliance cues"
echo "  - Substituting with $NUM_TRIGGERS safety triggers"
echo "  - Constructing preference pairs"
echo "============================================================"
python scripts/build_data.py \
    --input data/raw_rollouts.json \
    --output data/ipo_train.json \
    --num_triggers "$NUM_TRIGGERS"

# =============================================================
# Step 4: IPO Training (Stage 1 - Safety Alignment)
# Train on preference pairs using DPO
# =============================================================
echo ""
echo "============================================================"
echo "Step 4: IPO Training (Stage 1 - Safety Alignment)..."
echo "============================================================"
python scripts/train_ipo.py \
    --train_data data/ipo_train.json \
    --model_name "$MODEL_NAME" \
    --output_dir "${OUTPUT_DIR}_stage1" \
    --epochs "$EPOCHS"

# =============================================================
# Step 5 & 6: Stage 2 Training (Anti Over-Refusal)
# Paper: use 915 benign prompts; contrast base LRM normal responses
# with the trained model's refusal responses
# =============================================================
if [ "$SKIP_STAGE2" != "true" ]; then
    echo ""
    echo "============================================================"
    echo "Step 5: Building Stage 2 data (Anti Over-Refusal)..."
    echo "  - Using 915 benign prompts from STAR-1"
    echo "  - chosen: normal responses from base LRM"
    echo "  - rejected: refusal responses from Stage 1 model"
    echo "============================================================"
    python scripts/build_stage2_data.py \
        --benign_prompts data/star1_benign_prompts.json \
        --base_model "$MODEL_NAME" \
        --trained_model "${OUTPUT_DIR}_stage1" \
        --output data/stage2_train.json
    
    echo ""
    echo "============================================================"
    echo "Step 6: IPO Training (Stage 2 - Anti Over-Refusal)..."
    echo "============================================================"
    python scripts/train_ipo.py \
        --train_data data/stage2_train.json \
        --model_name "${OUTPUT_DIR}_stage1" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 1
else
    echo ""
    echo "Skipping Stage 2 training..."
    cp -r "${OUTPUT_DIR}_stage1" "$OUTPUT_DIR"
fi

# =============================================================
# Step 7: 评估
# =============================================================
echo ""
echo "============================================================"
echo "Step 7: Evaluating model..."
echo "============================================================"
python scripts/evaluate.py \
    --model_path "$OUTPUT_DIR" \
    --use_default_prompts \
    --output "${OUTPUT_DIR}/eval_results.json"

echo ""
echo "============================================================"
echo "Pipeline completed!"
echo "============================================================"
echo "Model saved to: $OUTPUT_DIR"
echo "Evaluation results: ${OUTPUT_DIR}/eval_results.json"
echo ""
echo "Expected dataset sizes from paper:"
echo "  DS-8B: ~1,438 samples"
echo "  DS-7B: ~1,346 samples"
echo "  Qwen3-8B: ~520 samples"
echo "============================================================"
