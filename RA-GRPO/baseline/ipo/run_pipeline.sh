#!/bin/bash
# IPO 完整训练Pipeline（基于论文Section 4.1 Training Settings）
# 使用方法: bash run_pipeline.sh

set -e

echo "============================================================"
echo "IPO (Intervened Preference Optimization) Training Pipeline"
echo "论文: Towards Safe Reasoning in Large Reasoning Models"
echo "       via Corrective Intervention"
echo "============================================================"

# 配置（可通过环境变量覆盖）
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"          # 论文使用N=1
NUM_TRIGGERS="${NUM_TRIGGERS:-6}"          # 论文使用6个safety triggers
EPOCHS="${EPOCHS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-./ipo_output}"
SKIP_STAGE2="${SKIP_STAGE2:-false}"

echo ""
echo "Configuration (论文设置):"
echo "  Model: $MODEL_NAME"
echo "  Rollouts per prompt (N): $NUM_ROLLOUTS"
echo "  Safety triggers: $NUM_TRIGGERS"
echo "  Training epochs: $EPOCHS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Skip Stage 2: $SKIP_STAGE2"
echo ""

# =============================================================
# Step 1: 下载STAR-1数据集
# 论文使用STAR-1的1000个有害提示 + 915个良性提示
# =============================================================
echo "============================================================"
echo "Step 1: Downloading STAR-1 dataset..."
echo "  - 1000 harmful prompts (for IPO training)"
echo "  - 915 benign prompts (for anti over-refusal)"
echo "============================================================"
python scripts/download_data.py

# =============================================================
# Step 2: 生成推理轨迹
# 对每个有害提示，使用base LRM生成推理过程
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
# Step 3: 构建IPO训练数据
# 执行Algorithm 1: 检测compliance cues → 用safety triggers替换
# 论文: 对6个safety triggers各执行一次，合并数据集
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
# Step 4: IPO训练（第一阶段）
# 使用DPO在偏好对上训练
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
# Step 5 & 6: 第二阶段训练（防止过度拒绝）
# 论文: 使用915个良性提示，对比base LRM正常回复与训练后模型的拒绝回复
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
echo "论文期望的数据集大小参考:"
echo "  DS-8B: ~1,438 samples"
echo "  DS-7B: ~1,346 samples"
echo "  Qwen3-8B: ~520 samples"
echo "============================================================"
