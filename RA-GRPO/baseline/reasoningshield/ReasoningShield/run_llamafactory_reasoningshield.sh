#!/usr/bin/env bash
set -euo pipefail

# 基础路径
ROOT_DIR="/projects/p32013/neurons/RA-GRPO"
DATASET_DIR="${ROOT_DIR}/baseline/reasoningshield/ReasoningShield/reasoningshield_Dataset/reasoningshield-train"

# 模型与输出目录（如模型名不同，可修改 MODEL_NAME）
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking}"
OUTPUT_DIR_SFT="${OUTPUT_DIR_SFT:-${ROOT_DIR}/baseline/reasoningshield/ReasoningShield/outputs/qwen3-4b-thinking_sft}"
OUTPUT_DIR_DPO="${OUTPUT_DIR_DPO:-${ROOT_DIR}/baseline/reasoningshield/ReasoningShield/outputs/qwen3-4b-thinking_dpo}"

# 单卡训练配置（GH200 96GB）
# 论文使用 8 张 A800，有效 batch size = 2 × 8 × 8 = 128
# 单卡保持 batch_size=2，通过增加 gradient_accumulation 来匹配有效 batch size
# 如果 OOM，可以减小 batch_size 到 1 或启用 gradient_checkpointing

# Stage 1: SFT (3 epochs, lr=1e-5, bs=2, grad_acc=64, cosine, warmup=0.1, bf16)
llamafactory-cli train \
  --stage sft \
  --model_name_or_path "${MODEL_NAME}" \
  --dataset reasoningshield_stage1_sft \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR_SFT}" \
  --finetuning_type full \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --gradient_checkpointing true \
  --report_to none

# Stage 2: DPO (2 epochs, lr=2e-6, bs=2, grad_acc=64, cosine, warmup=0.1, bf16)
llamafactory-cli train \
  --stage dpo \
  --model_name_or_path "${OUTPUT_DIR_SFT}" \
  --dataset reasoningshield_stage2_dpo \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR_DPO}" \
  --finetuning_type full \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 64 \
  --learning_rate 2e-6 \
  --num_train_epochs 2 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --gradient_checkpointing true \
  --report_to none
