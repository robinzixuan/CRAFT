#!/usr/bin/env bash
set -euo pipefail

# 复现论文配置：8张 NVIDIA A800-SXM4-80GB GPU
# 论文参数：Table 11 - Training Details of ReasoningShield

# 基础路径
ROOT_DIR="/projects/p32013/neurons/RA-GRPO"
DATASET_DIR="${ROOT_DIR}/baseline/reasoningshield/ReasoningShield/reasoningshield_Dataset/reasoningshield-train"

# 模型与输出目录
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Thinking}"
OUTPUT_DIR_SFT="${OUTPUT_DIR_SFT:-${ROOT_DIR}/baseline/reasoningshield/ReasoningShield/outputs/qwen3-4b-thinking_sft_8a800}"
OUTPUT_DIR_DPO="${OUTPUT_DIR_DPO:-${ROOT_DIR}/baseline/reasoningshield/ReasoningShield/outputs/qwen3-4b-thinking_dpo_8a800}"

# 多GPU训练：使用 torchrun 启动，自动使用所有可见GPU
# 如果只想使用8张GPU，可以设置: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Stage 1: Supervised Fine-Tuning (SFT)
# 数据集: 4,358 agreed-upon samples (Sa)
# 参数: batch_size=2, grad_acc=8, lr=1e-5, epochs=3, cosine scheduler, warmup=0.1, bf16
# 有效 batch size = 2 × 8 (grad_acc) × 8 (gpu) = 128
torchrun --nproc_per_node=8 llamafactory-cli train \
  --stage sft \
  --model_name_or_path "${MODEL_NAME}" \
  --dataset reasoningshield_stage1_sft \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR_SFT}" \
  --finetuning_type full \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --report_to none

# Stage 2: Direct Preference Optimization (DPO)
# 数据集: 2,642 hard negative samples (Sh)
# 参数: batch_size=2, grad_acc=8, lr=2e-6, epochs=2, cosine scheduler, warmup=0.1, bf16
# 其他设置与 Stage 1 保持一致
torchrun --nproc_per_node=8 llamafactory-cli train \
  --stage dpo \
  --model_name_or_path "${OUTPUT_DIR_SFT}" \
  --dataset reasoningshield_stage2_dpo \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR_DPO}" \
  --finetuning_type full \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-6 \
  --num_train_epochs 2 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --report_to none
