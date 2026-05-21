# Reproducibility Guide

This document lists the exact hardware, software, seeds, and commands that
produced the numbers in the CRAFT paper.

## Hardware

| Experiment | GPUs | CPU RAM |
| --- | --- | --- |
| LCLR training (Phase 1) | 1× A100 80 GB | 64 GB |
| R²L training, Qwen3-4B-Thinking (paper main) | 4× A100 80 GB | 256 GB |
| R²L training, DeepSeek-R1-Distill-Llama-8B | 4× A100 80 GB | 256 GB |
| Eval (JBB + StrongReject) | 1× A100 40 GB | 32 GB |

CRAFT was trained on Northwestern's Quest cluster; the original module-load
commands were:

```bash
module load cuda/12.4.0-gcc-12.4.0
module load gcc/12.4.0-gcc-8.5.0
module load cudnn/8.9.7.29-12-cuda-gcc-12.4.0
module load glibc/2.28-gcc-12.4.0
module load vllm/0.10.1-gpt-oss
```

Adapt for your own environment.

## Software

| Package | Pinned version |
| --- | --- |
| Python | 3.10 |
| PyTorch | ≥ 2.4 (CUDA 12.4) |
| transformers | ≥ 4.54 |
| flash-attn | ≥ 2.4.3 |
| vLLM | 0.10.1 |
| ray | (latest pinned by EasyR1) |

A Docker image is provided at `src/r2l/Dockerfile`.

## Seeds

The R²L config `data.seed` defaults to **238**, the LCLR data split uses
`random_state=42`. We report mean ± std over **three** repetitions with
seeds {238, 239, 240}. Variances are ≤ 0.2 on every reported metric.

## Per-table reproduction

### Table 1 — Safety (paper §6.1)

For each (model, method) row:

```bash
# 1. Train (if not using a released checkpoint)
bash src/r2l/examples/scripts/train_<MODEL>.sh

# 2. Merge checkpoint to HF format
python src/r2l/scripts/model_merger.py --local_dir ./checkpoints/<EXP>/global_step_<N>/actor

# 3. Evaluate
bash eval/jailbreakbench/jbb_qwen.sh  # set paths in config.yaml at the repo root first
python -c "from strongreject.evaluate import evaluate; import json, sys; data=json.load(open(sys.argv[1])); [print(evaluate(r['forbidden_prompt'],r['response'])) for r in data]" <output_from_jbb>.json
```

### Table 2 — Reasoning capability (paper §6.2)

Run each benchmark harness on the merged CRAFT checkpoint; see
`eval/README.md` for the list of harnesses.

### Table 3 — Ablation (paper §6.3)

```bash
bash src/r2l/examples/scripts/train_ablation.sh
```

To toggle individual reward components, edit
`src/r2l/examples/configs/config_ablation.yaml` and set the corresponding
weight to 0:

| Ablation row | Edit in config_ablation.yaml |
| --- | --- |
| no L_cons (consistency) | `worker.reward.w_cons: 0` |
| no R_ls (latent semantic) | `worker.reward.w_lat: 0` |
| no R_txt (textual safety) | `worker.reward.w_txt: 0` |
| no LCLR (random init heads) | use a fresh `projection_head.pt` / `safety_head.pt` |

Then evaluate as in Table 1.

### Fig. 4 — Adaptive jailbreaks

See `eval/advanced_attacks.md`.

### Fig. 5 — Latent geometry on Qwen3-0.6B

```bash
bash src/r2l/examples/scripts/train_qwen3_0_6b.sh
python src/lclr/evaluate_lca.py --model_ckpt <CRAFT_CKPT> --visualize pca
```

## TODO before release

- [ ] Upload LCLR artifacts (`outputs/lclr/*`) and CRAFT R²L checkpoints to
      the project's Hugging Face Hub and link them here.
- [ ] Replace the placeholder bibtex year if the camera-ready DOI is
      assigned.
