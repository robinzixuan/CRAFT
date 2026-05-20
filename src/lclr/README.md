# LCLR — Latent Contrastive Learning for Reasoning (Paper §4.1)

This package trains the **projection head** `f_ω` and the **safety head** `g_ψ`
that map a hidden state of a reasoning trace into a 512-dim safety latent
space and into 3-way probabilities `{unsafe, rethink, safe}`. Class
prototypes `μ_safe`, `μ_unsafe`, `μ_rethink` are saved alongside the heads
and are consumed by `src/r2l/` as part of the latent-semantic reward.

## What gets produced

After training, `./outputs/lclr/` contains:

| File | Shape | Source equation |
| --- | --- | --- |
| `projection_head.pt` | `f_ω` weights | eq. for `z = f_ω(h)` |
| `safety_head.pt` | `g_ψ` weights | eq. for `p_z = g_ψ(z)` |
| `mu_safe.npy` | `[512]` | prototype for SAFE class |
| `mu_unsafe.npy` | `[512]` | prototype for UNSAFE class |
| `mu_rethink.npy` | `[512]` | prototype for RETHINK class |

## Quickstart

```bash
# Requires 1× A100 (40 GB) for the default Qwen3-4B-Thinking backbone.
pip install -r requirements.txt

cd src/lclr && python train_lca.py \
    --data_path chuhac/R2D-R1 \
    --model_name Qwen/Qwen3-4B-Thinking-2507 \
    --output_dir ../../outputs/lclr \
    --epochs 3 \
    --batch_size 16
```

Optional: enable Weights & Biases tracking with `--wandb` (defaults to off).

## Files

| File | Role |
| --- | --- |
| `lca.py` | `ProjectionHead`, `SafetyHead`, contrastive losses, training loop |
| `train_lca.py` | Entry point: data loading, train/test split, checkpointing |
| `evaluate_lca.py` | Evaluation script (3-way classification + prototype distances) |
| `embedding.py` | Hidden-state extraction utility |

## Loss components

The training objective from the paper:

```
L_LCLR = L_proto + λ_inst * L_inst + λ_cal * L_cal
```

- `L_proto`: structured geometric alignment (margin-based triplet with a rethink-anchoring term)
- `L_inst`: SimCLR-style instance invariance (InfoNCE over augmented views)
- `L_cal`: safety calibration (BCE + KL distillation from a textual verifier)

See `lca.py:contractive_loss` and `lca.py:info_nce` for implementations.
