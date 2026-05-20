# R²L — Reinforcement over Reasoning Latents (Paper §4.2)

R²L is a red-teaming GRPO training framework that aligns latent reasoning
trajectories with the final response. It is a fork of
[EasyR1](https://github.com/hiyouga/EasyR1) / [veRL](https://github.com/volcengine/verl);
the modifications live in:

- `verl/workers/fsdp_workers.py` — adds `compute_latent_features` so the actor model exposes mean-pooled hidden states without spinning up a second LLM
- `verl/workers/actor/dp_actor.py` — wires `compute_latent_features` into the FSDP actor
- `verl/workers/reward/function.py` — passes the reward function the latents alongside the textual output
- `verl/trainer/ray_trainer.py` — synchronises the latent-feature pass with the rollout
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py` — saves model-only checkpoints to fit a 4×80 GB budget

## Hardware

CRAFT was trained on 4×A100 80 GB. The reward function runs CPU-side (no
extra GPU) because the latent features are computed inside the FSDP actor.

## Quickstart

```bash
# Pre-condition: src/lclr/train_lca.py has been run and ./outputs/lclr/
# contains the five LCLR artifacts.

# Activate the conda env with vllm 0.10.1, then:
bash examples/scripts/train_qwen3_4b_thinking.sh

# For DeepSeek-R1-Distill-Llama-8B:
bash examples/scripts/train_deepseek_r1_8b.sh

# Ablation (no L_cons, no R_ls, no R_txt — pick via the CLI overrides
# the script enables):
bash examples/scripts/train_ablation.sh
```

Checkpoints are written to `./checkpoints/<experiment_name>/`. Merge to a
Hugging Face-loadable format via:

```bash
python scripts/model_merger.py --local_dir ./checkpoints/qwen3_4b_thinking_craft/global_step_<N>/actor
```

## Configs

| File | Used for |
| --- | --- |
| `examples/configs/config_qwen3_4b_thinking.yaml` | Main 4-GPU Qwen3-4B-Thinking run (paper main results) |
| `examples/configs/config_qwen3_4b_thinking_2gpu.yaml` | 2-GPU variant (smaller batch) |
| `examples/configs/config_qwen3_4b_thinking_4gpu.yaml` | Verbose 4-GPU variant |
| `examples/configs/config_deepseek_r1_distill_llama_8b.yaml` | DeepSeek-R1-Distill-Llama-8B run |
| `examples/configs/config_qwen3_0_6b.yaml` | Tiny smoke test on Qwen3-0.6B (used for Fig. 5) |
| `examples/configs/config_ablation.yaml` | Ablation variants (paper Table 3) |

## Reward function

`examples/reward_function/reasoning_trace.py` implements the three reward
components from the paper:

```
R_total = w_lat * R_ls + w_txt * R_txt + w_cons * R_cons
```

- `R_ls` — latent semantic reward (cosine to the three prototypes)
- `R_txt` — textual safety reward via StrongReject (`R_txt = 2 * P(S|y) - 1`)
- `R_cons` — latent–textual consistency reward (`1 - |p_z - P(S|y)|`)

The corresponding ablation file is `reasoning_trace_ablation.py` (used for
Table 3 rows that drop one of the three terms).

## Upstream attribution

The `verl/` subtree is a fork of EasyR1 (Apache-2.0). All upstream files
that we did not modify retain their original headers. Our modifications are
limited to the five files listed at the top of this README. See `../../NOTICE`.
