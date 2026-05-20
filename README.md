# CRAFT: Contrastive Reasoning Alignment

> Reinforcement Learning from Hidden Representations
> **ICML 2026** · [paper (arXiv:2603.17305)](https://arxiv.org/abs/2603.17305)

CRAFT is a red-teaming alignment framework that mitigates *superficial safety alignment* (SSA) in large reasoning models. Instead of operating at the output level, CRAFT (a) structures the latent space of reasoning traces via contrastive learning, and (b) applies GRPO with a latent-aware reward that aligns intermediate reasoning states with the final response.

![CRAFT pipeline](docs/figures/pipeline.png)

## Highlights (vs. base models)

| Metric | Improvement |
| --- | --- |
| Reasoning-trace safety | **+82.1 %** (avg) |
| Final-response safety | **+89.6 %** (avg) |
| Reasoning ability (math + code) | **+8.0 %** (avg) |

Full results across DeepSeek-R1-Distill-Llama-8B and Qwen3-4B-Thinking on JailbreakBench, StrongReject, AIME24, MATH-500, Minerva, and LiveCodeBench: see [`docs/reproducibility.md`](docs/reproducibility.md) and the paper.

## Repository layout

```
src/
  lclr/        Phase 1 — Latent Contrastive Learning for Reasoning (§4.1)
  r2l/         Phase 2 — Reinforcement over Reasoning Latents (§4.2, vLLM-based)
eval/          JailbreakBench, StrongReject, advanced-attack pointers
docs/          Reproducibility guide and figures
```

## Quickstart

### Setup

```bash
# Conda
conda create -n craft python=3.10 -y
conda activate craft
pip install -r requirements.txt
pip install -r src/lclr/requirements.txt
pip install -e src/r2l   # installs the modified veRL trainer

# Or Docker
docker build -t craft src/r2l
```

You will need a Hugging Face account (login via `huggingface-cli login`) to download the base models and the `chuhac/R2D-R1` dataset. GPT-based safety eval also requires `OPENAI_API_KEY`.

### Three commands to reproduce the pipeline

```bash
# 1. Phase 1: train Latent Contrastive Learning heads on R2D-R1 (~30 min on 1× A100)
cd src/lclr && python train_lca.py --data_path chuhac/R2D-R1 --model_name Qwen/Qwen3-4B-Thinking-2507 --output_dir ../../outputs/lclr

# 2. Phase 2: R²L training (4× A100 80GB, ~14h for Qwen3-4B-Thinking)
bash src/r2l/examples/scripts/train_qwen3_4b_thinking.sh

# 3. Evaluation: JailbreakBench + StrongReject
bash eval/jailbreakbench/jbb_qwen.sh
```

See [`docs/reproducibility.md`](docs/reproducibility.md) for the exact hardware, seeds, and per-table reproduction steps.

## Citing CRAFT

```bibtex
@inproceedings{luo2026craft,
  title   = {Contrastive Reasoning Alignment: Reinforcement Learning from Hidden Representations},
  author  = {Luo, Haozheng and Wang, Yimin and Yu, Jiahao and Wang, Binghui and Chen, Yan},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year    = {2026},
  series  = {PMLR},
  volume  = {306},
}
```

## License & attribution

Apache-2.0 (see [`LICENSE`](LICENSE)). CRAFT builds on:

- [EasyR1](https://github.com/hiyouga/EasyR1) / [veRL](https://github.com/volcengine/verl) (Apache-2.0) — modified GRPO trainer under `src/r2l/verl/`
See [`NOTICE`](NOTICE) for the full attribution list.

## Contact

Correspondence to Haozheng Luo (`hluo@u.northwestern.edu`).
