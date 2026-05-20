# Baselines

CRAFT is compared against six red-teaming-alignment baselines (paper §6, Table 1).

| Baseline | Location | Type |
| --- | --- | --- |
| SafeChain | [upstream](https://github.com/uw-nsl/SafeChain) | SFT |
| RealSafe | [upstream](https://github.com/Realsafe-Project/RealSafe-R1) | SFT |
| STAR | [upstream](https://github.com/cooper-lab/STAR-1) | SFT |
| SafeKey | [upstream](https://github.com/safekey-project/safekey) | SFT |
| IPO | `baselines/ipo/` (this repo) | Preference optimization |
| ReasoningShield | `baselines/reasoningshield/` (this repo) | Safety detector + SFT |

The four baselines without local code (SafeChain, RealSafe, STAR, SafeKey)
are run from their upstream repos using their default hyperparameters on
the same R2D-R1 training set. See `docs/reproducibility.md` for the exact
commit hashes we used.

`baselines/ipo/` and `baselines/reasoningshield/` are direct forks; their
individual READMEs and licenses are preserved.
