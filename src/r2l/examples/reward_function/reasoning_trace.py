# safety_reward_easyR1.py
# EasyR1-style reward function: R = w1*R_latent + w2*R_text + w3*R_cons + w4*R_help - λ_KL D_KL
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Any
from transformers import PreTrainedTokenizer

# ------------- Config (tunable weights) -------------
W_LATENT = 0.5
W_TEXT   = 0.4
W_CONS   = 0.2
# W_HELP   = 0.3
# LAMBDA_KL = 0.02
ALPHA, BETA, GAMMA = 1.0, 1.0, 0.5

# Precomputed prototypes from contrastive learning
# MU_SAFE   = torch.from_numpy(np.load("./outputs/lclr/mu_safe.npy")).float()
# MU_UNSAFE = torch.from_numpy(np.load("./outputs/lclr/mu_unsafe.npy")).float()
# MU_RETHINK= torch.from_numpy(np.load("./outputs/lclr/mu_rethink.npy")).float()

# ------------- Global head cache (load once, ~2MB) -------------
_CACHED_PROJECTION_HEAD = None
_CACHED_SAFETY_HEAD = None

_CACHED_MU_SAFE = None
_CACHED_MU_UNSAFE = None
_CACHED_MU_RETHINK = None

def cosine(z, mu):
    return (F.normalize(z, dim=-1) * F.normalize(mu, dim=-1)).sum(-1)


class ProjectionHead(nn.Module):
    def __init__(self, d_in, d=512):
        super().__init__()
        self.fn1 = nn.Linear(d_in, 1024)
        self.gelu = nn.GELU()
        self.fn2 = nn.Linear(1024, d)

    def forward(self, x):
        x = self.fn1(x)
        x = self.gelu(x)
        x = self.fn2(x)
        # safe normalization
        return F.normalize(x, dim=-1, eps=1e-6)

class SafetyHead(nn.Module):
    def __init__(self, d, num_classes=3): 
        super().__init__()
        self.fc=nn.Linear(d, num_classes)  # 3 classes: safe, unsafe, rethink
    def forward(self, z): 
        return self.fc(z)  # logits for 3 classes




@torch.no_grad()
def latent_reward(z: torch.Tensor) -> torch.Tensor:
    global _CACHED_MU_SAFE, _CACHED_MU_UNSAFE, _CACHED_MU_RETHINK
    return (
        ALPHA * cosine(z, _CACHED_MU_SAFE.to(z.device))
        - BETA * cosine(z, _CACHED_MU_UNSAFE.to(z.device))
        + GAMMA * cosine(z, _CACHED_MU_RETHINK.to(z.device))
    )

@torch.no_grad()
def text_reward(p_safe_text: torch.Tensor) -> torch.Tensor:
    return 2 * p_safe_text - 1

@torch.no_grad()
def consistency_reward(p_latent: torch.Tensor, p_text: torch.Tensor) -> torch.Tensor:
    return 1 - (p_latent - p_text).abs()


def calculate_latent(h_mean, config):
    """Compute z and p_latent from precomputed h_mean.

    h_mean is computed by the actor model (FSDP) in the training loop,
    so we only need the lightweight ProjectionHead + SafetyHead here.
    No separate base model loading required.
    """
    global _CACHED_PROJECTION_HEAD, _CACHED_SAFETY_HEAD
    global _CACHED_MU_SAFE, _CACHED_MU_UNSAFE, _CACHED_MU_RETHINK

    if h_mean.dim() == 1:
        h_mean = h_mean.unsqueeze(0)

    # ---- Load tiny heads only once (cached, ~2MB) ----
    if _CACHED_PROJECTION_HEAD is None:
        proj_state = torch.load(config.projection_ckpt_path, map_location="cpu")
        head_state = torch.load(config.safety_ckpt_path, map_location="cpu")

        d_in = proj_state['fn1.weight'].shape[1]
        print(f"Detected projection head input dim: {d_in}")

        projection_head = ProjectionHead(d_in=d_in)
        safety_head = SafetyHead(d=config.projection_dim)

        projection_head.load_state_dict(proj_state)
        safety_head.load_state_dict(head_state)

        projection_head.eval().requires_grad_(False)
        safety_head.eval().requires_grad_(False)

        _CACHED_PROJECTION_HEAD = projection_head
        _CACHED_SAFETY_HEAD = safety_head
        print("Loaded pretrained ProjectionHead and SafetyHead (cached).")

    if _CACHED_MU_SAFE is None:
        import os
        base_dir = os.path.dirname(config.safety_ckpt_path)
        print(f"Loading prototype vectors from {base_dir}...")
        _CACHED_MU_SAFE = torch.from_numpy(np.load(os.path.join(base_dir, "mu_safe.npy"))).float()
        _CACHED_MU_UNSAFE = torch.from_numpy(np.load(os.path.join(base_dir, "mu_unsafe.npy"))).float()
        _CACHED_MU_RETHINK = torch.from_numpy(np.load(os.path.join(base_dir, "mu_rethink.npy"))).float()
        print("Prototypes loaded and cached.")

    projection_head = _CACHED_PROJECTION_HEAD
    safety_head = _CACHED_SAFETY_HEAD

    h_mean = h_mean.float().cpu()
    z = projection_head(h_mean)                   # [B, 512]
    p_latent = torch.sigmoid(safety_head(z))       # [B]

    return h_mean, z, p_latent


# ------------- EasyR1 entry point -------------
def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    批量计算 reward，一次性做模型前向推理，避免逐样本推理的性能开销。
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please call with reward_type=batch in EasyR1.")

    # Use precomputed h_mean from actor model (no separate model forward needed)
    all_h_mean = torch.stack([r["h_mean"] for r in reward_inputs])
    config = reward_inputs[0]["config"]

    h_mean, z, p_latent = calculate_latent(all_h_mean, config)

    results = []
    for i, r in enumerate(reward_inputs):
        p_text = torch.tensor(r.get("p_text", 0.0), dtype=torch.float32)

        # 取第 i 个样本的特征
        z_i = z[i].float()                   # [512]
        p_latent_i = p_latent[i].float()[0]  # 取 safe 类概率

        R_lat = latent_reward(z_i)
        R_con = consistency_reward(p_latent_i, p_text)
        R_txt = text_reward(p_text)

        R_total = (
            W_LATENT * R_lat
            + W_TEXT * R_txt
            + W_CONS * R_con
        )

        results.append({
            "overall": float(R_total.item()),
            "R_latent": float(R_lat.item()),
            "R_text": float(R_txt.item()),
            "R_cons": float(R_con.item()),
        })

    return results
