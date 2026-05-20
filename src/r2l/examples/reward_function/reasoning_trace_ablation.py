# safety_reward_easyR1.py
# EasyR1-style reward function: R = w1*R_latent + w2*R_text + w3*R_cons + w4*R_help - λ_KL D_KL
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Any
from transformers import PreTrainedTokenizer, AutoModel, AutoModelForCausalLM

# ------------- Config (tunable weights) -------------
W_LATENT = 0.5
W_TEXT   = 0.4
W_CONS   = 0.2
# W_HELP   = 0.3
# LAMBDA_KL = 0.02
ALPHA, BETA, GAMMA = 1.0, 1.0, 0.5

# Prototype paths default to ./outputs/lclr/ (see src/lclr/train_lca.py).
# Override via config: worker.reward.projection_ckpt_path / safety_ckpt_path
# (the directory containing these two .pt files is the same as mu_*.npy).

# ------------- Global model cache (load once) -------------
_CACHED_MODEL = None
_CACHED_PROJECTION_HEAD = None
_CACHED_SAFETY_HEAD = None
_CACHED_MODEL_PATH = None

_CACHED_MU_SAFE = None
_CACHED_MU_UNSAFE = None
_CACHED_MU_RETHINK = None

def cosine(z, mu):
    return (F.normalize(z, dim=-1) * F.normalize(mu, dim=-1)).sum(-1)


@torch.no_grad()
def latent_reward(z: torch.Tensor) -> torch.Tensor:
    global _CACHED_MU_SAFE, _CACHED_MU_UNSAFE, _CACHED_MU_RETHINK
    return (
        ALPHA * cosine(z, _CACHED_MU_SAFE.to(z.device))
        - BETA * cosine(z, _CACHED_MU_UNSAFE.to(z.device))
        + GAMMA * cosine(z, _CACHED_MU_RETHINK.to(z.device))
    )


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
def text_reward(p_safe_text: torch.Tensor) -> torch.Tensor:
    return 2 * p_safe_text - 1

@torch.no_grad()
def consistency_reward(p_latent: torch.Tensor, p_text: torch.Tensor) -> torch.Tensor:
    return 1 - (p_latent - p_text).abs()


def calculate_latent(input_ids, attention_mask, config):
    global _CACHED_MODEL, _CACHED_PROJECTION_HEAD, _CACHED_SAFETY_HEAD, _CACHED_MODEL_PATH
    global _CACHED_MU_SAFE, _CACHED_MU_UNSAFE, _CACHED_MU_RETHINK

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    elif attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    # ---- Load models only once (cached) ----
    if _CACHED_PROJECTION_HEAD is None:
        # Load checkpoint first, infer input dimension from weights
        proj_state = torch.load(config.projection_ckpt_path, map_location="cpu")
        head_state = torch.load(config.safety_ckpt_path, map_location="cpu")

        # Infer d_in from weights (compatible with different models)
        d_in = proj_state['fn1.weight'].shape[1]
        print(f"📐 Detected projection head input dim: {d_in}")
        
        projection_head = ProjectionHead(d_in=d_in)
        safety_head = SafetyHead(d=config.projection_dim)

        projection_head.load_state_dict(proj_state)
        safety_head.load_state_dict(head_state)

        projection_head.eval().requires_grad_(False)
        safety_head.eval().requires_grad_(False)

        _CACHED_PROJECTION_HEAD = projection_head
        _CACHED_SAFETY_HEAD = safety_head
        print("✅ Loaded pretrained ProjectionHead and SafetyHead (cached).")

    if _CACHED_MU_SAFE is None:
        base_dir = os.path.dirname(config.safety_ckpt_path)
        print(f"Loading prototype vectors from {base_dir}...")
        _CACHED_MU_SAFE = torch.from_numpy(np.load(os.path.join(base_dir, "mu_safe.npy"))).float()
        _CACHED_MU_UNSAFE = torch.from_numpy(np.load(os.path.join(base_dir, "mu_unsafe.npy"))).float()
        _CACHED_MU_RETHINK = torch.from_numpy(np.load(os.path.join(base_dir, "mu_rethink.npy"))).float()
        print("Prototypes loaded and cached.")

    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != config.model_path:
        print(f"🔄 Loading model: {config.model_path}...")
        _CACHED_MODEL = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        _CACHED_MODEL.eval()
        _CACHED_MODEL_PATH = config.model_path
        print(f"✅ Model loaded and cached.")

    model = _CACHED_MODEL
    projection_head = _CACHED_PROJECTION_HEAD
    safety_head = _CACHED_SAFETY_HEAD

    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device), output_hidden_states=True)
        h_last = outputs.hidden_states[-1].float()  # force FP32
        reason_mask = attention_mask.to(h_last.device).float()

        h_mean = (h_last * reason_mask.unsqueeze(-1)).sum(dim=1) / \
                reason_mask.sum(dim=1, keepdim=True)
        

    
    h_mean = h_mean.float().cpu()

    
    z = projection_head(h_mean)                   # [B, 512]
    
    p_latent = torch.sigmoid(safety_head(z))       # [B]

    return h_mean, z, p_latent


# ------------- EasyR1 entry point -------------
def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    Each item in reward_inputs should include:
        {
           response: str
        p_text: float
        input_ids: torch.Tensor
        attention_mask: torch.Tensor
        response_ids: torch.Tensor
        response_mask: torch.Tensor
        response_length: int
            }
    Returns: list of dicts with reward components and overall reward.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please call with reward_type=batch in EasyR1.")

    # Collect all inputs in batch and perform a single model forward pass
    all_input_ids = torch.stack([r["input_ids"] for r in reward_inputs])
    all_attention_mask = torch.stack([r["attention_mask"] for r in reward_inputs])
    config = reward_inputs[0]["config"]

    h_mean, z, p_latent = calculate_latent(all_input_ids, all_attention_mask, config)

    w_lat = getattr(config, "w_lat", W_LATENT)
    w_text = getattr(config, "w_text", W_TEXT)
    w_cons = getattr(config, "w_cons", W_CONS)

    results = []
    for i, r in enumerate(reward_inputs):
        p_text = torch.tensor(r.get("p_text", 0.0), dtype=torch.float32)

        # Extract features for the i-th sample
        z_i = z[i].float()                   # [512]
        p_latent_i = p_latent[i].float()[0]  # safe class probability

        R_lat = latent_reward(z_i)
        R_con = consistency_reward(p_latent_i, p_text)
        R_txt = text_reward(p_text)

        R_total = (
            w_lat * R_lat
            + w_text * R_txt
            + w_cons * R_con
        )

        results.append({
            "overall": float(R_total.item()),
            "R_latent": float(R_lat.item()),
            "R_text": float(R_txt.item()),
            "R_cons": float(R_con.item()),
        })

    return results
