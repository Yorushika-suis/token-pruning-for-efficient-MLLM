"""VLA-ACL patch scorer, LapSum soft top-K, and black-image gating.

Copied in behavior from ThirdParty/VLA-ACL without importing prismatic.
Single-image LLaVA uses one global top-K over the 576 CLIP patches.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchScorer(nn.Module):
    def __init__(
        self,
        vision_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 5,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_proj = nn.Identity() if vision_dim == hidden_dim else nn.Linear(vision_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.logit_head = nn.Linear(hidden_dim, 1)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        return self.logit_head(self.encoder(self.input_proj(vision_tokens))).squeeze(-1)


def hard_top_k_indices(patch_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Sorted keep indices (B, K) over the full patch sequence."""
    if k > patch_logits.shape[-1]:
        raise ValueError(f"k={k} exceeds {patch_logits.shape[-1]} patches")
    return patch_logits.topk(k, dim=-1).indices.sort(dim=-1).values


def patches_per_frame_for(retention: float, spatial: int = 169) -> int:
    """Per-frame keep count. 0.10 -> 17 and 0.25 -> 42 on a 13x13 grid."""
    return max(1, int(round(spatial * retention)))


def hard_top_k_per_frame(patch_logits: torch.Tensor, k: int) -> torch.Tensor:
    """patch_logits (T, N) -> flat spatial indices into a T*N sequence."""
    if patch_logits.ndim != 2:
        raise ValueError("per-frame logits must be (frames, patches)")
    local = hard_top_k_indices(patch_logits, k)
    offsets = torch.arange(local.shape[0], device=local.device).unsqueeze(1) * patch_logits.shape[1]
    return (local + offsets).reshape(-1)


def _laplace_cdf(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, 0.5 * torch.exp(x), 1.0 - 0.5 * torch.exp(-x))


@torch.no_grad()
def _solve_lapsum_threshold(scores: torch.Tensor, k: float, alpha: float, num_iters: int = 64) -> torch.Tensor:
    lo = scores.min(dim=-1, keepdim=True).values - 10.0 * alpha
    hi = scores.max(dim=-1, keepdim=True).values + 10.0 * alpha
    for _ in range(num_iters):
        mid = 0.5 * (lo + hi)
        above_k = _laplace_cdf((scores - mid) / alpha).sum(dim=-1, keepdim=True) > k
        lo = torch.where(above_k, mid, lo)
        hi = torch.where(above_k, hi, mid)
    return 0.5 * (lo + hi)


class SoftTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, k: float, alpha: float) -> torch.Tensor:
        x = scores.float()
        threshold = _solve_lapsum_threshold(x, k, alpha)
        ctx.save_for_backward(x, threshold)
        ctx.alpha = alpha
        ctx.dtype = scores.dtype
        return _laplace_cdf((x - threshold) / alpha).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        pdf = torch.exp(-(x - threshold).abs() / ctx.alpha) / (2.0 * ctx.alpha)
        grad_output = grad_output.float()
        weighted_mean = (pdf * grad_output).sum(dim=-1, keepdim=True) / pdf.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return (pdf * (grad_output - weighted_mean)).to(ctx.dtype), None, None


def soft_top_k(scores: torch.Tensor, k: int, alpha: float) -> torch.Tensor:
    return SoftTopK.apply(scores, float(k), float(alpha))


def anneal_alpha(step: int, end_step: int, alpha_start: float, alpha_end: float) -> float:
    progress = min(step, end_step) / max(end_step, 1)
    return alpha_end + 0.5 * (alpha_start - alpha_end) * (1.0 + math.cos(math.pi * progress))


def gate_vision_tokens(
    vision_tokens: torch.Tensor, selection_scores: torch.Tensor, black_embeddings: torch.Tensor
) -> torch.Tensor:
    s = selection_scores.float().unsqueeze(-1)
    black_weight = (1.0 - s.detach() ** 2).sqrt()
    gated = s * vision_tokens.float() + black_weight * black_embeddings.float()
    return gated.to(vision_tokens.dtype)


def use_full_rope_cache(model: torch.nn.Module) -> None:
    """Index RoPE by absolute position, not by the shortened sequence length."""
    for module in model.modules():
        if not hasattr(module, "cos_cached") or getattr(module, "_vla_acl_rope", False):
            continue

        def forward(x, seq_len=None, _module=module):
            needed = int(seq_len or _module.max_position_embeddings)
            if needed > _module.max_seq_len_cached:
                _module._set_cos_sin_cache(seq_len=needed, device=x.device, dtype=x.dtype)
            return _module.cos_cached.to(dtype=x.dtype), _module.sin_cached.to(dtype=x.dtype)

        module.forward = forward
        module._vla_acl_rope = True


def preserving_position_ids(
    prefix_len: int,
    keep_indices: torch.Tensor,
    full_patches: int,
    suffix_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Positions of [prefix, kept patches, suffix] using the unpruned image span."""
    prefix = torch.arange(prefix_len, device=device)
    vision = prefix_len + keep_indices.to(device=device, dtype=torch.long)
    suffix = prefix_len + full_patches + torch.arange(suffix_len, device=device)
    return torch.cat((prefix, vision, suffix), dim=0)
