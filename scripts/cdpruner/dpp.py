"""Conditional DPP selection used by CDPruner.

Same kernel and Fast MAP loop as ThirdParty/CDPruner/llava/model/llava_arch.py
encode_images. features are post-projector tokens; image_embeds and text_embeds
live in the vision-text aligned space (CLIP for LLaVA-1.5, SigLIP for LLaVA-Video).
"""
from __future__ import annotations

import torch


def select_indices(
    features: torch.Tensor,
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    keep: int,
) -> torch.Tensor:
    """Return sorted indices into the N spatial tokens.

    features: (N, D) post-projector, used for cosine similarity.
    image_embeds: (N, C) vision-text space.
    text_embeds: (M, C) instruction embedding segments.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be (N, D), got {tuple(features.shape)}")
    n = features.shape[0]
    keep = int(keep)
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    if keep >= n:
        return torch.arange(n, device=features.device)

    device = features.device
    image_norm = features.float()
    image_norm = image_norm / image_norm.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    similarity = torch.matmul(image_norm, image_norm.transpose(0, 1))

    img = image_embeds.float()
    img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    txt = text_embeds.float()
    txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    relevance = torch.matmul(img, txt.transpose(0, 1)).mean(dim=-1)
    relevance = (-relevance)
    span = relevance.max() - relevance.min()
    relevance = (relevance - relevance.min() + 1e-6) / span.clamp_min(1e-6)

    kernel = relevance.unsqueeze(1) * similarity * relevance.unsqueeze(0)

    cis = torch.zeros((keep, n), device=device, dtype=kernel.dtype)
    di2s = torch.diagonal(kernel).clone()
    select_idx = torch.empty(keep, dtype=torch.long, device=device)
    for i in range(keep):
        j = int(torch.argmax(di2s).item())
        select_idx[i] = j
        if i == 0:
            eis = kernel[j] / torch.sqrt(di2s[j]).clamp_min(1e-6)
        else:
            eis = (kernel[j] - torch.mv(cis[:i].transpose(0, 1), cis[:i, j])) / torch.sqrt(di2s[j]).clamp_min(1e-6)
        cis[i] = eis
        di2s = di2s - torch.square(eis)
        di2s[j] = -float("inf")
    return torch.sort(select_idx).values
