"""Install hard top-K pruning on LLaVA-1.5 without editing ThirdParty.

VLA_ACL=1 loads the scorer in VLA_ACL_SCORER and keeps VLA_ACL_TOKEN patches.
Kept patches stay in score order and receive new contiguous positions. This
torch SDPA build rejects the original grid positions.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from scorer import PatchScorer, hard_top_k_indices

_INSTALLED = False
_SCORER: PatchScorer | None = None


def _enabled() -> bool:
    return os.environ.get("VLA_ACL") == "1"


def _token_count() -> int:
    return int(os.environ.get("VLA_ACL_TOKEN", "128"))


def _load_scorer(device: torch.device, dtype: torch.dtype) -> PatchScorer:
    global _SCORER
    if _SCORER is not None:
        return _SCORER
    path = Path(os.environ["VLA_ACL_SCORER"])
    blob = torch.load(path, map_location="cpu")
    scorer = PatchScorer(
        vision_dim=int(blob["vision_dim"]),
        hidden_dim=int(blob["hidden_dim"]),
        num_layers=int(blob["num_layers"]),
        num_heads=int(blob["num_heads"]),
    )
    scorer.load_state_dict(blob["state_dict"])
    if int(blob["k"]) != _token_count():
        raise RuntimeError(
            f"scorer was trained with k={blob['k']} but VLA_ACL_TOKEN={_token_count()}"
        )
    scorer.eval()
    for param in scorer.parameters():
        param.requires_grad_(False)
    _SCORER = scorer.to(device=device, dtype=torch.float32)
    return _SCORER


def install() -> None:
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return
    import llava.model.llava_arch as arch

    if not hasattr(arch.LlavaMetaForCausalLM, "encode_images"):
        return
    original_encode = arch.LlavaMetaForCausalLM.encode_images

    def encode_images(self, images):
        tower = self.get_model().get_vision_tower()
        patches = tower(images)
        scorer = _load_scorer(patches.device, patches.dtype)
        with torch.no_grad():
            keep = hard_top_k_indices(scorer(patches.float()), _token_count())
            kept = torch.gather(patches, 1, keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        return self.get_model().mm_projector(kept)

    arch.LlavaMetaForCausalLM.encode_images = encode_images
    _INSTALLED = True
    # Silence unused binding if a caller still holds the original name.
    del original_encode
