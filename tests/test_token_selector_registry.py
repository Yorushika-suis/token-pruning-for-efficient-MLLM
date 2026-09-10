import sys
from pathlib import Path

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import token_selectors as selectors  # noqa: E402


class MinimalContext:
    def __init__(self, distances):
        self._distances = distances

    def dist169(self):
        return self._distances


def test_flashvid_div_ablation_has_non_paper_registry_name():
    assert "div_divprune" not in selectors.REGISTRY
    assert selectors.REGISTRY["FlashVID-div-ablation@P1U169"] is selectors.sel_div


def test_renamed_registry_entry_preserves_selector_computation():
    generator = torch.Generator().manual_seed(7)
    points = torch.randn((2, selectors.NV, 5), generator=generator)
    distances = selectors.pairwise_cosine(points)
    context = MinimalContext(distances)

    direct = selectors.sel_div(context, 8)
    registered = selectors.REGISTRY["FlashVID-div-ablation@P1U169"](context, 8)
    np.testing.assert_array_equal(direct, registered)
    assert direct.shape == (2, 8)
