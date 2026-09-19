"""Auto-install local Video-MME loader; probes when REPRO_PROBE=1."""
from __future__ import annotations

import os

try:
    import local_videomme

    local_videomme.patch()
except Exception:
    # Never break conda activate / unrelated python when PYTHONPATH includes this dir.
    pass

if os.environ.get("REPRO_PROBE") == "1":
    try:
        import install as _repro_install

        _repro_install.install_session()
    except Exception:
        pass


def _patch_hub_model_name() -> None:
    """HF snapshot dirs end in a SHA; LLaVA builders key off names like LLaVA-Video-7B-Qwen2."""
    try:
        import llava.mm_utils as mm_utils
    except Exception:
        return
    orig = mm_utils.get_model_name_from_path
    if getattr(orig, "_repro_hub_name", False):
        return

    def wrapped(model_path):
        path = str(model_path).replace("\\", "/").strip("/")
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part.startswith("models--") and i + 1 < len(parts) and parts[i + 1] == "snapshots":
                return part.split("--")[-1]
        return orig(model_path)

    wrapped._repro_hub_name = True
    mm_utils.get_model_name_from_path = wrapped


_patch_hub_model_name()
