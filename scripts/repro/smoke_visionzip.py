"""Single-image VisionZip retain-64 smoke (LLaVA-1.5-7B)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "weights"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
sys.path.insert(0, str(ROOT / "ThirdParty" / "VisionZip"))

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from visionzip import visionzip


def _hub_snapshot(repo_id: str) -> Path | None:
    hf = Path(os.environ.get("HF_HOME", ROOT / "weights"))
    root = hf / "hub" / f"models--{repo_id.replace('/', '--')}"
    snaps = root / "snapshots"
    if not snaps.exists():
        return None
    ref = root / "refs" / "main"
    cands = []
    if ref.exists():
        cands.append(snaps / ref.read_text().strip())
    cands.extend(p for p in snaps.iterdir() if p.is_dir())
    for cand in cands:
        if (cand / "config.json").exists():
            return cand
    return None


def main() -> int:
    probe = os.environ.get("REPRO_PROBE") == "1"
    if probe:
        os.environ.setdefault("REPRO_PROBE_METHOD", "visionzip")
        os.environ.setdefault("REPRO_PROBE_DIR", str(ROOT / "runs" / "visionzip" / "smoke_probe"))
        sys.path.insert(0, str(ROOT / "scripts" / "repro"))
        import install as repro_install
        from context import set_sample

        repro_install.install_session("visionzip")
        set_sample(sample_id="smoke-image")

    disable_torch_init()
    model_path = os.environ.get("VISIONZIP_MODEL") or str(
        _hub_snapshot("liuhaotian/llava-v1.5-7b") or "liuhaotian/llava-v1.5-7b"
    )
    clip = _hub_snapshot("openai/clip-vit-large-patch14-336")
    if clip is not None:
        from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

        _orig_init = CLIPVisionTower.__init__

        def _init(self, vision_tower, args, delay_load=False):
            return _orig_init(self, str(clip), args, delay_load=delay_load)

        CLIPVisionTower.__init__ = _init
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava-v1.5-7b"
    )
    model = visionzip(model, dominant=54, contextual=10)
    if probe:
        import install as repro_install

        repro_install._maybe_patch()

    image = Image.new("RGB", (336, 336), color=(40, 80, 160))
    image_tensor = process_images([image], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    conv = conv_templates["llava_v1"].copy()
    qs = DEFAULT_IMAGE_TOKEN + "\nDescribe the image in one sentence."
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    n_out = None
    # VisionZip calls CLIPVisionTower.forward() directly, skipping Module hooks.
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    _orig_fwd = CLIPVisionTower.forward

    def _fwd(self, images):
        nonlocal n_out
        out = _orig_fwd(self, images)
        hidden = out[0] if isinstance(out, tuple) else out
        n_out = int(hidden.shape[1])
        return out

    CLIPVisionTower.forward = _fwd
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=32,
                use_cache=True,
            )
    finally:
        CLIPVisionTower.forward = _orig_fwd
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("VISIONZIP_TEXT", text)
    print("VISIONZIP_N_OUT", n_out)
    if not text:
        raise SystemExit("VisionZip produced empty text")
    if n_out is None:
        raise SystemExit("VisionZip tower hook did not fire")
    if not (60 <= n_out <= 70):
        raise SystemExit(f"expected ~64 visual tokens, got {n_out}")
    print("VISIONZIP_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
