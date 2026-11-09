"""Re-run FastVID / VidCom2 keep on geometrically transformed Video-MME frames."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "scripts" / "repro"


def configure_offline() -> None:
    """Use the repo weight cache; do not hit huggingface.co."""
    os.environ.setdefault("HF_HOME", str(ROOT / "weights"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


configure_offline()

_MODEL = None
_MODEL_METHOD: str | None = None


def _ensure_repro_path() -> None:
    p = str(REPRO)
    if p not in sys.path:
        sys.path.insert(0, p)


def probe_mask_dir(out_dir: Path, method: str, geom: str) -> Path:
    return Path(out_dir) / "probe" / method / geom / "masks"


def _set_probe_dir(method: str, probe_root: Path) -> None:
    _ensure_repro_path()
    os.environ["REPRO_PROBE"] = "1"
    os.environ["REPRO_PROBE_METHOD"] = method
    os.environ["REPRO_PROBE_DIR"] = str(probe_root)
    import install as inst
    import session as sessmod

    if sessmod.SESSION is None:
        inst.install_session(method)
    else:
        sessmod.SESSION.root = probe_root
        (probe_root / "masks").mkdir(parents=True, exist_ok=True)
        (probe_root / "sidecars" / "failures").mkdir(parents=True, exist_ok=True)
        sessmod.SESSION.journal = probe_root / "sidecars" / "journal.jsonl"


@lru_cache(maxsize=1)
def _videomme_rows() -> dict[tuple[str, str], dict]:
    _ensure_repro_path()
    from local_videomme import parquet_path

    path = parquet_path()
    if path is None:
        raise SystemExit("Video-MME parquet not found under HF_HOME")
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    out: dict[tuple[str, str], dict] = {}
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = len(next(iter(cols.values())))
    for i in range(n):
        row = {k: cols[k][i] for k in cols}
        vid = str(row.get("videoID") or row.get("video_id") or "")
        qid = str(row.get("question_id") or "")
        out[(vid, qid)] = row
    return out


def videomme_doc(key: str) -> dict:
    stem = key.replace(".npz", "")
    if "__" not in stem:
        raise SystemExit(f"videomme key must be videoID__question_id, got {key!r}")
    vid, qid = stem.split("__", 1)
    row = _videomme_rows().get((vid, qid))
    if row is None:
        raise SystemExit(f"no Video-MME row for {vid} / {qid}")
    return row


def videomme_prompt(doc: dict) -> str:
    option_prompt = (
        "Select the best answer to the following multiple-choice question based on the video "
        "and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    )
    options = doc.get("options") or []
    if isinstance(options, np.ndarray):
        options = options.tolist()
    option = "\n".join(str(opt) for opt in options)
    question = str(doc.get("question") or "") + "\n" + option
    return option_prompt + "\n" + question + "\nThe best answer is:"


def _llava_kwargs(method: str, pretrained: Path, fastvid_ratio: float) -> dict:
    kw = {
        "pretrained": str(pretrained),
        "conv_template": "qwen_1_5",
        "max_frames_num": 64,
        "force_sample": True,
        "mm_spatial_pool_mode": "average",
        "attn_implementation": "sdpa",
        "device_map": "cuda:0",
    }
    if method == "fastvid":
        kw.update(
            fastvid_retention_ratio=float(fastvid_ratio),
            fastvid_DySeg_c=8,
            fastvid_DySeg_tau=0.9,
            fastvid_STPrune_d=0.4,
            fastvid_DTM_p=4,
            fastvid_DTM_beta=0.6,
        )
    return kw


def load_method_model(method: str, *, fastvid_ratio: float, r_ratio: float):
    global _MODEL, _MODEL_METHOD
    configure_offline()
    if _MODEL is not None and _MODEL_METHOD != method:
        raise SystemExit(
            f"this process already loaded {_MODEL_METHOD}; run VidCom2 in the VidCom2 env "
            "and FastVID in the fastvid env (separate --methods invocations)"
        )
    if _MODEL is not None:
        return _MODEL

    from analyse.lib.paths import SIGLIP_SNAPSHOT, read_pretrained

    pretrained = read_pretrained(method)
    if pretrained is None:
        raise SystemExit(f"missing pretrained_path.txt for {method}")
    if method == "fastvid":
        llava_root = ROOT / "ThirdParty" / "FastVID" / "fastvid_llavavideo"
        for extra in (llava_root, llava_root / "lmms-eval"):
            s = str(extra)
            if s not in sys.path:
                sys.path.insert(0, s)
        os.chdir(llava_root)
    else:
        vc = ROOT / "ThirdParty" / "VidCom2"
        for extra in (vc, vc / "lmms-eval"):
            s = str(extra)
            if s not in sys.path:
                sys.path.insert(0, s)
        os.environ["COMPRESSOR"] = "vidcom2"
        os.environ["R_RATIO"] = str(r_ratio)
        os.chdir(vc)

    next_root = Path.cwd() / "LLaVA-NeXT"
    if next_root.is_dir() and str(next_root) not in sys.path:
        sys.path.insert(0, str(next_root))
    if not SIGLIP_SNAPSHOT.is_dir():
        raise SystemExit(
            f"missing local SigLIP snapshot {SIGLIP_SNAPSHOT}; "
            "set HF_HOME to the repo weights/ directory"
        )
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

    if not getattr(SigLipVisionTower, "_repro_local_siglip", False):
        _orig_siglip_load = SigLipVisionTower.load_model

        def _load_siglip_local(self, device_map=None):
            self.vision_tower_name = str(SIGLIP_SNAPSHOT)
            return _orig_siglip_load(self, device_map=device_map)

        SigLipVisionTower.load_model = _load_siglip_local
        SigLipVisionTower._repro_local_siglip = True

    from lmms_eval.models.llava_vid import LlavaVid

    print(f"loading llava_vid ({method}) from {pretrained} (HF_HOME={os.environ.get('HF_HOME')})")
    _MODEL = LlavaVid(**_llava_kwargs(method, pretrained, fastvid_ratio))
    _MODEL_METHOD = method
    return _MODEL


def generate_keep(
    method: str,
    key: str,
    frames64: list[np.ndarray],
    probe_root: Path,
    *,
    fastvid_ratio: float,
    r_ratio: float,
    force: bool,
) -> Path:
    from context import clear_sample, set_sample

    from analyse.lib.paths import normalize_key

    probe_root = Path(probe_root)
    probe_root.mkdir(parents=True, exist_ok=True)
    dest = probe_root / "masks" / f"{normalize_key(key)}.npz"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"skip infer {method} {dest.name} (exists)")
        return dest

    _set_probe_dir(method, probe_root)
    wrapper = load_method_model(method, fastvid_ratio=fastvid_ratio, r_ratio=r_ratio)
    doc = videomme_doc(key)
    vid = str(doc.get("videoID") or doc.get("video_id"))
    qid = str(doc.get("question_id"))
    prompt = videomme_prompt(doc)
    video = np.stack([np.ascontiguousarray(fr) for fr in frames64], axis=0)
    device = wrapper.device
    pixels = wrapper._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device)
    if getattr(wrapper, "torch_dtype", "float16") == "bfloat16":
        pixels = pixels.bfloat16()
    else:
        pixels = pixels.half()

    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates[wrapper.conv_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), wrapper.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    pad = wrapper.tokenizer.pad_token_id or wrapper.tokenizer.eos_token_id
    attn = input_ids.ne(pad).long()
    stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop], wrapper.tokenizer, input_ids)

    set_sample(video_id=vid, question_id=qid)
    try:
        with torch.inference_mode():
            wrapper.model.generate(
                inputs=input_ids,
                images=[pixels],
                attention_mask=attn,
                modalities="video",
                use_cache=True,
                stopping_criteria=[stopping],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=16,
            )
    finally:
        clear_sample()
    if not dest.exists():
        raise RuntimeError(f"probe did not write {dest}; check REPRO_PROBE and PYTHONPATH=scripts/repro")
    return dest
