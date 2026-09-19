"""Default run dirs, media roots, and dataset/method compatibility."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METHOD_MASK_DIRS = {
    "fastvid": ROOT / "runs" / "fastvid" / "full_videomme_probe" / "masks",
    "vidcom2": ROOT / "runs" / "vidcom2" / "full_videomme_probe" / "masks",
    "visionzip": ROOT / "runs" / "visionzip" / "full_llava15_probe" / "masks",
}

METHOD_PRETRAINED = {
    "fastvid": ROOT / "runs" / "fastvid" / "full_videomme" / "pretrained_path.txt",
    "vidcom2": ROOT / "runs" / "vidcom2" / "full_videomme" / "pretrained_path.txt",
    "visionzip": ROOT / "runs" / "visionzip" / "full_llava15_retain64" / "pretrained_path.txt",
}

DATASET_METHODS = {
    "videomme": frozenset({"fastvid", "vidcom2"}),
    "gqa": frozenset({"visionzip"}),
    "mme": frozenset({"visionzip"}),
    "pope": frozenset({"visionzip"}),
}

VISION_ENCODER = {
    "fastvid": "siglip",
    "vidcom2": "siglip",
    "visionzip": "clip",
}

SIGLIP_SNAPSHOT = (
    ROOT
    / "weights"
    / "hub"
    / "models--google--siglip-so400m-patch14-384"
    / "snapshots"
    / "ff792b42c7709e0444bffb50bd2faf66216b9dc8"
)
CLIP_SNAPSHOT = (
    ROOT
    / "weights"
    / "hub"
    / "models--openai--clip-vit-large-patch14-336"
    / "snapshots"
    / "96971839baa62a3493f8f1465dbc0a59d1b135ca"
)

N_SHOW_FRAMES = 8
N_MODEL_FRAMES = 64


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", str(ROOT / "weights")))


def videomme_dir() -> Path:
    return hf_home() / "videomme" / "data"


def llava_eval_root() -> Path:
    return Path(
        os.environ.get(
            "LLAVA_EVAL_ROOT",
            str(ROOT / "ThirdParty" / "LLaVA" / "playground" / "data" / "eval"),
        )
    )


def parse_methods(text: str) -> list[str]:
    methods = [m.strip().lower() for m in text.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_MASK_DIRS]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown}; expected {sorted(METHOD_MASK_DIRS)}")
    if not methods:
        raise SystemExit("need at least one --methods entry")
    return methods


def parse_mask_dirs(text: str | None) -> dict[str, Path] | None:
    """Parse 'fastvid=/path,vidcom2=/path' overrides for probe NPZ directories."""
    if not text:
        return None
    out: dict[str, Path] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--mask-dirs expected method=path, got {part!r}")
        method, path = part.split("=", 1)
        method = method.strip().lower()
        if method not in METHOD_MASK_DIRS:
            raise SystemExit(f"unknown method in --mask-dirs: {method}")
        out[method] = Path(path.strip()).expanduser().resolve()
    return out


def merged_mask_dirs(overrides: dict[str, Path] | None) -> dict[str, Path]:
    merged = dict(METHOD_MASK_DIRS)
    if overrides:
        merged.update(overrides)
    return merged


def check_dataset_methods(dataset: str, methods: list[str]) -> None:
    ds = dataset.strip().lower()
    if ds not in DATASET_METHODS:
        raise SystemExit(f"unknown dataset {dataset!r}; expected {sorted(DATASET_METHODS)}")
    allowed = DATASET_METHODS[ds]
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise SystemExit(
            f"methods {bad} cannot be compared on dataset={ds} "
            f"(allowed: {sorted(allowed)}). "
            "FastVID/VidCom2 are Video-MME; VisionZip is GQA/MME/POPE."
        )


def mask_path(method: str, key: str, mask_dir: Path | None = None) -> Path:
    directory = mask_dir or METHOD_MASK_DIRS[method]
    name = key_to_filename(key)
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"missing mask {path}")
    return path


def key_to_filename(key: str) -> str:
    text = key.strip()
    if text.endswith(".npz"):
        text = text[: -len(".npz")]
    text = text.replace("/", "_")
    if "__" not in text and "_" in text and text.count("_") == 1:
        # allow videoID_qid as well as videoID__qid
        pass
    return text + ".npz"


def normalize_key(key: str) -> str:
    text = key.strip()
    if text.endswith(".npz"):
        text = text[: -len(".npz")]
    return text.replace("/", "_")


def read_pretrained(method: str) -> Path | None:
    marker = METHOD_PRETRAINED.get(method)
    if marker is None or not marker.exists():
        return None
    line = marker.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return Path(line) if line else None
