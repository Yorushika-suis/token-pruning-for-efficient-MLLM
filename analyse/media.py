"""Load Video-MME frames (64 then 8) and LLaVA-eval images."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from analyse.paths import (
    N_MODEL_FRAMES,
    N_SHOW_FRAMES,
    llava_eval_root,
    videomme_dir,
)


def uniform_indices(n_total: int, n_pick: int) -> np.ndarray:
    n_pick = min(int(n_pick), int(n_total))
    if n_total <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.linspace(0, n_total - 1, n_pick, dtype=np.int64))


def videomme_video_id(key: str) -> str:
    stem = key.replace(".npz", "")
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def videomme_mp4(key: str) -> Path:
    vid = videomme_video_id(key)
    path = videomme_dir() / f"{vid}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Video-MME mp4 missing: {path}")
    return path


def load_videomme_frames(key: str, n_model: int = N_MODEL_FRAMES, n_show: int = N_SHOW_FRAMES) -> tuple[list[np.ndarray], np.ndarray]:
    """force_sample 64 frames (linspace over the mp4), then uniformly keep n_show of those."""
    path = videomme_mp4(key)
    all64 = _read_uniform_frames(path, n_model)
    show_idx = uniform_indices(len(all64), n_show)
    return [all64[int(i)] for i in show_idx], show_idx


def _read_uniform_frames(path: Path, n: int) -> list[np.ndarray]:
    try:
        import decord
        from decord import cpu

        vr = decord.VideoReader(str(path), ctx=cpu(0), num_threads=1)
        total = len(vr)
        idx = uniform_indices(total, n).tolist()
        batch = vr.get_batch(idx).asnumpy()
        return [np.ascontiguousarray(batch[i]) for i in range(batch.shape[0])]
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("cv2 cannot open")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames:
                raise RuntimeError("empty video")
            idx = uniform_indices(len(frames), n)
            return [frames[int(i)] for i in idx]
        idx = uniform_indices(total, n)
        out = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not out:
            raise RuntimeError("cv2 read failed")
        return out
    except Exception as exc:
        raise RuntimeError(
            f"cannot decode {path}: need decord or opencv-python ({exc})"
        ) from exc


def _open_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        pass
    try:
        import cv2

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("imread None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        raise RuntimeError(f"cannot read image {path}: install pillow or opencv ({exc})") from exc


@lru_cache(maxsize=8)
def _jsonl_index(path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return mapping
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            mapping[qid] = str(row["image"])
            mapping[qid.replace("/", "_")] = str(row["image"])
    return mapping


def load_eval_image(dataset: str, key: str) -> np.ndarray:
    ds = dataset.lower()
    stem = key.replace(".npz", "")
    root = llava_eval_root()
    if ds == "gqa":
        js = _jsonl_index(str(root / "gqa" / "llava_gqa_testdev_balanced.jsonl"))
        rel = js.get(stem, f"{stem}.jpg")
        path = root / "gqa" / "data" / "images" / rel
    elif ds == "mme":
        js = _jsonl_index(str(root / "MME" / "llava_mme.jsonl"))
        rel = js.get(stem, stem.replace("_", "/", 1) if "_" in stem else stem)
        # prefer original slash path from jsonl
        cand = [
            root / "MME" / "MME_Benchmark_release_version" / rel,
            root / "MME" / "MME_Benchmark_release_version" / stem.replace("_", "/", 1),
        ]
        path = next((c for c in cand if c.exists()), cand[0])
    elif ds == "pope":
        js = _jsonl_index(str(root / "pope" / "llava_pope_test.jsonl"))
        rel = js.get(stem) or js.get(stem.lstrip("0")) or ""
        if not rel:
            # numeric qid
            rel = js.get(str(int(stem))) if stem.isdigit() else ""
        if not rel:
            raise FileNotFoundError(f"POPE image for key={stem} not in jsonl")
        path = root / "pope" / "val2014" / rel
    else:
        raise ValueError(f"not an image dataset: {dataset}")
    if not path.exists():
        raise FileNotFoundError(f"eval image missing: {path}")
    return _open_rgb(path)


def load_show_frames(dataset: str, key: str, n_show: int = N_SHOW_FRAMES) -> tuple[list[np.ndarray], np.ndarray]:
    ds = dataset.lower()
    if ds == "videomme":
        return load_videomme_frames(key, n_show=n_show)
    img = load_eval_image(ds, key)
    return [img], np.array([0], dtype=np.int64)
