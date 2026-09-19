"""Hook SigLIP / CLIP encoder layers; dump spatial energy maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from analyse.paths import CLIP_SNAPSHOT, SIGLIP_SNAPSHOT, VISION_ENCODER


def select_layer_indices(n_layers: int, n_pick: int = 5) -> list[int]:
    """First, last, and uniformly sampled intermediate layers (target 5)."""
    n = int(n_layers)
    if n <= 0:
        return []
    if n <= n_pick:
        return list(range(n))
    raw = np.linspace(0, n - 1, n_pick)
    idx = [int(round(x)) for x in raw]
    idx[0] = 0
    idx[-1] = n - 1
    # unique, keep order
    seen = set()
    ordered = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    # fill if rounding collapsed
    mid = [j for j in range(1, n - 1) if j not in seen]
    k = 0
    while len(ordered) < n_pick and k < len(mid):
        ordered.append(mid[k])
        k += 1
    ordered = sorted(set(ordered))
    if ordered[0] != 0:
        ordered = [0] + ordered
    if ordered[-1] != n - 1:
        ordered.append(n - 1)
        ordered = sorted(set(ordered))
    if len(ordered) > n_pick:
        extra = ordered[1:-1]
        keep_mid = np.linspace(0, len(extra) - 1, n_pick - 2, dtype=int)
        ordered = [0] + [extra[i] for i in keep_mid] + [n - 1]
    return ordered


def tokens_to_energy(hidden: np.ndarray) -> np.ndarray:
    """(N, C) tokens → (H, W) mean squared activation. Drops CLS if leftover."""
    x = np.asarray(hidden, dtype=np.float32)
    if x.ndim == 3:
        x = x[0]
    n = x.shape[0]
    side = int(n**0.5)
    if side * side == n:
        patches = x
    elif side * side == n - 1:
        patches = x[1:]
        side = int(patches.shape[0] ** 0.5)
    else:
        # pad/crop to nearest square
        side = int(n**0.5)
        patches = x[: side * side]
    energy = (patches * patches).mean(axis=-1).reshape(side, side)
    return energy


def _find_encoder_layers(model):
    vm = model
    if hasattr(vm, "vision_model"):
        vm = vm.vision_model
    if hasattr(vm, "encoder") and hasattr(vm.encoder, "layers"):
        return vm.encoder.layers
    raise RuntimeError("cannot find vision encoder.layers on model")


def _processor_and_model(kind: str, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    snap = SIGLIP_SNAPSHOT if kind == "siglip" else CLIP_SNAPSHOT
    if not snap.exists():
        raise FileNotFoundError(f"vision snapshot missing: {snap}")
    processor = AutoImageProcessor.from_pretrained(str(snap), local_files_only=True)
    model = AutoModel.from_pretrained(str(snap), local_files_only=True)
    model.to(device)
    model.eval()
    return processor, model


def cache_dir(out_root: Path, method: str, key: str) -> Path:
    safe = key.replace("/", "_")
    return Path(out_root) / "cache" / method / safe


def layer_cache_complete(folder: Path, indices: list[int]) -> bool:
    return all((folder / f"layer_{i}.npy").exists() for i in indices)


def encode_layers(
    method: str,
    frames_rgb: list[np.ndarray],
    out_root: Path,
    key: str,
    n_pick: int = 5,
    device: str | None = None,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Return layer indices and maps {layer: (T, H, W)} float32. Uses disk cache."""
    import torch

    kind = VISION_ENCODER[method]
    folder = cache_dir(out_root, method, key)
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "layer_indices.json"
    if meta_path.exists():
        meta = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
        indices = [int(x) for x in meta.get("indices", [])]
        if indices and layer_cache_complete(folder, indices):
            maps = {i: np.load(folder / f"layer_{i}.npy") for i in indices}
            return indices, maps

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = _processor_and_model(kind, device)
    layers = _find_encoder_layers(model)
    indices = select_layer_indices(len(layers), n_pick)
    meta_path.write_text(
        __import__("json").dumps({"method": method, "kind": kind, "indices": indices, "n_layers": len(layers)})
        + "\n",
        encoding="utf-8",
    )
    if layer_cache_complete(folder, indices):
        maps = {i: np.load(folder / f"layer_{i}.npy") for i in indices}
        del model
        return indices, maps

    from PIL import Image

    captured: dict[int, list] = {i: [] for i in indices}
    handles = []
    for i in indices:

        def _hook(mod, inp, out, layer_i=i):
            h = out[0] if isinstance(out, (tuple, list)) else out
            captured[layer_i].append(h.detach().float().cpu().numpy())

        handles.append(layers[i].register_forward_hook(_hook))

    try:
        with torch.no_grad():
            for fr in frames_rgb:
                im = Image.fromarray(np.ascontiguousarray(fr, dtype=np.uint8))
                batch = processor(images=im, return_tensors="pt")
                pixel = batch["pixel_values"].to(device)
                if hasattr(model, "vision_model"):
                    model.vision_model(pixel_values=pixel)
                else:
                    model(pixel_values=pixel)
    finally:
        for h in handles:
            h.remove()
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    maps: dict[int, np.ndarray] = {}
    for i in indices:
        energies = [tokens_to_energy(t) for t in captured[i]]
        stacked = np.stack(energies, axis=0)
        np.save(folder / f"layer_{i}.npy", stacked)
        maps[i] = stacked
    return indices, maps
