"""Convert a Transformers LLaVA dump (model_type=llava) to haotian-liu format."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch

HF_SNAP = Path(
    "/data0/lyx/token-pruning-for-efficient-MLLM/weights/hub/"
    "models--liuhaotian--llava-v1.5-7b/snapshots/"
    "6f798d78731e34b2fa643748a3c55cacc7afc5e2"
)
OUT = HF_SNAP.parent / "haotian-from-hf"
HUB = HF_SNAP.parents[1]  # models--liuhaotian--llava-v1.5-7b


def remap_key(key: str) -> str | None:
    if key.startswith("vision_tower."):
        return None
    if key.startswith("language_model.lm_head."):
        return key.replace("language_model.", "", 1)
    if key.startswith("language_model.model."):
        return key.replace("language_model.", "", 1)
    if key == "multi_modal_projector.linear_1.weight":
        return "model.mm_projector.0.weight"
    if key == "multi_modal_projector.linear_1.bias":
        return "model.mm_projector.0.bias"
    if key == "multi_modal_projector.linear_2.weight":
        return "model.mm_projector.2.weight"
    if key == "multi_modal_projector.linear_2.bias":
        return "model.mm_projector.2.bias"
    raise KeyError(f"unmapped key: {key}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hf_cfg = json.loads((HF_SNAP / "config.json").read_text())
    text = hf_cfg.get("text_config") or {}
    vision = hf_cfg.get("vision_config") or {}
    cfg = {
        "_name_or_path": text.get("_name_or_path", "lmsys/vicuna-7b-v1.5"),
        "architectures": ["LlavaLlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": int(text.get("max_position_embeddings", 4096)),
        "model_type": "llava_llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": float(text.get("rms_norm_eps", 1e-5)),
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.31.0",
        "use_cache": True,
        "vocab_size": int(text.get("vocab_size") or hf_cfg.get("vocab_size") or 32064),
        "mm_vision_tower": "openai/clip-vit-large-patch14-336",
        "mm_hidden_size": int(vision.get("hidden_size", 1024)),
        "mm_use_im_patch_token": False,
        "mm_use_im_start_end": False,
        "mm_vision_select_feature": "patch",
        "mm_vision_select_layer": -2,
        "mm_projector_type": "mlp2x_gelu",
        "image_aspect_ratio": "pad",
        "image_grid_pinpoints": None,
        "tokenizer_model_max_length": 2048,
        "tokenizer_padding_side": "right",
        "tune_mm_mlp_adapter": False,
        "freeze_mm_mlp_adapter": False,
        "use_mm_proj": True,
    }
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (OUT / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "transformers_version": "4.31.0"}, indent=2)
        + "\n"
    )

    copy_names = [
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    for name in copy_names:
        src = HF_SNAP / name
        if src.exists():
            shutil.copy2(src, OUT / name)
            print("copy", name, flush=True)

    shards = sorted(HF_SNAP.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors in {HF_SNAP}")

    projector: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    converted_files = []
    for i, shard in enumerate(shards, start=1):
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as fh:
            for key in fh.keys():
                new = remap_key(key)
                if new is None:
                    continue
                tensor = fh.get_tensor(key)
                if new.startswith("model.mm_projector."):
                    projector[new] = tensor
                tensors[new] = tensor
        out_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        dest = OUT / out_name
        print(f"write {out_name} keys={len(tensors)}", flush=True)
        save_file(tensors, str(dest), metadata={"format": "pt"})
        converted_files.append(out_name)
        for k in tensors:
            weight_map[k] = out_name

    index = {"metadata": {"format": "pt"}, "weight_map": weight_map}
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    torch.save(projector, OUT / "mm_projector.bin")
    print("mm_projector.bin keys", sorted(projector), flush=True)
    print("saved", OUT, flush=True)

    refs = HUB / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text("haotian-from-hf\n")
    print("pinned refs/main -> haotian-from-hf", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
