"""LLaVA model_vqa_loader + VisionZip retain-64 (dominant=54, contextual=10).

Same CLI as llava.eval.model_vqa_loader, plus optional REPRO_PROBE sample_id
from each question_id.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _prefer_haotian_llava() -> None:
    """Keep haotian-liu/LLaVA ahead of VidCom2/FastVID forks on sys.path."""
    drop_substrings = (
        "/ThirdParty/VidCom2",
        "/ThirdParty/FastVID",
        "/fastvid_llavavideo",
        "/fastvid_llavaonevision",
        "/fastvid_qwen25vl",
    )
    cleaned = []
    for p in sys.path:
        if any(s in p.replace("\\", "/") for s in drop_substrings):
            continue
        cleaned.append(p)
    sys.path[:] = cleaned
    for path in (ROOT / "ThirdParty" / "VisionZip", ROOT / "ThirdParty" / "LLaVA"):
        s = str(path)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


_prefer_haotian_llava()

import shortuuid
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from visionzip import visionzip


def _assert_haotian_llava() -> None:
    import llava

    path = llava.__file__.replace("\\", "/")
    if "/ThirdParty/LLaVA/llava/" not in path:
        raise RuntimeError(
            f"expected haotian-liu LLaVA, got {llava.__file__}. "
            "Unset PYTHONPATH leftovers from VidCom2/FastVID before running."
        )


def split_list(lst, n):
    import math

    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    return torch.stack(input_ids, 0), torch.stack(image_tensors, 0), image_sizes


def _maybe_point_clip(clip_snap: Path | None) -> None:
    if clip_snap is None:
        return
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    _orig = CLIPVisionTower.__init__

    def _init(self, vision_tower, args, delay_load=False):
        return _orig(self, str(clip_snap), args, delay_load=delay_load)

    CLIPVisionTower.__init__ = _init


def eval_model(args):
    if os.environ.get("REPRO_PROBE") == "1":
        sys.path.insert(0, str(ROOT / "scripts" / "repro"))
        import install as repro_install

        repro_install.install_session("visionzip")

    _prefer_haotian_llava()
    _assert_haotian_llava()
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    # Snapshot folder names like haotian-from-hf break LLaVA builders; pin the known name.
    model_name = args.model_name or get_model_name_from_path(model_path)
    if "llava" not in model_name.lower():
        model_name = "llava-v1.5-7b"
    clip = None
    hf = Path(os.environ.get("HF_HOME", ROOT / "weights"))
    clip_root = hf / "hub" / "models--openai--clip-vit-large-patch14-336" / "snapshots"
    if clip_root.exists():
        for cand in clip_root.iterdir():
            if (cand / "config.json").exists():
                clip = cand
                break
    _maybe_point_clip(clip)

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise RuntimeError("model has no vision_tower after load_pretrained_model")
    if not getattr(vision_tower, "is_loaded", False):
        vision_tower.load_model()
    device = next(model.parameters()).device
    vision_tower.to(device=device, dtype=torch.float16)
    if not hasattr(vision_tower, "vision_tower"):
        raise RuntimeError(
            "CLIPVisionTower.vision_tower missing after load_model(); "
            f"llava={__import__('llava').__file__}"
        )
    model = visionzip(model, dominant=args.dominant, contextual=args.contextual)
    if os.environ.get("REPRO_PROBE") == "1":
        import install as repro_install

        repro_install._maybe_patch()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)
    ans_file = open(answers_file, "w")

    conv_mode = args.conv_mode
    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in conv_mode:
        conv_mode = conv_mode + "_mmtag"

    loader = DataLoader(
        CustomDataset(questions, args.image_folder, tokenizer, image_processor, model.config, conv_mode),
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(loader, questions), total=len(questions)):
        idx = line["question_id"]
        if os.environ.get("REPRO_PROBE") == "1":
            from context import set_sample

            set_sample(sample_id=str(idx))

        input_ids = input_ids.to(device="cuda", non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                image_sizes=image_sizes,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": line["text"],
                    "text": outputs,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
    ans_file.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None, help="override get_model_name_from_path")
    p.add_argument("--image-folder", type=str, required=True)
    p.add_argument("--question-file", type=str, required=True)
    p.add_argument("--answers-file", type=str, required=True)
    p.add_argument("--conv-mode", type=str, default="vicuna_v1")
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--chunk-idx", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dominant", type=int, default=54)
    p.add_argument("--contextual", type=int, default=10)
    args = p.parse_args()
    eval_model(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
