#!/usr/bin/env python3
"""Run LLaVA on reconstructed MT-GQA conversations.

The script keeps the official LLaVA generation path and intercepts only the
projector output returned by ``encode_images``.  Consequently ``base`` uses
the untouched 576 patch tokens, ``sample`` uses deterministic equidistant
indices, and ``metacompress`` applies a saved MetaCompress checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ThirdParty" / "LLaVA"))
sys.path.insert(0, str(ROOT / "ThirdParty" / "MetaCompress"))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from metacompress import MetaCompress  # noqa: E402


def equidistant_sample(features: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"expected [B,N,D] projector features, got {tuple(features.shape)}")
    if not 1 <= target_tokens <= features.shape[1]:
        raise ValueError(f"target_tokens must be in [1,{features.shape[1]}]")
    indices = torch.linspace(
        0, features.shape[1] - 1, steps=target_tokens,
        device=features.device, dtype=torch.float32,
    ).round().to(torch.long)
    if indices.unique().numel() != target_tokens:
        raise RuntimeError("equidistant sampler produced duplicate token indices")
    return features.index_select(1, indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dialogs", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("base", "sample", "metacompress"), default="base")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--compressor", type=Path, default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument(
        "--image-protocol",
        choices=("persist", "first-turn-only"),
        default="persist",
        help=(
            "persist keeps the first-turn image token in later prompts. "
            "first-turn-only removes it, so turns 2 and 3 are text-only."
        ),
    )
    parser.add_argument("--question-suffix", default="Answer the question using a single word or phrase.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.method in ("sample", "metacompress") and args.target_tokens is None:
        raise SystemExit(f"--target-tokens is required for --method {args.method}")
    if args.method == "metacompress" and args.compressor is None:
        raise SystemExit("--compressor is required for --method metacompress")
    dialogs = json.loads(args.dialogs.read_text(encoding="utf-8"))
    if args.limit is not None:
        dialogs = dialogs[: args.limit]
    if not dialogs:
        raise SystemExit("no dialogs to evaluate")
    if args.conv_mode not in conv_templates:
        raise SystemExit(f"unknown conversation template: {args.conv_mode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "model_path": str(args.model_path),
        "dialogs": str(args.dialogs),
        "image_root": str(args.image_root),
        "output": str(args.output),
        "dialog_count": len(dialogs),
        "visual_tokens_before_reduction": 576,
        "sampling": "round(linspace(0,575,m))" if args.method == "sample" else None,
        "image_protocol": args.image_protocol,
    }
    args.output.with_suffix(".config.json").write_text(
        json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8"
    )

    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(args.model_path), None, "llava-v1.5-7b",
        load_8bit=args.load_8bit, load_4bit=args.load_4bit, device=args.device,
    )
    model.eval()

    original_encode_images = model.encode_images
    compressor = None
    if args.method == "metacompress":
        compressor = MetaCompress.from_pretrained(args.compressor, map_location="cpu")
        compressor.to(device=model.device, dtype=model.dtype).eval()

    with args.output.open("w", encoding="utf-8") as output:
        for dialog in dialogs:
            image_path = args.image_root / dialog["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = torch.stack(image_tensor)
            image_tensor = image_tensor.to(device=model.device, dtype=model.dtype)
            with torch.inference_mode():
                visual_features = original_encode_images(image_tensor)
                if args.method == "sample":
                    visual_features = equidistant_sample(visual_features, args.target_tokens)
                elif args.method == "metacompress":
                    assert compressor is not None
                    visual_features = compressor(
                        visual_features, target_tokens=args.target_tokens
                    )

            # The stored conversation always records the image on turn 1.
            # persist re-sends that token on later turns. first-turn-only
            # deletes it before building the turn-2 and turn-3 prompts.
            model.encode_images = lambda _images: visual_features
            conversation = conv_templates[args.conv_mode].copy()
            try:
                for turn_index, turn in enumerate(dialog["turns"], start=1):
                    question = turn["question"].strip()
                    if turn_index == 1:
                        question = DEFAULT_IMAGE_TOKEN + "\n" + question
                    if args.question_suffix:
                        question = question + "\n" + args.question_suffix
                    conversation.append_message(conversation.roles[0], question)
                    conversation.append_message(conversation.roles[1], None)
                    prompt_conversation = conversation
                    generate_images = image_tensor
                    generate_sizes = [image.size]
                    if args.image_protocol == "first-turn-only" and turn_index > 1:
                        prompt_conversation = conversation.copy()
                        first_message = prompt_conversation.messages[0][1]
                        if not isinstance(first_message, str) or DEFAULT_IMAGE_TOKEN not in first_message:
                            raise RuntimeError("turn 1 did not record an image token")
                        prompt_conversation.messages[0][1] = first_message.replace(
                            DEFAULT_IMAGE_TOKEN + "\n", "", 1
                        )
                        generate_images = None
                        generate_sizes = None
                    prompt = prompt_conversation.get_prompt()
                    if args.image_protocol == "first-turn-only" and turn_index > 1:
                        if DEFAULT_IMAGE_TOKEN in prompt:
                            raise RuntimeError("later turn prompt still contains an image token")
                    input_ids = tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).to(model.device)
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=generate_images,
                            image_sizes=generate_sizes,
                            do_sample=False,
                            temperature=0.0,
                            num_beams=1,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                        )
                    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    conversation.messages[-1][1] = text
                    output.write(json.dumps({
                        "dialog_id": dialog["dialog_id"],
                        "question_id": str(turn["question_id"]),
                        "turn_index": turn_index,
                        "prediction": text,
                        "text": text,
                        "method": args.method,
                        "target_tokens": args.target_tokens,
                    }, ensure_ascii=True) + "\n")
            finally:
                model.encode_images = original_encode_images
            output.flush()


if __name__ == "__main__":
    main()
