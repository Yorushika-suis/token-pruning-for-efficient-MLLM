#!/usr/bin/env python3
"""Train MetaCompress with a frozen local LLaVA-1.5 model.

The adapter follows the paper's teacher/student protocol: the original
projector output is the teacher input, MetaCompress produces the student
visual sequence, and only assistant response positions contribute to the
distillation loss. Dialogues are processed one at a time to keep GPU memory
bounded.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ThirdParty" / "LLaVA"))
sys.path.insert(0, str(ROOT / "ThirdParty" / "MetaCompress"))

from llava.constants import IGNORE_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from metacompress import (  # noqa: E402
    MetaCompress,
    MetaCompressLoss,
    build_paper_optimizer,
    distillation_forward,
    freeze_module,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dialogs", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compression-dim", type=int, default=256)
    parser.add_argument("--target-tokens", type=int, default=57)
    parser.add_argument("--pool-mode", choices=("avg", "max"), default="max")
    parser.add_argument(
        "--position-embedding",
        choices=("sinusoidal", "learned", "none"),
        default="sinusoidal",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conv-mode", choices=("vicuna_v1", "v1"), default="vicuna_v1")
    parser.add_argument(
        "--image-protocol",
        choices=("persist", "first-turn-only"),
        default="persist",
        help=(
            "persist lets later answers attend to the turn-1 image tokens. "
            "first-turn-only trains on the first question alone, matching an "
            "eval where turns 2 and 3 contain no image token."
        ),
    )
    parser.add_argument(
        "--question-suffix",
        default="Answer the question using a single word or phrase.",
        help="Must match run_mt_gqa.py so distillation uses the same user turns as eval.",
    )
    parser.add_argument(
        "--testdev-question-ids",
        type=Path,
        default=None,
        help="Reject the run if any training question ID appears in this testdev list.",
    )
    parser.add_argument(
        "--testdev-image-ids",
        type=Path,
        default=None,
        help="Optional testdev image IDs; the intersection is recorded, not rejected.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def load_id_list(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def make_source(
    dialog: dict[str, Any],
    question_suffix: str,
    image_protocol: str = "persist",
) -> list[dict[str, str]]:
    """Convert a reconstructed three-turn item to LLaVA's train format.

    ``first-turn-only`` keeps the first question. Later questions have no
    image token at eval time, and the frozen LVLM makes their teacher and
    student logits identical, so they do not train MetaCompress.
    """
    source: list[dict[str, str]] = []
    suffix = question_suffix.strip()
    turns = dialog["turns"][:1] if image_protocol == "first-turn-only" else dialog["turns"]
    for index, turn in enumerate(turns):
        question = str(turn["question"]).strip()
        if index == 0:
            question = "<image>\n" + question
        if suffix:
            question = question + "\n" + suffix
        # Short GQA `answer` is the first-version teacher-forced target; do not
        # fall back to the natural-language fullAnswer.
        answer = str(turn.get("answer") or "").strip()
        if not answer:
            raise ValueError(f"{dialog.get('dialog_id')}: turn {index} has an empty short answer")
        source.append({"from": "human", "value": question})
        source.append({"from": "gpt", "value": answer})
    return source


def preprocess_vicuna_v1(
    source: list[dict[str, str]], tokenizer: Any, conv_mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small dependency-free equivalent of LLaVA's ``preprocess_v1``.

    Importing ``llava.train.train`` also imports ``transformers.Trainer`` and
    optional sklearn integrations.  Those are unrelated to this adapter and
    can be ABI-incompatible in an otherwise usable inference environment.
    """
    conv = conv_templates[conv_mode].copy()
    if conv.sep2 is None:
        raise ValueError(f"conversation template {conv_mode!r} is not a two-separator template")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conv.messages = []
    for sentence in source:
        conv.append_message(roles[sentence["from"]], sentence["value"])
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    labels = input_ids.clone()

    # Mirror preprocess_v1's response-only mask. Each round consists of the
    # user part followed by `` ASSISTANT: answer</s>``.
    separator = conv.sep + conv.roles[1] + ": "
    total_len = int(input_ids[0].ne(tokenizer.pad_token_id).sum())
    rounds = prompt.split(conv.sep2)
    cursor = 1
    labels[0, :cursor] = IGNORE_INDEX
    for round_index, round_text in enumerate(rounds):
        if not round_text:
            break
        parts = round_text.split(separator)
        if len(parts) != 2:
            break
        instruction = parts[0] + separator
        round_len = len(tokenizer_image_token(round_text, tokenizer))
        instruction_len = len(tokenizer_image_token(instruction, tokenizer)) - 2
        if round_index != 0 and not getattr(tokenizer, "legacy", True):
            round_len -= 1
            instruction_len -= 1
        labels[0, cursor : cursor + instruction_len] = IGNORE_INDEX
        cursor += round_len
    labels[0, cursor:] = IGNORE_INDEX
    if cursor != total_len:
        labels.fill_(IGNORE_INDEX)
        raise RuntimeError(
            f"tokenization mismatch in vicuna_v1 preprocessing: {cursor} vs {total_len}"
        )
    return input_ids, labels


def prepare_example(
    dialog: dict[str, Any],
    image_root: Path,
    tokenizer: Any,
    processor: Any,
    model: torch.nn.Module,
    conv_mode: str,
    question_suffix: str,
    image_protocol: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Tokenize one dialogue and process its image."""
    input_ids, labels = preprocess_vicuna_v1(
        copy.deepcopy(make_source(dialog, question_suffix, image_protocol)),
        tokenizer,
        conv_mode,
    )

    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)
    image_path = image_root / dialog["image"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image_size = image.size
    image_tensor = process_images([image], processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor)
    image_tensor = image_tensor.to(device=model.device, dtype=model.dtype)
    return input_ids, labels, image_tensor, [image_size]


def make_model_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    image_tensor: torch.Tensor,
    image_sizes: list[tuple[int, int]],
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]:
    """Return a callback that extracts logits at assistant response tokens."""
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    original_encode_images = model.encode_images
    response_count: int | None = None

    def model_forward(visual_tokens: torch.Tensor) -> torch.Tensor:
        nonlocal response_count
        # LLaVA's multimodal preparation calls encode_images. Replacing this
        # instance method injects either teacher or compressed features while
        # retaining the model's normal text/vision sequence construction.
        model.encode_images = lambda _images: visual_tokens
        try:
            (
                _prepared_ids,
                position_ids,
                prepared_attention,
                _past,
                inputs_embeds,
                expanded_labels,
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                attention_mask,
                None,
                labels,
                image_tensor,
                image_sizes,
            )
            if inputs_embeds is None or expanded_labels is None:
                raise RuntimeError("LLaVA did not expand multimodal inputs")
            # CausalLM logits at position t predict label t+1. Keep the
            # assistant mask in that shifted coordinate system for both
            # teacher and student, whose visual prefix lengths differ.
            current_mask = expanded_labels[:, 1:].ne(IGNORE_INDEX)
            current_count = int(current_mask.sum().item())
            if current_count == 0:
                raise RuntimeError("dialogue has no supervised assistant tokens")
            if response_count is None:
                response_count = current_count
            elif response_count != current_count:
                raise RuntimeError("teacher/student response token counts differ")
            outputs = model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=prepared_attention,
                inputs_embeds=inputs_embeds,
                labels=None,
                return_dict=True,
            )
            return outputs.logits[:, :-1, :][current_mask]
        finally:
            model.encode_images = original_encode_images

    # The number of response labels is unchanged by replacing one image token
    # with a visual sequence, so this all-ones mask matches flattened logits.
    response_mask = torch.ones(
        (int(labels[:, 1:].ne(IGNORE_INDEX).sum().item()),),
        device=model.device,
        dtype=torch.bool,
    )
    return model_forward, response_mask


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.target_tokens <= 0:
        raise SystemExit("epochs, batch-size, and target-tokens must be positive")
    if args.max_grad_norm <= 0 or args.learning_rate <= 0:
        raise SystemExit("learning-rate and max-grad-norm must be positive")
    if args.conv_mode not in conv_templates:
        raise SystemExit(f"unknown conversation template: {args.conv_mode}")

    torch.manual_seed(args.seed)
    dialogs = json.loads(args.dialogs.read_text(encoding="utf-8"))
    if args.limit is not None:
        dialogs = dialogs[: args.limit]
    if not dialogs:
        raise SystemExit("no dialogs to train")

    train_question_ids = [
        str(turn["question_id"]) for dialog in dialogs for turn in dialog["turns"]
    ]
    testdev_overlap: list[str] = []
    if args.testdev_question_ids is not None:
        holdout = load_id_list(args.testdev_question_ids)
        testdev_overlap = sorted(set(train_question_ids) & holdout)
        if testdev_overlap:
            raise SystemExit(
                f"{len(testdev_overlap)} training question IDs overlap testdev; "
                f"first IDs: {testdev_overlap[:10]}"
            )
    image_intersection: list[str] = []
    if args.testdev_image_ids is not None:
        testdev_images = load_id_list(args.testdev_image_ids)
        train_images = {str(dialog["image_id"]) for dialog in dialogs}
        image_intersection = sorted(train_images & testdev_images)

    tokenizer, model, processor, _ = load_pretrained_model(
        str(args.model_path),
        None,
        "llava-v1.5-7b",
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )
    model.eval()
    freeze_module(model)
    model_dtype = next(model.parameters()).dtype
    hidden_size = int(model.config.hidden_size)
    compressor = MetaCompress(
        input_dim=hidden_size,
        compression_dim=args.compression_dim,
        reduction_rate=0.9,
        pool_kernel_size=3,
        pool_mode=args.pool_mode,
        position_embedding=args.position_embedding,
    ).to(device=model.device)
    compressor.train()
    optimizer = build_paper_optimizer(compressor, learning_rate=args.learning_rate)
    criterion = MetaCompressLoss(entropy_weight=1.0, collapse_weight=1.0)

    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args))
    config.update(
        {
            "model_path": str(args.model_path),
            "dialogs": str(args.dialogs),
            "image_root": str(args.image_root),
            "output": str(args.output),
            "dialog_count": len(dialogs),
            "input_dim": hidden_size,
            "model_dtype": str(model_dtype),
            "teacher_forced_field": "answer",
            "question_suffix": args.question_suffix,
            "image_protocol": args.image_protocol,
            "testdev_question_overlap_count": len(testdev_overlap),
            "testdev_image_intersection_count": len(image_intersection),
            "testdev_image_intersection": image_intersection,
        }
    )
    (args.output / "run_config.json").write_text(
        json.dumps(config, default=str, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = args.output / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    global_step = 0

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, args.epochs + 1):
            epoch_values: list[float] = []
            pending = 0
            epoch_order = list(range(len(dialogs)))
            random.Random(args.seed + epoch - 1).shuffle(epoch_order)
            for dialog_index, source_index in enumerate(epoch_order):
                dialog = dialogs[source_index]
                input_ids, labels, image_tensor, image_sizes = prepare_example(
                    dialog,
                    args.image_root,
                    tokenizer,
                    processor,
                    model,
                    args.conv_mode,
                    args.question_suffix,
                    args.image_protocol,
                )
                with torch.no_grad():
                    visual_tokens = model.encode_images(image_tensor).detach()
                if visual_tokens.ndim != 3 or visual_tokens.shape[0] != 1:
                    raise RuntimeError(
                        f"expected one [N,D] visual sequence, got {tuple(visual_tokens.shape)}"
                    )
                if visual_tokens.shape[1] != 576 or visual_tokens.shape[2] != hidden_size:
                    raise RuntimeError(
                        "this adapter expects LLaVA-1.5 projector output [1,576,hidden_size], "
                        f"got {tuple(visual_tokens.shape)}"
                    )
                model_forward, token_mask = make_model_forward(
                    model, input_ids, labels, image_tensor, image_sizes
                )
                result = distillation_forward(
                    compressor,
                    visual_tokens,
                    model_forward,
                    criterion=criterion,
                    target_tokens=args.target_tokens,
                    token_mask=token_mask,
                )
                loss_terms = torch.stack(
                    [
                        result.losses.loss,
                        result.losses.prediction,
                        result.losses.entropy,
                        result.losses.collapse,
                    ]
                )
                if not torch.isfinite(loss_terms).all():
                    raise FloatingPointError(f"non-finite loss at dialog {dialog['dialog_id']}")
                row_sum_error = float(
                    (result.projection.sum(dim=-1) - 1.0).abs().max().detach().cpu()
                )
                if row_sum_error > 1e-4:
                    raise RuntimeError(f"projection rows do not sum to one: {row_sum_error}")
                (result.losses.loss / args.batch_size).backward()
                if dialog_index == 0 and any(
                    parameter.grad is not None for parameter in model.parameters()
                ):
                    raise RuntimeError("a frozen LLaVA parameter received a gradient")
                pending += 1
                loss_value = float(result.losses.loss.detach().cpu())
                epoch_values.append(loss_value)

                should_step = pending == args.batch_size or dialog_index == len(dialogs) - 1
                grad_norm = None
                if should_step:
                    if pending != args.batch_size:
                        scale = args.batch_size / pending
                        for parameter in compressor.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(scale)
                    if any(
                        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                        for parameter in compressor.parameters()
                    ):
                        raise FloatingPointError("MetaCompress produced a non-finite gradient")
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        compressor.parameters(), max_norm=args.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    global_step += 1

                record = {
                    "epoch": epoch,
                    "dialog_index": dialog_index,
                    "source_index": source_index,
                    "dialog_id": dialog["dialog_id"],
                    "loss": loss_value,
                    "prediction": float(result.losses.prediction.detach().cpu()),
                    "entropy": float(result.losses.entropy.detach().cpu()),
                    "collapse": float(result.losses.collapse.detach().cpu()),
                    "projection_row_sum_error": row_sum_error,
                    "max_column_mass": float(
                        result.projection.sum(dim=-2).amax().detach().cpu()
                    ),
                    "grad_norm": None if grad_norm is None else float(grad_norm.detach().cpu()),
                    "global_step": global_step,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()

            compressor.save_pretrained(args.output / f"checkpoint-epoch{epoch}")
            (args.output / f"epoch{epoch}.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "mean_loss": sum(epoch_values) / max(1, len(epoch_values)),
                        "dialog_count": len(epoch_values),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    compressor.save_pretrained(args.output / "checkpoint-final")


if __name__ == "__main__":
    main()
