#!/usr/bin/env python3
"""Train a VLA-ACL patch scorer on frozen LLaVA-1.5 with GQA short answers.

Soft top-K keeps all 576 patches. The loss is answer CE plus KL to the
full-token teacher. A periodic hard top-K forward is logged and not differentiated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ThirdParty" / "LLaVA"))
sys.path.insert(0, str(ROOT / "scripts" / "vla_acl"))

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from scorer import (  # noqa: E402
    PatchScorer,
    anneal_alpha,
    gate_vision_tokens,
    hard_top_k_indices,
    preserving_position_ids,
    soft_top_k,
    use_full_rope_cache,
)

QUESTION_SUFFIX = "Answer the question using a single word or phrase."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--testdev-question-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gt-weight", type=float, default=0.5)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--alpha-start", type=float, default=2.0)
    parser.add_argument("--alpha-end", type=float, default=0.1)
    parser.add_argument("--hard-log-freq", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="If >0, time each stage for this many steps, print the breakdown, and exit without saving.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_examples(args: argparse.Namespace) -> list[dict]:
    """Load the MetaCompress 20k-dialog subset, or a raw GQA question dict.

    Dialogs are expanded to one single-turn example per question. The short
    GQA answer is the target, matching the image eval prompt.
    """
    holdout = {line.strip() for line in args.testdev_question_ids.read_text().splitlines() if line.strip()}
    payload = json.loads(args.questions.read_text())
    examples = []

    def add(qid: str, image_name: str, question: str, answer: str) -> bool:
        if qid in holdout:
            return False
        image_path = args.image_root / image_name
        if not image_path.is_file():
            return False
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return False
        examples.append({"qid": qid, "image": image_path, "question": question, "answer": answer})
        return args.limit is not None and len(examples) >= args.limit

    if isinstance(payload, list):
        for dialog in payload:
            image_name = str(dialog.get("image") or f"{dialog['image_id']}.jpg")
            for turn in dialog["turns"]:
                if add(str(turn["question_id"]), image_name, str(turn["question"]), str(turn.get("answer") or "")):
                    return examples
        return examples

    for qid, item in payload.items():
        if add(str(qid), f"{item['imageId']}.jpg", str(item.get("question") or ""), str(item.get("answer") or "")):
            break
    return examples


def tokenize(tokenizer, question: str, answer: str):
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question}\n{QUESTION_SUFFIX}")
    conv.append_message(conv.roles[1], answer)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0)
    labels = input_ids.clone()
    separator = conv.sep + conv.roles[1] + ": "
    rounds = prompt.split(conv.sep2)
    cursor = 1
    labels[0, :cursor] = IGNORE_INDEX
    total_len = int(input_ids[0].ne(tokenizer.pad_token_id).sum())
    for round_index, round_text in enumerate(rounds):
        if not round_text:
            break
        parts = round_text.split(separator)
        if len(parts) != 2:
            break
        instruction_len = len(tokenizer_image_token(parts[0] + separator, tokenizer)) - 2
        round_len = len(tokenizer_image_token(round_text, tokenizer))
        if round_index != 0 and not getattr(tokenizer, "legacy", True):
            round_len -= 1
            instruction_len -= 1
        labels[0, cursor : cursor + instruction_len] = IGNORE_INDEX
        cursor += round_len
    labels[0, cursor:] = IGNORE_INDEX
    if cursor != total_len:
        raise RuntimeError(f"tokenization mismatch: {cursor} vs {total_len}")
    return input_ids, labels


def vision_patches(model, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.get_model().get_vision_tower()(images)


def project(model, patches: torch.Tensor) -> torch.Tensor:
    return model.get_model().mm_projector(patches)


def answer_logits(model, input_ids, labels, image_tensor, projected, position_ids=None, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    original = model.encode_images
    model.encode_images = lambda _images: projected
    try:
        _, default_positions, prepared_attention, _, inputs_embeds, expanded_labels = (
            model.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, labels, image_tensor, None
            )
        )
        if position_ids is None:
            position_ids = default_positions
        elif inputs_embeds is not None and position_ids.shape[1] != inputs_embeds.shape[1]:
            raise RuntimeError(
                f"position ids {tuple(position_ids.shape)} do not match embeds {tuple(inputs_embeds.shape)}"
            )
        outputs = model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=prepared_attention,
            inputs_embeds=inputs_embeds,
            labels=None,
            return_dict=True,
        )
        mask = expanded_labels[:, 1:].ne(IGNORE_INDEX)
        return outputs.logits[:, :-1, :][mask], expanded_labels[:, 1:][mask]
    finally:
        model.encode_images = original


def hard_position_ids(
    input_ids: torch.Tensor, keep: torch.Tensor, full_patches: int, seq_len: int
) -> torch.Tensor:
    prefix_len = int((input_ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0].item())
    suffix_len = seq_len - prefix_len - int(keep.shape[1])
    if suffix_len < 0:
        raise RuntimeError(f"pruned sequence shorter than prefix+keep: {seq_len}")
    ids = preserving_position_ids(prefix_len, keep[0], full_patches, suffix_len, input_ids.device)
    return ids.unsqueeze(0)


def collate_batch(tokenizer, processor, model, batch: list[dict]):
    encoded = [tokenize(tokenizer, item["question"], item["answer"]) for item in batch]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(ids.shape[1] for ids, _ in encoded)
    ids_out, labels_out, mask_out, images = [], [], [], []
    for (ids, labels), item in zip(encoded, batch):
        pad = width - ids.shape[1]
        ids_out.append(F.pad(ids, (0, pad), value=pad_id))
        labels_out.append(F.pad(labels, (0, pad), value=IGNORE_INDEX))
        mask = torch.ones(1, ids.shape[1], dtype=torch.bool)
        mask_out.append(F.pad(mask, (0, pad), value=False))
        with Image.open(item["image"]) as opened:
            image = opened.convert("RGB")
        images.append(process_images([image], processor, model.config))
    return (
        torch.cat(ids_out, dim=0),
        torch.cat(labels_out, dim=0),
        torch.cat(mask_out, dim=0),
        torch.cat(images, dim=0),
    )


class StageTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def report(self) -> None:
        order = [
            "load_image",
            "vision",
            "scorer",
            "teacher_llm",
            "student_llm",
            "loss_backward",
            "hard_topk",
            "optimizer",
        ]
        names = [name for name in order if name in self.totals]
        names += [name for name in self.totals if name not in names]
        total = sum(self.totals[name] for name in names)
        print("profile seconds", flush=True)
        for name in names:
            sec = self.totals[name]
            n = self.counts[name]
            share = 100.0 * sec / total if total else 0.0
            print(f"  {name:16} {sec:8.2f}s  n={n:<4} mean={sec / n:6.3f}s  {share:5.1f}%", flush=True)
        print(f"  {'total':16} {total:8.2f}s", flush=True)
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(f"  cuda_allocated {alloc:.2f} GiB  cuda_reserved {reserved:.2f} GiB", flush=True)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")
    weight = args.gt_weight + args.teacher_weight
    if weight <= 0:
        raise SystemExit("gt-weight + teacher-weight must be > 0")
    torch.manual_seed(args.seed)
    examples = load_examples(args)
    if not examples:
        raise SystemExit("no training examples with a local image")
    print(f"examples {len(examples)} k {args.k}", flush=True)

    tokenizer, model, processor, _ = load_pretrained_model(
        str(args.model_path), None, "llava-v1.5-7b", device_map=args.device
    )
    model.config.use_cache = False
    use_full_rope_cache(model)
    model.gradient_checkpointing_enable()
    # Checkpointing runs only on modules in train mode. Keep the vision tower
    # in eval so its dropout stays off; the language stack is frozen either way.
    model.model.train()
    model.get_vision_tower().eval()
    for param in model.parameters():
        param.requires_grad_(False)

    black = Image.new("RGB", (336, 336))
    black_tensor = process_images([black], processor, model.config)
    black_tensor = black_tensor.to(device=model.device, dtype=model.dtype)
    black_embeddings = vision_patches(model, black_tensor)

    scorer = PatchScorer().to(device=model.device, dtype=torch.float32)
    optimizer = AdamW(scorer.parameters(), lr=args.learning_rate)
    steps_per_epoch = len(examples)
    total_steps = steps_per_epoch * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=0.0)
    optimizer.zero_grad(set_to_none=True)

    step = 0
    micro = 0
    timer = StageTimer() if args.profile_steps > 0 else None
    for epoch in range(args.epochs):
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            if timer is not None:
                sync_cuda()
                mark = time.perf_counter()
            input_ids, labels, attention_mask, image_tensor = collate_batch(tokenizer, processor, model, batch)
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            attention_mask = attention_mask.to(model.device)
            image_tensor = image_tensor.to(device=model.device, dtype=model.dtype)
            if timer is not None:
                sync_cuda()
                timer.add("load_image", time.perf_counter() - mark)
                mark = time.perf_counter()
            patches = vision_patches(model, image_tensor)
            if timer is not None:
                sync_cuda()
                timer.add("vision", time.perf_counter() - mark)
                mark = time.perf_counter()
            full_patches = patches.shape[1]
            alpha = anneal_alpha(step, total_steps, args.alpha_start, args.alpha_end)
            logits = scorer(patches.float())
            scores = soft_top_k(logits, args.k, alpha)
            gated = gate_vision_tokens(patches, scores, black_embeddings)
            student_projected = project(model, gated.to(patches.dtype))
            if timer is not None:
                sync_cuda()
                timer.add("scorer", time.perf_counter() - mark)
                mark = time.perf_counter()
            with torch.no_grad():
                teacher_projected = project(model, patches)
                teacher_logits, teacher_targets = answer_logits(
                    model, input_ids, labels, image_tensor, teacher_projected, attention_mask=attention_mask
                )
            if timer is not None:
                sync_cuda()
                timer.add("teacher_llm", time.perf_counter() - mark)
                mark = time.perf_counter()
            student_logits, student_targets = answer_logits(
                model, input_ids, labels, image_tensor, student_projected, attention_mask=attention_mask
            )
            if timer is not None:
                sync_cuda()
                timer.add("student_llm", time.perf_counter() - mark)
                mark = time.perf_counter()
            if student_targets.shape != teacher_targets.shape or not torch.equal(student_targets, teacher_targets):
                raise RuntimeError("teacher and student answer tokens diverged")
            ce = F.cross_entropy(student_logits.float(), student_targets)
            kl = F.kl_div(
                F.log_softmax(student_logits.float(), dim=-1),
                F.softmax(teacher_logits.float(), dim=-1),
                reduction="batchmean",
            )
            loss = (args.gt_weight / weight) * ce + (args.teacher_weight / weight) * kl
            (loss / args.grad_accumulation).backward()
            if timer is not None:
                sync_cuda()
                timer.add("loss_backward", time.perf_counter() - mark)
            message = f"step {step} loss {loss.item():.4f} ce {ce.item():.4f} kl {kl.item():.4f} alpha {alpha:.3f}"
            if args.hard_log_freq and step % args.hard_log_freq == 0:
                if timer is not None:
                    sync_cuda()
                    mark = time.perf_counter()
                with torch.no_grad():
                    keep = hard_top_k_indices(logits, args.k)
                    kept = torch.gather(
                        patches, 1, keep.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
                    )
                    kept_projected = project(model, kept)
                    seq_len = int(input_ids.shape[1] - 1 + kept_projected.shape[1])
                    hard_logits, hard_targets = answer_logits(
                        model,
                        input_ids,
                        labels,
                        image_tensor,
                        kept_projected,
                        hard_position_ids(input_ids, keep, full_patches, seq_len) if input_ids.shape[0] == 1 else None,
                        attention_mask=attention_mask,
                    )
                    hard_ce = F.cross_entropy(hard_logits.float(), hard_targets)
                if timer is not None:
                    sync_cuda()
                    timer.add("hard_topk", time.perf_counter() - mark)
                message += f" hard_ce {hard_ce.item():.4f}"
            print(message, flush=True)
            micro += 1
            step += len(batch)
            if micro % args.grad_accumulation == 0 or start + args.batch_size >= len(examples):
                if timer is not None:
                    sync_cuda()
                    mark = time.perf_counter()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if timer is not None:
                    sync_cuda()
                    timer.add("optimizer", time.perf_counter() - mark)
            if timer is not None and step >= args.profile_steps:
                timer.report()
                return

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "patch_scorer.pt"
    torch.save(
        {
            "state_dict": scorer.state_dict(),
            "k": args.k,
            "vision_dim": scorer.vision_dim,
            "hidden_dim": scorer.hidden_dim,
            "num_layers": scorer.num_layers,
            "num_heads": scorer.num_heads,
        },
        path,
    )
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
