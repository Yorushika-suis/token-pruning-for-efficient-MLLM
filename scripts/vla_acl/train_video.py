#!/usr/bin/env python3
"""Train a per-frame VLA-ACL scorer on video multiple-choice data.

Sources:
  Video-MME train split (--split), or FlexSelect's LLaVA-Video-178K 5% recipe
  (--data-yaml + --json-root). First 178K pass keeps only *_mc_* jsons.

Training clips are 64 uniformly sampled frames, the same count as FastVID /
FlexSelect with force_sample=True. Frames come from --frame-cache, or from
decord when --no-frame-cache is set.

The student LLM sees K spatial tokens per frame plus every newline token.
The frozen teacher sees every spatial token of those frames.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
FAST = ROOT / "ThirdParty" / "FastVID" / "fastvid_llavavideo"
sys.path.insert(0, str(FAST / "LLaVA-NeXT"))
sys.path.insert(0, str(ROOT / "scripts" / "vla_acl"))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX  # noqa: E402
from llava.conversation import conv_templates  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from frames import cache_rel_key, load_training_frames  # noqa: E402
from hook_video import _patch_fastvid, pool_siglip  # noqa: E402
from scorer import (  # noqa: E402
    PatchScorer,
    anneal_alpha,
    patches_per_frame_for,
    soft_top_k,
    use_full_rope_cache,
)

POST_PROMPT = "The best answer is:"
OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video "
    "and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
)
ANSWER_LETTER = re.compile(r"([A-D])", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=None, help="Video-MME split.json with train_video_ids")
    parser.add_argument("--data-yaml", type=Path, default=None, help="FlexSelect output_rnd05.yaml")
    parser.add_argument("--json-root", type=Path, default=None, help="Directory holding the 178K subset jsons")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, default=ROOT / "decode" / "output")
    parser.add_argument(
        "--no-frame-cache",
        action="store_true",
        help="Decode mp4 with decord each step instead of reading decode/output",
    )
    parser.add_argument("--sample-seed", type=int, default=1234, help="Shuffle seed for yaml random:5%")
    parser.add_argument(
        "--mc-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --data-yaml, keep only *_mc_* jsons (default: true)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention", type=float, default=0.10)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--vision-chunk", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gt-weight", type=float, default=0.5)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--alpha-start", type=float, default=2.0)
    parser.add_argument("--alpha-end", type=float, default=0.1)
    parser.add_argument("--hard-log-freq", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.data_yaml is None and args.split is None:
        raise SystemExit("provide --split (Video-MME) or --data-yaml (LLaVA-Video-178K)")
    if args.data_yaml is not None and args.json_root is None:
        raise SystemExit("--json-root is required with --data-yaml")
    if args.no_frame_cache:
        args.frame_cache = None
    return args


def parquet_path() -> Path:
    hf = Path(os.environ.get("HF_HOME", ROOT / "weights"))
    hits = sorted((hf / "hub" / "datasets--lmms-lab--Video-MME" / "snapshots").glob("*/videomme/*.parquet"))
    if not hits:
        raise SystemExit("Video-MME parquet not found")
    return hits[0]


def answer_letter(text: str) -> str | None:
    match = ANSWER_LETTER.search(str(text).strip())
    return match.group(1).upper() if match else None


def strip_image_token(text: str) -> str:
    return str(text).replace("<image>", "").strip()


def resolve_178k_video(video_root: Path, data_source: str, rel: str) -> Path | None:
    rel = str(rel)
    candidates = [rel]
    if not rel.endswith((".mp4", ".MP4")):
        candidates.extend([rel + ".mp4", rel + ".MP4"])
    for name in candidates:
        path = video_root / data_source / name
        if path.is_file():
            return path
    return None


def yaml_json_paths(data_yaml: Path, json_root: Path, mc_only: bool) -> list[Path]:
    paths = []
    for line in data_yaml.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("- json_path:"):
            continue
        rel = stripped.split("LLaVA-Video-178K/", 1)[-1]
        name = Path(rel).name
        if mc_only and "_mc_" not in name:
            continue
        path = json_root / rel
        if not path.is_file():
            raise SystemExit(f"missing annotation json: {path}")
        paths.append(path)
    if not paths:
        raise SystemExit(f"no json paths selected from {data_yaml}")
    return paths


def load_questions_178k(
    data_yaml: Path,
    json_root: Path,
    video_root: Path,
    sample_seed: int,
    mc_only: bool,
    limit: int | None,
) -> list[dict]:
    jsons = yaml_json_paths(data_yaml, json_root, mc_only)
    random.seed(sample_seed)
    examples = []
    missing = 0
    skipped_answer = 0
    for path in jsons:
        data = json.loads(path.read_text())
        k = math.ceil(5 * len(data) / 100)
        random.shuffle(data)
        for item in data[:k]:
            video = resolve_178k_video(video_root, item["data_source"], item["video"])
            if video is None:
                missing += 1
                continue
            turns = item.get("conversations") or []
            for index in range(0, len(turns) - 1, 2):
                human = turns[index]
                gpt = turns[index + 1]
                if human.get("from") != "human" or gpt.get("from") != "gpt":
                    continue
                letter = answer_letter(gpt.get("value", ""))
                if letter is None:
                    skipped_answer += 1
                    continue
                question = strip_image_token(human.get("value", ""))
                if not question:
                    continue
                examples.append(
                    {
                        "video": video,
                        "question": f"{OPTION_PROMPT}\n{question}\n{POST_PROMPT}",
                        "answer": letter,
                    }
                )
                if limit is not None and len(examples) >= limit:
                    print(
                        f"178k mc questions {len(examples)} missing_video {missing} "
                        f"bad_answer {skipped_answer}",
                        flush=True,
                    )
                    return examples
    print(
        f"178k mc questions {len(examples)} missing_video {missing} bad_answer {skipped_answer}",
        flush=True,
    )
    return examples


def load_questions_videomme(train_ids: set[str], video_root: Path, limit: int | None) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path())
    rows = table.to_pylist()
    examples = []
    for row in rows:
        vid = str(row["videoID"])
        if vid not in train_ids:
            continue
        path = video_root / f"{vid}.mp4"
        if not path.is_file():
            alt = video_root / f"{vid}.MP4"
            path = alt if alt.is_file() else path
        if not path.is_file():
            continue
        options = row["options"]
        if isinstance(options, str):
            options = [options]
        question = str(row["question"]) + "\n" + "\n".join(str(opt) for opt in options)
        examples.append(
            {
                "video": path,
                "question": f"{OPTION_PROMPT}\n{question}\n{POST_PROMPT}",
                "answer": str(row["answer"]).strip(),
            }
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def load_examples(args: argparse.Namespace) -> list[dict]:
    if args.data_yaml is not None:
        return load_questions_178k(
            args.data_yaml,
            args.json_root,
            args.video_root,
            args.sample_seed,
            args.mc_only,
            args.limit,
        )
    split = json.loads(args.split.read_text())
    train_ids = set(split["train_video_ids"])
    heldout = set(split["heldout_video_ids"])
    if train_ids & heldout:
        raise SystemExit("split overlap")
    return load_questions_videomme(train_ids, args.video_root, args.limit)


def filter_examples_with_frame_cache(
    examples: list[dict],
    cache_dir: Path,
    video_root: Path | None,
) -> list[dict]:
    """Drop questions whose decode npz is missing (corrupt / never cached)."""
    kept = []
    missing = 0
    for item in examples:
        key = cache_rel_key(item["video"], video_root)
        if (cache_dir / f"{key}.npz").is_file():
            kept.append(item)
        else:
            missing += 1
    print(f"frame-cache keep {len(kept)} drop_missing_cache {missing}", flush=True)
    return kept


def tokenize(tokenizer, question: str, answer: str):
    conv = conv_templates["qwen_1_5"].copy()
    user = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], user)
    conv.append_message(conv.roles[1], None)
    context = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], user)
    conv.append_message(conv.roles[1], answer)
    full = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    labels = full.clone()
    labels[0, : context.shape[1]] = IGNORE_INDEX
    return full, labels


def install_chunked_vision(model, chunk: int) -> None:
    tower = model.get_vision_tower()
    if getattr(tower.forward, "_vla_acl_chunked", False):
        return
    original = tower.forward

    def forward(images):
        if images.shape[0] <= chunk:
            return original(images)
        parts = [original(images[start : start + chunk]) for start in range(0, images.shape[0], chunk)]
        return torch.cat(parts, dim=0)

    forward._vla_acl_chunked = True
    tower.forward = forward


def grid_keep_positions(num_tokens: int, side: int, spatial_index: torch.Tensor) -> torch.Tensor:
    """Positions in the newline grid for the chosen spatial tokens plus every newline."""
    period = side + 1
    index = torch.arange(num_tokens, device=spatial_index.device)
    newline = (index + 1) % period == 0
    spatial_pos = index[~newline]
    chosen = torch.cat((spatial_pos[spatial_index], index[newline])).sort().values
    return chosen


def pad_text(encoded: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int):
    width = max(ids.shape[1] for ids, _ in encoded)
    ids_out, labels_out, mask_out = [], [], []
    for ids, labels in encoded:
        pad = width - ids.shape[1]
        ids_out.append(F.pad(ids, (0, pad), value=pad_id))
        labels_out.append(F.pad(labels, (0, pad), value=IGNORE_INDEX))
        mask = torch.ones(1, ids.shape[1], dtype=torch.bool)
        mask_out.append(F.pad(mask, (0, pad), value=False))
    return torch.cat(ids_out, dim=0), torch.cat(labels_out, dim=0), torch.cat(mask_out, dim=0)


def answer_logits(model, input_ids, labels, frames, pooled_override=None, spatial_index=None, attention_mask=None):
    """One video forward. pooled_override is a list of projected tokens (T, 169, D)."""
    original_pool = model.get_2dPool
    original_grid = model.add_token_per_grid
    pool_cursor = {"n": 0}
    grid_cursor = {"n": 0}

    def student_pool(self, image_feature, stride=2):
        item = pooled_override[pool_cursor["n"]]
        pool_cursor["n"] += 1
        return item

    def student_grid(self, image_feature):
        full = original_grid(image_feature)
        side = int(image_feature.shape[1] ** 0.5)
        index = spatial_index[grid_cursor["n"]].to(full.device)
        grid_cursor["n"] += 1
        keep = grid_keep_positions(full.shape[0], side, index)
        return full[keep]

    if pooled_override is not None:
        model.get_2dPool = student_pool.__get__(model, type(model))
        model.add_token_per_grid = student_grid.__get__(model, type(model))
    try:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        videos = frames if isinstance(frames, list) else [frames]
        prepared = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, labels, videos, ["video"] * len(videos), None
        )
        _, position_ids, prepared_attention, _, inputs_embeds, expanded = prepared[:6]
        outputs = model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=prepared_attention,
            inputs_embeds=inputs_embeds,
            labels=None,
            return_dict=True,
        )
        mask = expanded[:, 1:].ne(IGNORE_INDEX)
        return outputs.logits[:, :-1][mask], expanded[:, 1:][mask]
    finally:
        model.get_2dPool = original_pool
        model.add_token_per_grid = original_grid


def checkpoint_blob(scorer, optimizer, scheduler, step: int, k: int, args, vision_dim: int) -> dict:
    return {
        "state_dict": scorer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "k": k,
        "retention": args.retention,
        "train_frames": args.frames,
        "vision_dim": vision_dim,
        "hidden_dim": scorer.hidden_dim,
        "num_layers": scorer.num_layers,
        "num_heads": scorer.num_heads,
    }


def save_checkpoint(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(blob, temporary)
    temporary.replace(path)


def move_optimizer_state(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_resume(path: Path, scorer, optimizer, scheduler, k: int, args, total_steps: int) -> int:
    blob = torch.load(path, map_location="cpu")
    if int(blob["k"]) != k or abs(float(blob["retention"]) - args.retention) > 1e-6:
        raise SystemExit(
            f"checkpoint k={blob['k']} retention={blob['retention']} "
            f"does not match this run k={k} retention={args.retention}"
        )
    if int(blob.get("train_frames", args.frames)) != args.frames:
        raise SystemExit(f"checkpoint train_frames={blob.get('train_frames')} does not match {args.frames}")
    # Checkpoints written before resume support only exist after a finished run.
    saved_step = int(blob["step"]) if "step" in blob else total_steps
    if saved_step >= total_steps:
        print(f"checkpoint already finished at step {saved_step}, nothing to resume", flush=True)
        return saved_step
    if "optimizer" not in blob or "scheduler" not in blob:
        raise SystemExit(f"{path} has no optimizer state, cannot resume a partial run")
    scorer.load_state_dict(blob["state_dict"])
    optimizer.load_state_dict(blob["optimizer"])
    move_optimizer_state(optimizer, scorer.parameters().__next__().device)
    scheduler.load_state_dict(blob["scheduler"])
    print(f"resumed {path} at step {saved_step}", flush=True)
    return saved_step


def main() -> None:
    args = parse_args()
    weight = args.gt_weight + args.teacher_weight
    if weight <= 0:
        raise SystemExit("gt-weight + teacher-weight must be > 0")
    k = patches_per_frame_for(args.retention)
    examples = load_examples(args)
    if args.frame_cache is not None:
        examples = filter_examples_with_frame_cache(examples, args.frame_cache, args.video_root)
    if not examples:
        raise SystemExit("no training questions with a local video")
    source = "178k" if args.data_yaml is not None else "videomme"
    cache = "online-decord" if args.frame_cache is None else str(args.frame_cache)
    print(
        f"source {source} questions {len(examples)} retention {args.retention} "
        f"k {k} frames {args.frames} cache {cache}",
        flush=True,
    )
    total_steps = max(len(examples) * args.epochs, 1)
    ckpt_path = args.output / "patch_scorer.pt"
    if args.resume and ckpt_path.is_file():
        preview = torch.load(ckpt_path, map_location="cpu")
        if int(preview["k"]) != k or abs(float(preview["retention"]) - args.retention) > 1e-6:
            raise SystemExit(
                f"checkpoint k={preview['k']} retention={preview['retention']} "
                f"does not match this run k={k} retention={args.retention}"
            )
        finished = int(preview["step"]) if "step" in preview else total_steps
        if finished >= total_steps:
            print(f"checkpoint already finished at step {finished}, nothing to resume", flush=True)
            return

    tokenizer, model, processor, _ = load_pretrained_model(
        str(args.model_path),
        None,
        "llava-video-qwen",
        device_map=args.device,
        attn_implementation="sdpa",
        # The released checkpoint has no FastVID fields. Qwen2Model.__init__ reads them.
        # Retention 1.0 keeps FastVID's own prune from dropping tokens; the scorer does that.
        overwrite_config={
            "mm_spatial_pool_mode": "average",
            "mm_spatial_pool_stride": 2,
            "mm_newline_position": "grid",
            "fastvid_retention_ratio": 1.0,
            "fastvid_DySeg_c": 8,
            "fastvid_DySeg_tau": 0.9,
            "fastvid_STPrune_d": 0.4,
            "fastvid_DTM_p": 4,
            "fastvid_DTM_beta": 0.6,
        },
    )
    model.config.use_cache = False
    use_full_rope_cache(model)
    model.gradient_checkpointing_enable()
    model.model.train()
    model.get_vision_tower().eval()
    install_chunked_vision(model, args.vision_chunk)
    for param in model.parameters():
        param.requires_grad_(False)
    from llava.model.language_model.modeling_qwen2 import Qwen2Model

    _patch_fastvid(Qwen2Model)
    inner = getattr(model, "model", None)
    qwen = getattr(inner, "model", None) if inner is not None else None
    for module in (inner, qwen):
        if module is not None:
            module.vla_acl_skip_fastvid = True

    black = torch.zeros(1, 3, processor.crop_size["height"], processor.crop_size["width"])
    black = processor.preprocess(black, return_tensors="pt")["pixel_values"]
    black = black.to(device=model.device, dtype=model.dtype)
    with torch.no_grad():
        black_raw = model.get_vision_tower()(black)
        black_pooled = pool_siglip(black_raw)

    vision_dim = int(black_pooled.shape[-1])
    scorer = PatchScorer(vision_dim=vision_dim).to(device=model.device, dtype=torch.float32)
    optimizer = AdamW(scorer.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)
    optimizer.zero_grad(set_to_none=True)
    projector = model.get_model().mm_projector
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    step = 0
    if args.resume:
        if ckpt_path.is_file():
            step = load_resume(ckpt_path, scorer, optimizer, scheduler, k, args, total_steps)
            if step >= total_steps:
                return
        else:
            print(f"no checkpoint at {ckpt_path}, starting from step 0", flush=True)
    for epoch in range(args.epochs):
        for start in range(0, len(examples), args.batch_size):
            if epoch * len(examples) + start < step:
                continue
            batch = examples[start : start + args.batch_size]
            videos = []
            kept = []
            for item in batch:
                try:
                    clip = load_training_frames(
                        item["video"], args.frames, args.frame_cache, args.video_root
                    )
                except Exception as exc:
                    print(f"skip bad video {item['video']}: {exc}", flush=True)
                    continue
                clip = processor.preprocess(clip, return_tensors="pt")["pixel_values"]
                videos.append(clip.to(device=model.device, dtype=model.dtype))
                kept.append(item)
            if not kept:
                continue
            batch = kept
            encoded = [tokenize(tokenizer, item["question"], item["answer"]) for item in batch]
            input_ids, labels, attention_mask = pad_text(encoded, pad_id)
            input_ids = input_ids.to(model.device)
            labels = labels.to(model.device)
            attention_mask = attention_mask.to(model.device)
            pooled_list = []
            projected_list = []
            index_list = []
            alpha = anneal_alpha(step, total_steps, args.alpha_start, args.alpha_end)
            for clip in videos:
                with torch.no_grad():
                    raw = model.get_vision_tower()(clip)
                pooled = pool_siglip(raw)
                pooled_list.append(pooled)
                logits = scorer(pooled.float())
                scores = soft_top_k(logits, k, alpha)
                local = logits.detach().topk(k, dim=-1).indices
                hard = torch.zeros_like(scores).scatter_(-1, local, 1.0)
                straight = hard.detach() - scores.detach() + scores
                weighted = straight.unsqueeze(-1) * pooled
                projected_list.append(projector(weighted.to(pooled.dtype)))
                offset = torch.arange(local.shape[0], device=local.device).unsqueeze(1) * pooled.shape[1]
                index_list.append((local + offset).reshape(-1))
            student_logits, student_targets = answer_logits(
                model, input_ids, labels, videos, projected_list, index_list, attention_mask
            )
            with torch.no_grad():
                teacher_logits, teacher_targets = answer_logits(
                    model, input_ids, labels, videos, None, None, attention_mask
                )
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
            message = (
                f"step {step} loss {loss.item():.4f} ce {ce.item():.4f} "
                f"kl {kl.item():.4f} alpha {alpha:.3f} batch {len(batch)}"
            )
            print(message, flush=True)
            step += len(batch)
            if step % args.grad_accumulation == 0 or start + args.batch_size >= len(examples):
                optimizer.step()
                for _ in range(len(batch)):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if args.save_every > 0 and (step % args.save_every == 0 or step >= total_steps):
                    save_checkpoint(
                        ckpt_path,
                        checkpoint_blob(scorer, optimizer, scheduler, step, k, args, vision_dim),
                    )
                    print(f"saved {ckpt_path} step {step}", flush=True)

    save_checkpoint(ckpt_path, checkpoint_blob(scorer, optimizer, scheduler, step, k, args, vision_dim))
    print(f"saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
