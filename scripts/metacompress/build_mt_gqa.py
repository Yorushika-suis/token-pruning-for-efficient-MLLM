#!/usr/bin/env python3
"""Build an auditable, unofficial MT-GQA training subset from GQA questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_SEED = 42
DEFAULT_DIALOGS = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path: Path) -> OrderedDict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        questions = json.load(handle, object_pairs_hook=OrderedDict)
    if not isinstance(questions, dict):
        raise ValueError("GQA questions file must contain a JSON object")
    return questions


def build_candidates(questions: OrderedDict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_image: OrderedDict[str, list[tuple[str, dict[str, Any]]]] = OrderedDict()
    for question_id, question in questions.items():
        image_id = str(question["imageId"])
        by_image.setdefault(image_id, []).append((str(question_id), question))

    candidates: list[dict[str, Any]] = []
    for image_id, image_questions in by_image.items():
        for group_index in range(len(image_questions) // 3):
            group = image_questions[group_index * 3 : group_index * 3 + 3]
            turns = []
            for question_id, question in group:
                turns.append(
                    {
                        "question_id": question_id,
                        "question": question["question"],
                        "answer": question.get("answer"),
                        "full_answer": question.get("fullAnswer"),
                    }
                )
            candidates.append(
                {
                    "image_id": image_id,
                    "image": f"{image_id}.jpg",
                    "source_group_index": group_index,
                    "turns": turns,
                }
            )
    return candidates


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-dialogs", type=int, default=DEFAULT_DIALOGS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_dialogs <= 0:
        raise ValueError("--num-dialogs must be positive")

    questions = load_questions(args.questions)
    candidates = build_candidates(questions)
    if args.num_dialogs > len(candidates):
        raise ValueError(
            f"requested {args.num_dialogs} dialogs, only {len(candidates)} candidates exist"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.num_dialogs)
    selected.sort(key=lambda item: (item["image_id"], item["source_group_index"]))

    records = []
    for dialog_index, candidate in enumerate(selected):
        records.append(
            {
                "dialog_id": f"gqa_train_balanced_{dialog_index:06d}",
                "source_split": "train_balanced",
                **candidate,
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "mt_gqa_train_balanced_20000.jsonl"
    json_path = output_dir / "mt_gqa_train_balanced_20000.json"
    manifest_path = output_dir / "manifest.json"
    image_ids_path = output_dir / "image_ids.txt"
    question_ids_path = output_dir / "question_ids.txt"

    write_jsonl(jsonl_path, records)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    image_ids = sorted({record["image_id"] for record in records})
    question_ids = [
        turn["question_id"] for record in records for turn in record["turns"]
    ]
    image_ids_path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    question_ids_path.write_text("\n".join(question_ids) + "\n", encoding="utf-8")

    source_counts: OrderedDict[str, int] = OrderedDict()
    for question in questions.values():
        image_id = str(question["imageId"])
        source_counts[image_id] = source_counts.get(image_id, 0) + 1

    manifest = {
        "dataset": "MT-GQA unofficial reconstructed training subset",
        "source": "GQA train_balanced_questions.json",
        "source_questions": str(args.questions),
        "source_questions_sha256": sha256(args.questions),
        "source_question_count": len(questions),
        "source_image_count": len(source_counts),
        "candidate_dialog_count": len(candidates),
        "candidate_used_question_count": len(candidates) * 3,
        "candidate_dropped_question_count": sum(count % 3 for count in source_counts.values()),
        "selected_dialog_count": len(records),
        "selected_question_count": len(question_ids),
        "selected_image_count": len(image_ids),
        "selection": "random.sample over candidate dialogs, then sort by image_id/group index",
        "seed": args.seed,
        "grouping": "preserve source JSON order per image; consecutive triples; drop per-image remainder",
        "image_naming": "{image_id}.jpg",
        "gold_fields": ["answer", "full_answer"],
        "outputs": {
            "jsonl": jsonl_path.name,
            "json": json_path.name,
            "image_ids": image_ids_path.name,
            "question_ids": question_ids_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
