#!/usr/bin/env python3
"""Build the deterministic three-turn MT-GQA testdev reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


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
        raise ValueError("GQA questions must be a JSON object")
    return questions


def build_dialogs(questions: OrderedDict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_image: OrderedDict[str, list[tuple[str, dict[str, Any]]]] = OrderedDict()
    for question_id, question in questions.items():
        image_id = str(question["imageId"])
        by_image.setdefault(image_id, []).append((str(question_id), question))

    dialogs: list[dict[str, Any]] = []
    dropped = 0
    for image_id, image_questions in by_image.items():
        usable = len(image_questions) - len(image_questions) % 3
        dropped += len(image_questions) - usable
        for group_index in range(usable // 3):
            group = image_questions[group_index * 3 : group_index * 3 + 3]
            dialogs.append(
                {
                    "dialog_id": f"gqa_testdev_balanced_{len(dialogs):06d}",
                    "source_split": "testdev_balanced",
                    "image_id": image_id,
                    "image": f"{image_id}.jpg",
                    "source_group_index": group_index,
                    "turns": [
                        {
                            "question_id": question_id,
                            "question": question["question"],
                            "answer": question.get("answer"),
                            "full_answer": question.get("fullAnswer"),
                        }
                        for question_id, question in group
                    ],
                }
            )
    return dialogs, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    dialogs, dropped = build_dialogs(questions)
    image_ids = {dialog["image_id"] for dialog in dialogs}
    missing = sorted(
        image_id for image_id in image_ids
        if not (args.image_root / f"{image_id}.jpg").is_file()
    )
    question_ids = [
        turn["question_id"] for dialog in dialogs for turn in dialog["turns"]
    ]
    if missing:
        raise SystemExit(f"missing {len(missing)} images; first IDs: {missing[:10]}")
    if len(image_ids) != 398 or len(dialogs) != 4061 or len(question_ids) != 12183:
        raise SystemExit(
            f"unexpected reconstruction: images={len(image_ids)}, dialogs={len(dialogs)}, "
            f"turns={len(question_ids)}"
        )
    if len(set(question_ids)) != len(question_ids):
        raise SystemExit("question IDs are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mt_gqa_testdev_balanced_4061.json"
    jsonl_path = args.output_dir / "mt_gqa_testdev_balanced_4061.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    image_ids_path = args.output_dir / "image_ids.txt"
    question_ids_path = args.output_dir / "question_ids.txt"
    json_path.write_text(json.dumps(dialogs, ensure_ascii=True, indent=2) + "\n")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for dialog in dialogs:
            handle.write(json.dumps(dialog, ensure_ascii=True) + "\n")
    image_ids_path.write_text("\n".join(sorted(image_ids)) + "\n")
    question_ids_path.write_text("\n".join(question_ids) + "\n")
    manifest = {
        "dataset": "MT-GQA unofficial reconstructed testdev",
        "source_questions": str(args.questions),
        "source_questions_sha256": sha256(args.questions),
        "source_question_count": len(questions),
        "image_root": str(args.image_root),
        "image_count": len(image_ids),
        "dialog_count": len(dialogs),
        "turn_count": len(question_ids),
        "dropped_question_count": dropped,
        "turns_per_dialog": 3,
        "grouping": "preserve source JSON order per image; consecutive triples; drop remainder",
        "outputs": {
            "json": json_path.name,
            "jsonl": jsonl_path.name,
            "image_ids": image_ids_path.name,
            "question_ids": question_ids_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
