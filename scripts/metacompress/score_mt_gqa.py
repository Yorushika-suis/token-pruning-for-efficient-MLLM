#!/usr/bin/env python3
"""Score MT-GQA predictions turn by turn using GQA short-answer exact match."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    return str(value or "").strip().rstrip(".").lower()


def load_predictions(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    predictions: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            records = (json.loads(line) for line in handle if line.strip())
        else:
            records = iter(json.load(handle))
        for record in records:
            dialog_id = str(record["dialog_id"])
            turn_index = int(record["turn_index"])
            if turn_index not in (1, 2, 3):
                raise ValueError(f"turn_index must be 1, 2, or 3: {record}")
            if turn_index in predictions[dialog_id]:
                raise ValueError(f"duplicate prediction: {dialog_id} turn {turn_index}")
            predictions[dialog_id][turn_index] = record
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--expected-dialogs",
        type=int,
        default=4061,
        help="Expected number of dialogs; use the smoke-test limit instead of 4061 for partial runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dialogs = json.loads(args.dialogs.read_text(encoding="utf-8"))
    if args.limit is not None:
        dialogs = dialogs[: args.limit]
    predictions = load_predictions(args.predictions)
    expected_ids = {str(dialog["dialog_id"]) for dialog in dialogs}
    extra_ids = sorted(set(predictions) - expected_ids)
    if extra_ids:
        raise SystemExit(f"prediction file contains dialogs outside the selected set: {extra_ids[:5]}")
    totals = [0, 0, 0]
    correct = [0, 0, 0]
    scored: list[dict[str, Any]] = []
    for dialog in dialogs:
        dialog_id = str(dialog["dialog_id"])
        if dialog_id not in predictions:
            raise SystemExit(f"missing dialog prediction: {dialog_id}")
        for index, turn in enumerate(dialog["turns"], start=1):
            record = predictions[dialog_id].get(index)
            if record is None:
                raise SystemExit(f"missing prediction: {dialog_id} turn {index}")
            predicted = str(record.get("prediction", record.get("text", "")))
            gold = normalize(turn.get("answer"))
            pred = normalize(predicted)
            is_correct = pred == gold
            totals[index - 1] += 1
            correct[index - 1] += int(is_correct)
            scored.append(
                {
                    "dialog_id": dialog_id,
                    "question_id": str(turn["question_id"]),
                    "turn_index": index,
                    "prediction": predicted,
                    "normalized_prediction": pred,
                    "gold": turn.get("answer"),
                    "normalized_gold": gold,
                    "correct": is_correct,
                }
            )
    expected_turns = [args.expected_dialogs] * 3
    if totals != expected_turns:
        raise SystemExit(f"expected {args.expected_dialogs} examples per turn, got {totals}")
    accuracies = [100.0 * c / t for c, t in zip(correct, totals)]
    metrics = {
        "dialog_count": len(dialogs),
        "turn_count": sum(totals),
        "correct": {f"acc{i + 1}": correct[i] for i in range(3)},
        "total": {f"acc{i + 1}": totals[i] for i in range(3)},
        "acc1": accuracies[0],
        "acc2": accuracies[1],
        "acc3": accuracies[2],
        "avg": sum(accuracies) / 3.0,
        "normalization": "strip, rstrip('.'), lower",
        "scored_predictions": str(args.output.with_suffix(".scored.jsonl")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=True, indent=2) + "\n")
    with args.output.with_suffix(".scored.jsonl").open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
