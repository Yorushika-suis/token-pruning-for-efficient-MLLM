#!/usr/bin/env python3
"""Rescore saved Video-MME sample logs on the held-out video ids.

Does not rerun FastVID or VidCom2. Overall and duration scores are the
fraction of questions whose predicted letter matches the answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_ids(path: Path, key: str) -> set[str]:
    payload = json.loads(path.read_text())
    return set(payload[key])


def score_file(path: Path, heldout: set[str]) -> dict:
    groups = {"overall": [0, 0], "short": [0, 0], "medium": [0, 0], "long": [0, 0]}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            doc = row.get("doc") or {}
            video_id = str(doc.get("videoID") or "")
            if video_id not in heldout:
                continue
            item = row.get("videomme_perception_score") or {}
            pred = str(item.get("pred_answer") or "")
            answer = str(item.get("answer") or "")
            duration = str(item.get("duration") or doc.get("duration") or "")
            hit = int(pred == answer and answer != "")
            groups["overall"][0] += hit
            groups["overall"][1] += 1
            if duration in groups:
                groups[duration][0] += hit
                groups[duration][1] += 1
    out = {"file": str(path)}
    for name, (correct, total) in groups.items():
        out[name] = None if total == 0 else 100.0 * correct / total
        out[f"{name}_n"] = total
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--logs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    heldout = load_ids(args.split, "heldout_video_ids")
    rows = [score_file(path, heldout) for path in args.logs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(
            f"{row['file']} n={row['overall_n']} overall={row['overall']:.2f} "
            f"short={row['short']:.2f} medium={row['medium']:.2f} long={row['long']:.2f}"
        )


if __name__ == "__main__":
    main()
