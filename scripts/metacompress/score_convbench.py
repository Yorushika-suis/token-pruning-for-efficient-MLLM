#!/usr/bin/env python3
"""Score ConvBench generations with the official pairwise judge prompts.

The judge is an OpenAI-compatible chat API. The default endpoint is
https://mintcn.macaron.xin and the default model is glm-5.2. The API key is
read from CONV_BENCH_API_KEY and is never stored in the repository. Prompt
text and A/B order come from the official ConvBenchEval scripts. S1/S2/S3 are
win rates against the human reference, multiplied by 100. The paper denominator
is 577. Dialogs listed in convbench_judge_blacklist.json are not judged and are
removed from that denominator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ThirdParty" / "ConvBenchEval"))

import prompts  # noqa: E402


TURN_PROMPTS = {
    "first": (
        prompts._PROMPT_SYSTEM_PAIRWISE_V1_MT_first_turn,
        prompts._PROMPT_USER_PAIRWISE_V1_MT_first_turn,
        prompts._PROMPT_ASSISTANT_PAIRWISE_V1_MT_first_turn,
        "first turn",
    ),
    "second": (
        prompts._PROMPT_SYSTEM_PAIRWISE_V1_MT_second_turn,
        prompts._PROMPT_USER_PAIRWISE_V1_MT_second_turn,
        prompts._PROMPT_ASSISTANT_PAIRWISE_V1_MT_second_turn,
        "second turn",
    ),
    "third": (
        prompts._PROMPT_SYSTEM_PAIRWISE_V1_MT_third_turn,
        prompts._PROMPT_USER_PAIRWISE_V1_MT_third_turn,
        prompts._PROMPT_ASSISTANT_PAIRWISE_V1_MT_third_turn,
        "third turn",
    ),
}

PAPER_DENOMINATOR = 577
DEFAULT_BLACKLIST = Path(__file__).resolve().parent / "convbench_judge_blacklist.json"


def conversation_block(
    caption: str,
    questions: list[str],
    answers_a: list[str],
    answers_b: list[str],
    turn_name: str,
    demands: str | None = None,
    prior: tuple[str, str, str] | None = None,
) -> str:
    """Fill the official user message. A and B are already ordered."""
    def side(label: str, answers: list[str]) -> str:
        return (
            f"<|The Start of Assistant {label}’s Conversation with User|>\n"
            f"### The first turn question from user:\n{questions[0]}\n\n"
            f"### The first turn response from Assistant {label}:\n{answers[0]}\n\n"
            f"### The second turn question from user:\n{questions[1]}\n\n"
            f"### The second turn response from Assistant {label}:\n{answers[1]}\n\n"
            f"### The third question from user:\n{questions[2]}\n\n"
            f"### The third turn response from Assistant {label}:\n{answers[2]}\n\n"
            f"<|The End of Assistant {label}’s Conversation with User|>"
        )

    body = (
        "OK. Here are the image, the image description, the instructions, the high-quality "
        f"references, and the responses.\nImage context: {caption}\n\n"
        f"{side('A', answers_a)}\n\n{side('B', answers_b)}\n\n"
    )
    if demands is not None:
        body += (
            "There are some concerns which you should focus when make your judgements "
            f"for the response:{demands}.\n\n"
        )
    if prior is not None:
        body += (
            f"The first turn evaluation: {prior[0]}.\n"
            f" The second turn evaluation: {prior[1]}.\n"
            f" The third turn evaluation: {prior[2]}.\n\n"
        )
    body += (
        f'Think step-by-step, compare the {turn_name} responses from the two assistants, '
        'and finish your response with "Overall, Response X is better." where X is either A or B.'
    )
    return body


def messages_for(turn: str, caption: str, questions: list[str], answers_a: list[str], answers_b: list[str], demands: str, prior=None):
    system_p, user_p, assistant_p, turn_name = TURN_PROMPTS.get(turn, (None, None, None, "overall conversation"))
    if turn == "overall":
        system_p = prompts._PROMPT_SYSTEM_PAIRWISE_V1_MT_overall_conversation
        user_p = prompts._PROMPT_USER_PAIRWISE_V1_MT_overall_conversation
        assistant_p = prompts._PROMPT_ASSISTANT_PAIRWISE_V1_MT_overall_conversation
    user_content = conversation_block(
        caption, questions, answers_a, answers_b, turn_name,
        demands if turn == "third" else None,
        prior if turn == "overall" else None,
    )
    return [
        {"role": "system", "content": system_p},
        {"role": "user", "content": user_p},
        {"role": "assistant", "content": assistant_p},
        {"role": "user", "content": user_content},
    ]


def extract_choice(text: str) -> str | None:
    import re
    matches = re.findall(r"overall,\s*response\s+([ab])\s+is better", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    selected = {
        letter: int(f"response {letter.lower()} is better" in text.lower())
        for letter in "AB"
    }
    winners = [letter for letter, flag in selected.items() if flag]
    if len(winners) == 1:
        return winners[0]
    return None


def model_wins(choice: str | None, position: int) -> bool:
    if position == 0:
        return choice == "A"
    if position == 1:
        return choice == "B"
    raise ValueError(f"unexpected pairwise position {position}")


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines. Existing environment variables stay in place."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_blacklist(path: Path) -> set[int]:
    """Load dialog ids that must be skipped and left out of the score."""
    if not path.is_file():
        raise SystemExit(f"judge blacklist not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    ids = {int(case["convbench_id"]) for case in cases}
    invalid = sorted(convbench_id for convbench_id in ids if convbench_id < 1 or convbench_id > 578)
    if invalid:
        raise SystemExit(f"judge blacklist has ids outside 1..578: {invalid}")
    return ids


def aggregate_scores(records: list[dict], blacklist: set[int]) -> tuple[dict[str, int], int, int]:
    """Count wins on judged dialogs. Blacklisted ids are not wins and not losses."""
    wins = {"s1": 0, "s2": 0, "s3": 0}
    judged = 0
    for record in records:
        convbench_id = int(record["convbench_id"])
        if record.get("skipped") or convbench_id in blacklist:
            continue
        judged += 1
        for key, name in (("s1", "first"), ("s2", "second"), ("s3", "third")):
            wins[key] += int(model_wins(record["choices"][name], record["position"]))
    denominator = PAPER_DENOMINATOR - len(blacklist)
    return wins, judged, denominator


def load_predictions(path: Path) -> dict[str, dict[int, str]]:
    grouped: dict[str, dict[int, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            grouped.setdefault(record["dialog_id"], {})[int(record["turn_index"])] = record["prediction"]
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pairwise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="https://mintcn.macaron.xin/v1")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--blacklist", type=Path, default=DEFAULT_BLACKLIST)
    return parser.parse_args()


def call_judge(
    api_base: str,
    api_key: str,
    model: str,
    chat: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    retries: int = 30,
    retry_interval: float = 10.0,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {"model": model, "messages": chat, "temperature": 0, "max_tokens": max_tokens}
    ).encode()
    retryable = {408, 429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode())
            message = body["choices"][0]["message"]
            content = message.get("content") or ""
            if not content.strip():
                content = message.get("reasoning_content") or ""
            if not content.strip():
                raise KeyError("judge response has empty content and reasoning_content")
            return content
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            last_error = error
            if error.code not in retryable or attempt == retries:
                raise SystemExit(f"judge API HTTP {error.code}: {detail}") from error
            print(f"judge HTTP {error.code}, retry {attempt}/{retries} in {retry_interval:.0f}s", flush=True)
        except (urllib.error.URLError, TimeoutError, KeyError) as error:
            last_error = error
            if attempt == retries:
                raise SystemExit(f"judge API failed after {retries} attempts: {error}") from error
            print(f"judge error {error}, retry {attempt}/{retries} in {retry_interval:.0f}s", flush=True)
        time.sleep(retry_interval)
    raise SystemExit(f"judge API failed: {last_error}")


def main() -> None:
    args = parse_args()
    load_env_file(ROOT / ".env")
    api_key = os.environ.get("CONV_BENCH_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Put CONV_BENCH_API_KEY in token-pruning-for-efficient-MLLM/.env. "
            "The key is not written into the repository."
        )

    dialogs = json.loads(args.dialogs.read_text(encoding="utf-8"))
    by_id = {int(dialog["convbench_id"]): dialog for dialog in dialogs}
    predictions = load_predictions(args.predictions)
    pairwise = np.load(args.pairwise)
    if len(pairwise) != 578:
        raise SystemExit(f"pairwise order must have 578 entries, got {len(pairwise)}")
    blacklist = load_blacklist(args.blacklist)

    def judge(chat: list[dict[str, str]]) -> str:
        return call_judge(args.api_base, api_key, args.model, chat, args.max_tokens, args.timeout)

    def score_one(convbench_id: int) -> dict:
        position = int(pairwise[convbench_id - 1])
        dialog = by_id.get(convbench_id)
        blacklisted = convbench_id in blacklist
        record = {
            "convbench_id": convbench_id,
            "position": position,
            "skipped": dialog is None or blacklisted,
            "skip_reason": "blacklist" if blacklisted else ("missing_image" if dialog is None else None),
        }
        if record["skipped"]:
            return record
        answers = [predictions[dialog["dialog_id"]][turn] for turn in (1, 2, 3)]
        references = [turn["answer"] for turn in dialog["turns"]]
        questions = [turn["question"] for turn in dialog["turns"]]
        if position == 0:
            answers_a, answers_b = answers, references
        else:
            answers_a, answers_b = references, answers
        turn_text = {}
        choices = {}
        for name in ("first", "second", "third"):
            text = judge(messages_for(
                name, dialog["image_description"], questions, answers_a, answers_b,
                dialog["third_turn_demands"],
            ))
            turn_text[name] = text
            choices[name] = extract_choice(text)
        overall_text = judge(messages_for(
            "overall", dialog["image_description"], questions, answers_a, answers_b,
            dialog["third_turn_demands"],
            prior=(turn_text["first"], turn_text["second"], turn_text["third"]),
        ))
        choices["overall"] = extract_choice(overall_text)
        record.update({"choices": choices, "evaluations": {**turn_text, "overall": overall_text}})
        return record

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output.with_suffix(".details.jsonl")
    write_lock = threading.Lock()
    records: list[dict] = []
    with detail_path.open("w", encoding="utf-8") as detail:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(score_one, convbench_id) for convbench_id in range(1, 579)]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                with write_lock:
                    detail.write(json.dumps(record, ensure_ascii=True) + "\n")
                    detail.flush()

    wins, judged, denominator = aggregate_scores(records, blacklist)
    scores = {key: 100.0 * value / denominator for key, value in wins.items()}
    metrics = {
        "judge": args.api_base,
        "judge_name": args.model,
        "prompt_source": "ConvBenchEval official pairwise prompts",
        "denominator": denominator,
        "paper_denominator": PAPER_DENOMINATOR,
        "blacklist": str(args.blacklist),
        "excluded_ids": sorted(blacklist),
        "judged_dialogs": judged,
        "workers": args.workers,
        "wins": wins,
        "s1": scores["s1"],
        "s2": scores["s2"],
        "s3": scores["s3"],
        "avg": sum(scores.values()) / 3.0,
        "paper_base": [4.33, 5.72, 5.55, 5.20],
        "paper_sample": [3.64, 4.85, 3.81, 4.10],
        "details": str(detail_path),
    }
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
