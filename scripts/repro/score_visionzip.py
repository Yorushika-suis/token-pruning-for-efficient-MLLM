#!/usr/bin/env python3
"""Score VisionZip LLaVA-1.5 merge.jsonl with official GQA / MME / POPE metrics.

Usage:
  python scripts/repro/score_visionzip.py --run-dir runs/visionzip/full_llava15_retain64
  python scripts/repro/score_visionzip.py --run-dir ... --tasks pope,mme

Writes:
  <run-dir>/metrics/{gqa,mme,pope}.json
  <run-dir>/metrics/summary.json
  and human-readable *.txt logs under the same directory.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "ThirdParty" / "LLaVA" / "playground" / "data" / "eval"
LLAVA_ROOT = ROOT / "ThirdParty" / "LLaVA"

MME_PERCEPTION = [
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
]
MME_COGNITION = [
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_gqa(merge: Path, eval_root: Path, out_dir: Path) -> dict:
    """Official testdev_balanced Accuracy (exact match after lowercasing).

    Full GQA eval.py also needs sceneGraphs + choices (~GB); LLaVA/VisionZip
    papers report Accuracy, which only needs questions + predictions.
    """
    questions_path = eval_root / "gqa" / "data" / "testdev_balanced_questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(
            f"missing {questions_path}; re-run download with scoring assets "
            "or extract testdev_balanced_questions.json from GQA questions1.2.zip"
        )
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    convert = LLAVA_ROOT / "scripts" / "convert_gqa_for_eval.py"
    pred_path = out_dir / "testdev_balanced_predictions.json"
    subprocess.check_call(
        [sys.executable, str(convert), "--src", str(merge), "--dst", str(pred_path)]
    )
    predictions = {
        str(p["questionId"]): str(p["prediction"]).strip().lower()
        for p in json.loads(pred_path.read_text(encoding="utf-8"))
    }

    scores: list[float] = []
    binary: list[float] = []
    open_q: list[float] = []
    missing = 0
    for qid, q in questions.items():
        if not q.get("isBalanced", True):
            continue
        gold = str(q["answer"]).strip().lower()
        pred = predictions.get(str(qid))
        if pred is None:
            missing += 1
            continue
        ok = float(pred == gold)
        scores.append(ok)
        # structural verify/logical => binary-ish; match GQA eval.py heuristic
        structural = (q.get("types") or {}).get("structural", "")
        if structural == "verify":
            binary.append(ok)
        else:
            open_q.append(ok)

    if missing:
        raise RuntimeError(f"GQA missing predictions for {missing} balanced questions")

    def avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    result = {
        "task": "gqa",
        "n": len(scores),
        "accuracy": avg(scores),
        "accuracy_pct": avg(scores) * 100.0,
        "binary_accuracy": avg(binary),
        "open_accuracy": avg(open_q),
        "predictions": str(pred_path),
    }
    (out_dir / "gqa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out_dir / "gqa.txt").write_text(
        f"GQA testdev_balanced Accuracy: {result['accuracy_pct']:.2f}% "
        f"(n={result['n']})\n",
        encoding="utf-8",
    )
    print(f"GQA Accuracy: {result['accuracy_pct']:.2f}% (n={result['n']})", flush=True)
    return result


def _parse_pred_ans(pred_ans: str) -> str:
    pred_ans = pred_ans.lower().strip()
    if pred_ans in {"yes", "no"}:
        return pred_ans
    prefix = pred_ans[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def _mme_task_score(lines: list[str]) -> tuple[float, dict]:
    """Port of BradyFU eval_tool/calculation.py without sklearn."""
    assert len(lines) % 2 == 0
    gts: list[str] = []
    preds: list[str] = []
    img_num = len(lines) // 2
    acc_plus_correct = 0
    for i in range(0, len(lines), 2):
        img_correct = 0
        for line in lines[i : i + 2]:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"bad MME line (need 4 cols): {line!r}")
            _, _, gt_ans, pred_ans = parts
            gt_ans = gt_ans.lower().strip()
            pred_ans = _parse_pred_ans(pred_ans)
            assert gt_ans in {"yes", "no"}
            gts.append(gt_ans)
            preds.append(pred_ans)
            if gt_ans == pred_ans:
                img_correct += 1
        if img_correct == 2:
            acc_plus_correct += 1

    label = {"yes": 1, "no": 0, "other": -1}
    y_true = [label[x] for x in gts]
    y_pred = [label[x] for x in preds]
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

    clean_t, clean_p = [], []
    other_num = 0
    for t, p in zip(y_true, y_pred):
        if p == -1:
            other_num += 1
            continue
        clean_t.append(t)
        clean_p.append(p)

    tp = fp = tn = fn = 0
    for t, p in zip(clean_t, clean_p):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    acc_plus = acc_plus_correct / img_num if img_num else 0.0
    # Official score per task = (acc + acc_plus) * 100
    task_score = (acc + acc_plus) * 100.0
    detail = {
        "acc": acc,
        "acc_plus": acc_plus,
        "precision": precision,
        "recall": recall,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "other_num": other_num,
        "score": task_score,
        "n_images": img_num,
    }
    return task_score, detail


def _ensure_mme_gt_txt(eval_root: Path) -> None:
    """Synthesize questions_answers_YN from ModelScope parquet answers if missing."""
    mme_root = eval_root / "MME" / "MME_Benchmark_release_version"
    sample = next(mme_root.glob("*/questions_answers_YN/*.txt"), None)
    if sample is not None:
        return
    cache = eval_root / "_download_cache"
    pqs = sorted(cache.glob("test-*-of-00004-*.parquet"))
    if not pqs:
        raise FileNotFoundError(
            f"no MME GT txt under {mme_root} and no parquet in {cache}"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow required to synthesize MME GT") from exc

    # group by (category, stem) -> list of (question, answer)
    buckets: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for path in pqs:
        table = pq.read_table(path, columns=["question_id", "question", "answer", "category"])
        for qid, question, answer, cat in zip(
            table.column("question_id").to_pylist(),
            table.column("question").to_pylist(),
            table.column("answer").to_pylist(),
            table.column("category").to_pylist(),
        ):
            rel = str(qid)  # e.g. existence/000000006040.jpg
            category = str(cat)
            stem = Path(rel).stem + ".txt"
            buckets[(category, stem)].append((str(question).strip(), str(answer).strip()))

    n = 0
    for (category, stem), pairs in buckets.items():
        qa_dir = mme_root / category / "questions_answers_YN"
        # Our download layout is flat images under category/; keep that and add YN dir.
        qa_dir.mkdir(parents=True, exist_ok=True)
        out = qa_dir / stem
        with out.open("w", encoding="utf-8") as f:
            for q, a in pairs:
                f.write(f"{q}\t{a}\n")
        n += 1
    print(f"synthesized MME questions_answers_YN ({n} files)", flush=True)


def score_mme(merge: Path, eval_root: Path, out_dir: Path) -> dict:
    _ensure_mme_gt_txt(eval_root)
    mme_dir = eval_root / "MME"
    exp = "visionzip_llava15_retain64"
    ans_dir = mme_dir / "answers"
    ans_dir.mkdir(parents=True, exist_ok=True)
    ans_file = ans_dir / f"{exp}.jsonl"
    ans_file.write_bytes(merge.read_bytes())

    # Build GT map + upload txt (same as convert_answer_to_mme.py)
    convert = mme_dir / "convert_answer_to_mme.py"
    if not convert.exists():
        raise FileNotFoundError(convert)
    subprocess.check_call(
        [sys.executable, str(convert), "--experiment", exp],
        cwd=str(mme_dir),
    )
    upload = mme_dir / "answers_upload" / exp
    # Also copy into eval_tool answers/ for people who run calculation.py by hand
    calc_ans = mme_dir / "eval_tool" / "answers" / exp
    calc_ans.mkdir(parents=True, exist_ok=True)
    for src in upload.glob("*.txt"):
        (calc_ans / src.name).write_bytes(src.read_bytes())

    perception_scores = {}
    cognition_scores = {}
    details = {}
    perception_total = 0.0
    cognition_total = 0.0
    for name in MME_PERCEPTION:
        lines = (upload / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        score, detail = _mme_task_score(lines)
        perception_scores[name] = score
        details[name] = detail
        perception_total += score
    for name in MME_COGNITION:
        lines = (upload / f"{name}.txt").read_text(encoding="utf-8").splitlines()
        score, detail = _mme_task_score(lines)
        cognition_scores[name] = score
        details[name] = detail
        cognition_total += score

    result = {
        "task": "mme",
        "perception": perception_total,
        "cognition": cognition_total,
        "total": perception_total + cognition_total,
        "perception_tasks": perception_scores,
        "cognition_tasks": cognition_scores,
        "details": details,
        "upload_dir": str(upload),
    }
    (out_dir / "mme.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    txt = (
        f"MME Perception: {perception_total:.2f}\n"
        f"MME Cognition:  {cognition_total:.2f}\n"
        f"MME Total:      {result['total']:.2f}\n"
    )
    (out_dir / "mme.txt").write_text(txt, encoding="utf-8")
    print(txt, end="", flush=True)
    return result


def _eval_pope_answers(answers: list[dict], label_file: Path) -> dict:
    """Same metric math as ThirdParty/LLaVA/llava/eval/eval_pope.py."""
    label_list = [json.loads(q)["label"] for q in label_file.open(encoding="utf-8")]
    preds = []
    for answer in answers:
        text = answer["text"]
        if "." in text:
            text = text.split(".")[0]
        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            preds.append(0)
        else:
            preds.append(1)
    labels = [0 if x == "no" else 1 for x in label_list]
    assert len(preds) == len(labels)
    tp = fp = tn = fn = 0
    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1
    precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    yes_ratio = preds.count(1) / len(preds)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "n": len(preds),
    }


def score_pope(merge: Path, eval_root: Path, out_dir: Path) -> dict:
    """POPE metrics (F1 / Acc / …) per random/popular/adversarial.

    Official eval_pope.py keys questions by question_id only, but POPE reuses
    ids 1..3000 across the three splits. Match answers via (question_id + prompt).
    """
    ann_dir = eval_root / "pope" / "coco"
    answers = _load_jsonl(merge)
    # index: question_id -> list of answers (one per category)
    by_qid: dict = defaultdict(list)
    for a in answers:
        by_qid[a["question_id"]].append(a)

    categories: dict[str, dict] = {}
    log_lines: list[str] = []
    for path in sorted(ann_dir.glob("coco_pope_*.json")):
        category = path.name[len("coco_pope_") : -len(".json")]
        labels = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        cur_answers = []
        for lab in labels:
            qid = lab["question_id"]
            stem = lab["text"].strip()
            cands = by_qid.get(qid, [])
            hit = next((a for a in cands if stem in a.get("prompt", "")), None)
            if hit is None:
                raise RuntimeError(
                    f"POPE {category}: no answer for question_id={qid} text={stem!r}"
                )
            cur_answers.append(hit)
        metrics = _eval_pope_answers(cur_answers, path)
        categories[category] = metrics
        log_lines.append(f"Category: {category}, # samples: {metrics['n']}")
        log_lines.append(
            f"Accuracy: {metrics['accuracy']}\n"
            f"Precision: {metrics['precision']}\n"
            f"Recall: {metrics['recall']}\n"
            f"F1 score: {metrics['f1']}\n"
            f"Yes ratio: {metrics['yes_ratio']}"
        )
        log_lines.append("====================================")

    f1s = [v["f1"] for v in categories.values()]
    accs = [v["accuracy"] for v in categories.values()]
    result = {
        "task": "pope",
        "categories": categories,
        "mean_f1": float(sum(f1s) / len(f1s)) if f1s else None,
        "mean_accuracy": float(sum(accs) / len(accs)) if accs else None,
        "mean_accuracy_pct": (float(sum(accs) / len(accs)) * 100.0) if accs else None,
    }
    (out_dir / "pope.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out_dir / "pope_eval_stdout.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    lines = [
        f"POPE mean Accuracy: {result['mean_accuracy_pct']:.2f}%",
        f"POPE mean F1: {result['mean_f1']:.4f}",
    ]
    for cat, m in categories.items():
        lines.append(f"  {cat}: Acc={m['accuracy']*100:.2f}% F1={m['f1']:.4f}")
    text = "\n".join(lines) + "\n"
    (out_dir / "pope.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path(os.environ.get("LLAVA_EVAL_ROOT", str(DEFAULT_EVAL))),
    )
    p.add_argument("--tasks", default="gqa,mme,pope")
    args = p.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    eval_root = args.eval_root.expanduser().resolve()
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"run_dir": str(run_dir), "tasks": {}}
    failed = 0
    for task in tasks:
        merge = run_dir / "answers" / task / "merge.jsonl"
        if not merge.exists():
            print(f"SKIP {task}: missing {merge}", flush=True)
            failed += 1
            continue
        try:
            if task == "gqa":
                summary["tasks"]["gqa"] = score_gqa(merge, eval_root, out_dir)
            elif task == "mme":
                summary["tasks"]["mme"] = score_mme(merge, eval_root, out_dir)
            elif task == "pope":
                summary["tasks"]["pope"] = score_pope(merge, eval_root, out_dir)
            else:
                print(f"unknown task {task}", flush=True)
                failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {task}: {exc}", flush=True)
            failed += 1

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"metrics → {out_dir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
