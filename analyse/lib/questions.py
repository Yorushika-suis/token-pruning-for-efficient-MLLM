"""Look up the text question for sampled keys (Video-MME parquet; image jsonl)."""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.lib.paths import hf_home, llava_eval_root, normalize_key


def _as_list(val) -> list[str]:
    if val is None:
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return [str(val)]


@lru_cache(maxsize=1)
def _videomme_index() -> dict[str, dict]:
    import pyarrow.parquet as pq

    base = hf_home() / "hub" / "datasets--lmms-lab--Video-MME" / "snapshots"
    hits = sorted(base.glob("*/videomme/*.parquet")) if base.exists() else []
    if not hits:
        return {}
    table = pq.read_table(hits[0])
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = len(next(iter(cols.values())))
    out: dict[str, dict] = {}
    for i in range(n):
        row = {k: cols[k][i] for k in cols}
        vid = str(row.get("videoID") or row.get("video_id") or "")
        qid = str(row.get("question_id") or "")
        rec = _videomme_record(row)
        if vid and qid:
            out[f"{vid}__{qid}"] = rec
        if qid:
            out.setdefault(qid, rec)
    return out


def _videomme_record(row: dict) -> dict:
    options = _as_list(row.get("options"))
    question = str(row.get("question") or "").strip()
    return {
        "key": f"{row.get('videoID')}__{row.get('question_id')}",
        "videoID": str(row.get("videoID") or ""),
        "question_id": str(row.get("question_id") or ""),
        "question": question,
        "options": options,
        "answer": str(row.get("answer") or ""),
        "task_type": str(row.get("task_type") or ""),
        "domain": str(row.get("domain") or ""),
        "sub_category": str(row.get("sub_category") or ""),
        "duration": str(row.get("duration") or ""),
        "prompt": _mcq_prompt(question, options),
    }


def _mcq_prompt(question: str, options: list[str]) -> str:
    lines = [question]
    lines.extend(options)
    return "\n".join(x for x in lines if x)


def _jsonl_text(path: Path, key: str) -> dict | None:
    if not path.is_file():
        return None
    stem = normalize_key(key)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or row.get("id") or "")
            aliases = {qid, qid.replace("/", "_"), stem}
            if stem.isdigit():
                aliases.add(str(int(stem)))
            if qid not in aliases and stem not in aliases:
                continue
            text = str(row.get("text") or row.get("question") or "").strip()
            return {
                "key": stem,
                "question_id": qid,
                "question": text,
                "options": [],
                "answer": str(row.get("answer") or ""),
                "prompt": text,
            }
    return None


def lookup_question(dataset: str, key: str) -> dict | None:
    ds = dataset.lower()
    stem = normalize_key(key)
    if ds == "videomme":
        idx = _videomme_index()
        return idx.get(stem) or idx.get(stem.split("__", 1)[-1] if "__" in stem else stem)
    root = llava_eval_root()
    if ds == "gqa":
        return _jsonl_text(root / "gqa" / "llava_gqa_testdev_balanced.jsonl", stem)
    if ds == "mme":
        return _jsonl_text(root / "MME" / "llava_mme.jsonl", stem)
    if ds == "pope":
        return _jsonl_text(root / "pope" / "llava_pope_test.jsonl", stem)
    return None


def lookup_questions(dataset: str, keys: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in keys:
        rec = lookup_question(dataset, key)
        if rec is not None:
            out[normalize_key(key)] = rec
    return out


def format_question_block(key: str, rec: dict | None) -> str:
    if rec is None:
        return f"### {key}\n(missing question text)\n"
    bits = []
    meta = [rec.get("task_type"), rec.get("domain"), rec.get("sub_category"), rec.get("duration")]
    meta = [m for m in meta if m]
    bits.append(f"### {key}")
    if meta:
        bits.append(" | ".join(meta))
    bits.append(rec.get("prompt") or rec.get("question") or "")
    ans = rec.get("answer")
    if ans:
        bits.append(f"answer: {ans}")
    return "\n".join(bits) + "\n"


def write_questions(dest_dir: Path, dataset: str, keys: list[str]) -> tuple[Path, Path, dict[str, dict]]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    recs = lookup_questions(dataset, keys)
    blocks = [format_question_block(normalize_key(k), recs.get(normalize_key(k))) for k in keys]
    txt = dest_dir / "questions.txt"
    js = dest_dir / "jsons" / "questions.json"
    txt.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")
    js.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "keys": [normalize_key(k) for k in keys],
        "questions": recs,
    }
    js.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return txt, js, recs


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="videomme")
    p.add_argument("--key", default=None)
    p.add_argument("--keys-file", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if args.key:
        keys = [args.key]
    elif args.keys_file:
        from analyse.lib.keys import read_keys_file

        keys = read_keys_file(args.keys_file)
    else:
        raise SystemExit("need --key or --keys-file")
    recs = lookup_questions(args.dataset, keys)
    for k in keys:
        kn = normalize_key(k)
        print(format_question_block(kn, recs.get(kn)).rstrip())
        print()
    if args.out_dir is not None:
        txt, js, _ = write_questions(args.out_dir, args.dataset, keys)
        print(f"wrote {txt}")
        print(f"wrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
