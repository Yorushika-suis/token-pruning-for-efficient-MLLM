#!/usr/bin/env python3
"""Download GQA / MME / POPE assets for LLaVA-1.5 / VisionZip eval.

Layout (matches scripts/repro/full_visionzip_llava15.sh):

  ThirdParty/LLaVA/playground/data/eval/
    gqa/llava_gqa_testdev_balanced.jsonl
    gqa/data/images/{id}.jpg
    MME/llava_mme.jsonl
    MME/MME_Benchmark_release_version/{category}/...
    pope/llava_pope_test.jsonl
    pope/val2014/...
    pope/coco/coco_pope_{random,popular,adversarial}.json

This cluster cannot use huggingface.co; GQA/MME images+QA come from
ModelScope (lmms-lab/* parquets). COCO val2014 from pjreddie mirror.
POPE label files from GitHub (AoiDragon/POPE @ e3e3926).

Optional: --eval-zip tries Google Drive LLaVA eval.zip via gdown for official
jsonl / MME eval_tool; synthesized question files are used as fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "ThirdParty" / "LLaVA" / "playground" / "data" / "eval"
MS = "https://www.modelscope.cn"

GQA_IMAGE_PQ = (
    f"{MS}/datasets/lmms-lab/GQA/resolve/master/"
    "testdev_balanced_images/testdev-00000-of-00001.parquet"
)
GQA_INST_PQ = (
    f"{MS}/datasets/lmms-lab/GQA/resolve/master/"
    "testdev_balanced_instructions/testdev-00000-of-00001.parquet"
)
MME_PQS = [
    f"{MS}/datasets/lmms-lab/MME/resolve/master/data/"
    "test-00000-of-00004-a25dbe3b44c4fda6.parquet",
    f"{MS}/datasets/lmms-lab/MME/resolve/master/data/"
    "test-00001-of-00004-7d22c7f1aba6fca4.parquet",
    f"{MS}/datasets/lmms-lab/MME/resolve/master/data/"
    "test-00002-of-00004-594798fd3f5b029c.parquet",
    f"{MS}/datasets/lmms-lab/MME/resolve/master/data/"
    "test-00003-of-00004-53ae1794f93b1e35.parquet",
]
COCO_VAL2014 = "https://data.pjreddie.com/files/val2014.zip"
POPE_BASE = (
    "https://raw.githubusercontent.com/AoiDragon/POPE/"
    "e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco"
)
POPE_FILES = [
    "coco_pope_random.json",
    "coco_pope_popular.json",
    "coco_pope_adversarial.json",
]
EVAL_ZIP_GDRIVE_ID = "1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy"
GQA_PROMPT = "\nAnswer the question using a single word or phrase."
POPE_PROMPT = "\nAnswer the question using a single word or phrase."


def _curl(url: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(retries),
        "--retry-delay",
        "2",
        "-A",
        "token-pruning-llava-eval/1.0",
        "-C",
        "-",
        "-o",
        str(tmp),
        url,
    ]
    print(f"GET {url}\n -> {dest}", flush=True)
    subprocess.check_call(cmd)
    tmp.replace(dest)


def _need_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401

        return pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required. e.g. "
            "`/opt/miniconda3/envs/llava/bin/python -m pip install pyarrow` "
            f"or run with the fastvid env.\n{exc}"
        ) from exc


def _image_bytes(cell) -> bytes:
    if cell is None:
        raise ValueError("empty image cell")
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b is None:
            raise ValueError(f"image dict missing bytes: keys={list(cell)}")
        return bytes(b)
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unsupported image cell type: {type(cell)}")


def download_eval_zip(eval_root: Path, cache: Path) -> bool:
    """Best-effort official LLaVA eval.zip (annotations + eval_tool)."""
    zip_path = cache / "llava_eval.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        try:
            import gdown
        except ImportError:
            print("skip eval.zip: gdown not installed", flush=True)
            return False
        url = f"https://drive.google.com/uc?id={EVAL_ZIP_GDRIVE_ID}"
        print(f"gdown {url}", flush=True)
        try:
            gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
        except Exception as exc:  # noqa: BLE001
            print(f"eval.zip download failed: {exc}", flush=True)
            return False
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        print("eval.zip missing or too small", flush=True)
        return False
    print(f"extract {zip_path} → {eval_root}", flush=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(eval_root)
    # Some dumps nest under eval/
    nested = eval_root / "eval"
    if nested.is_dir() and not (eval_root / "gqa").exists():
        for child in nested.iterdir():
            dest = eval_root / child.name
            if dest.exists():
                continue
            shutil.move(str(child), str(dest))
    return True


def prepare_gqa(eval_root: Path, cache: Path) -> None:
    pq = _need_pyarrow()
    gqa = eval_root / "gqa"
    img_dir = gqa / "data" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_pq = cache / "gqa_testdev_balanced_images.parquet"
    inst_pq = cache / "gqa_testdev_balanced_instructions.parquet"
    if not img_pq.exists():
        _curl(GQA_IMAGE_PQ, img_pq)
    if not inst_pq.exists():
        _curl(GQA_INST_PQ, inst_pq)

    print("extracting GQA images…", flush=True)
    table = pq.read_table(img_pq)
    ids = table.column("id").to_pylist()
    images = table.column("image").to_pylist()
    for i, (iid, im) in enumerate(zip(ids, images)):
        out = img_dir / f"{iid}.jpg"
        if out.exists() and out.stat().st_size > 0:
            continue
        out.write_bytes(_image_bytes(im))
        if (i + 1) % 100 == 0:
            print(f"  images {i + 1}/{len(ids)}", flush=True)
    print(f"  images done ({len(list(img_dir.glob('*.jpg')))} files)", flush=True)

    jsonl = gqa / "llava_gqa_testdev_balanced.jsonl"
    if not jsonl.exists():
        print("writing GQA question jsonl…", flush=True)
        inst = pq.read_table(inst_pq)
        qids = inst.column("id").to_pylist()
        image_ids = inst.column("imageId").to_pylist()
        questions = inst.column("question").to_pylist()
        with jsonl.open("w", encoding="utf-8") as f:
            for qid, iid, q in zip(qids, image_ids, questions):
                f.write(
                    json.dumps(
                        {
                            "question_id": qid,
                            "image": f"{iid}.jpg",
                            "text": f"{q}{GQA_PROMPT}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  wrote {jsonl} ({len(qids)} lines)", flush=True)
    else:
        print(f"keep existing {jsonl}", flush=True)


def prepare_mme(eval_root: Path, cache: Path) -> None:
    pq = _need_pyarrow()
    mme = eval_root / "MME"
    img_root = mme / "MME_Benchmark_release_version"
    img_root.mkdir(parents=True, exist_ok=True)
    local_pqs = []
    for url in MME_PQS:
        name = url.rsplit("/", 1)[-1]
        dest = cache / name
        if not dest.exists():
            _curl(url, dest)
        local_pqs.append(dest)

    jsonl = mme / "llava_mme.jsonl"
    seen_img: set[str] = set()
    rows_out: list[dict] = []
    print("extracting MME images + questions…", flush=True)
    for path in local_pqs:
        table = pq.read_table(path)
        qids = table.column("question_id").to_pylist()
        questions = table.column("question").to_pylist()
        categories = table.column("category").to_pylist()
        images = table.column("image").to_pylist()
        for qid, question, cat, im in zip(qids, questions, categories, images):
            # question_id like "code_reasoning/0020.png"
            rel = str(qid)
            out_img = img_root / rel
            out_img.parent.mkdir(parents=True, exist_ok=True)
            if rel not in seen_img:
                if not out_img.exists() or out_img.stat().st_size == 0:
                    out_img.write_bytes(_image_bytes(im))
                seen_img.add(rel)
            rows_out.append(
                {
                    "question_id": rel,
                    "image": rel,
                    "text": question,
                    "category": cat,
                }
            )
    if not jsonl.exists():
        with jsonl.open("w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {jsonl} ({len(rows_out)} lines, {len(seen_img)} images)", flush=True)
    else:
        print(f"keep existing {jsonl} (extracted {len(seen_img)} images)", flush=True)


def _http_range(url: str, start: int, end: int) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def _zip_extract_stored_or_deflate(url: str, loff: int, csize: int) -> bytes:
    import struct
    import zlib

    hdr = _http_range(url, loff, loff + 29)
    if hdr[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header at {loff} for {url}")
    method = struct.unpack("<H", hdr[8:10])[0]
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    data_off = loff + 30 + nlen + xlen
    data = _http_range(url, data_off, data_off + csize - 1)
    if method == 0:
        return data
    if method == 8:
        return zlib.decompress(data, -15)
    raise RuntimeError(f"unsupported zip method {method}")


def prepare_gqa_scoring(eval_root: Path, cache: Path) -> None:
    """Fetch GQA eval.py + testdev_balanced_questions.json (Accuracy only)."""
    data = eval_root / "gqa" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "eval").mkdir(parents=True, exist_ok=True)
    eval_py = data / "eval" / "eval.py"
    questions = data / "testdev_balanced_questions.json"

    eval_zip = "https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip"
    q_zip = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
    if not eval_py.exists() or eval_py.stat().st_size < 1000:
        print("fetching GQA eval/eval.py…", flush=True)
        eval_py.write_bytes(_zip_extract_stored_or_deflate(eval_zip, 0, 5877))
    if not questions.exists() or questions.stat().st_size < 1_000_000:
        print("fetching testdev_balanced_questions.json…", flush=True)
        questions.write_bytes(
            _zip_extract_stored_or_deflate(q_zip, 119369412, 1427737)
        )
    print(f"GQA scoring assets ready under {data}", flush=True)


def prepare_mme_scoring(eval_root: Path, cache: Path) -> None:
    """Install BradyFU eval_tool; convert_answer_to_mme.py is expected beside it."""
    mme = eval_root / "MME"
    mme.mkdir(parents=True, exist_ok=True)
    convert = mme / "convert_answer_to_mme.py"
    if not convert.exists():
        print(
            f"WARN: {convert} missing — place LLaVA/Bunny convert_answer_to_mme.py here",
            flush=True,
        )

    tool = mme / "eval_tool" / "calculation.py"
    if not tool.exists():
        zip_path = cache / "mme_eval_tool_branch.zip"
        url = (
            "https://codeload.github.com/BradyFU/"
            "Awesome-Multimodal-Large-Language-Models/zip/refs/heads/Evaluation"
        )
        if not zip_path.exists() or zip_path.stat().st_size < 1000:
            _curl(url, zip_path)
        print(f"extract MME eval_tool from {zip_path}", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            inner = None
            for name in zf.namelist():
                if name.endswith("tools/eval_tool.zip"):
                    inner = name
                    break
            if inner is None:
                raise RuntimeError("eval_tool.zip not found in BradyFU Evaluation branch")
            raw = zf.read(inner)
        nested = cache / "mme_eval_tool.zip"
        nested.write_bytes(raw)
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(mme)
    print(f"MME scoring assets ready under {mme}", flush=True)


def prepare_pope(eval_root: Path, cache: Path, skip_images: bool) -> None:
    pope = eval_root / "pope"
    coco_ann = pope / "coco"
    coco_ann.mkdir(parents=True, exist_ok=True)
    for name in POPE_FILES:
        dest = coco_ann / name
        if not dest.exists():
            _curl(f"{POPE_BASE}/{name}", dest)

    jsonl = pope / "llava_pope_test.jsonl"
    if not jsonl.exists():
        print("merging POPE → llava_pope_test.jsonl…", flush=True)
        n = 0
        with jsonl.open("w", encoding="utf-8") as out:
            for name in POPE_FILES:
                for line in (coco_ann / name).read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row["text"]
                    if not text.endswith(POPE_PROMPT.strip()) and POPE_PROMPT.strip() not in text:
                        text = f"{text}{POPE_PROMPT}"
                    # Keep category: POPE reuses question_id 1..N across splits.
                    category = name[len("coco_pope_") : -len(".json")]
                    out.write(
                        json.dumps(
                            {
                                "question_id": row["question_id"],
                                "image": row["image"],
                                "text": text,
                                "category": category,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    n += 1
        print(f"  wrote {jsonl} ({n} lines)", flush=True)
    else:
        print(f"keep existing {jsonl}", flush=True)

    val_dir = pope / "val2014"
    if skip_images:
        print("skip COCO val2014 images (--skip-coco-images)", flush=True)
        return
    if val_dir.is_dir() and any(val_dir.glob("*.jpg")):
        n = len(list(val_dir.glob("*.jpg")))
        print(f"keep existing val2014 ({n} jpgs)", flush=True)
        return
    zip_path = cache / "val2014.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000_000:
        _curl(COCO_VAL2014, zip_path)
    print(f"extract {zip_path} → {pope}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(pope)
    if not val_dir.exists():
        # some zips nest differently
        hits = list(pope.rglob("COCO_val2014_*.jpg"))
        if hits:
            real = hits[0].parent
            if real != val_dir:
                val_dir.parent.mkdir(parents=True, exist_ok=True)
                if val_dir.exists():
                    shutil.rmtree(val_dir)
                shutil.move(str(real), str(val_dir))
    print(f"  val2014 images: {len(list(val_dir.glob('*.jpg')))}", flush=True)


def verify(eval_root: Path, tasks: list[str]) -> int:
    checks = {
        "gqa": [
            eval_root / "gqa" / "llava_gqa_testdev_balanced.jsonl",
            eval_root / "gqa" / "data" / "images",
        ],
        "mme": [
            eval_root / "MME" / "llava_mme.jsonl",
            eval_root / "MME" / "MME_Benchmark_release_version",
        ],
        "pope": [
            eval_root / "pope" / "llava_pope_test.jsonl",
            eval_root / "pope" / "val2014",
        ],
    }
    bad = 0
    for task in tasks:
        for path in checks[task]:
            ok = path.exists() and (
                path.is_file()
                and path.stat().st_size > 0
                or path.is_dir()
                and any(path.iterdir())
            )
            print(("OK  " if ok else "MISS") + f" {path}", flush=True)
            if not ok:
                bad += 1
    return bad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-root",
        type=Path,
        default=Path(os.environ.get("LLAVA_EVAL_ROOT", str(DEFAULT_EVAL))),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="download cache (default: <eval-root>/_download_cache)",
    )
    p.add_argument(
        "--tasks",
        default="gqa,mme,pope",
        help="comma list: gqa,mme,pope",
    )
    p.add_argument(
        "--eval-zip",
        action="store_true",
        help="try Google Drive LLaVA eval.zip via gdown first",
    )
    p.add_argument(
        "--skip-coco-images",
        action="store_true",
        help="POPE annotations only (skip ~6.3GB val2014.zip)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="parallel downloads when fetching independent URLs",
    )
    args = p.parse_args(argv)
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in {"gqa", "mme", "pope"}:
            raise SystemExit(f"unknown task: {t}")

    eval_root = args.eval_root.expanduser().resolve()
    cache = (args.cache_dir or (eval_root / "_download_cache")).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    print(f"EVAL_ROOT={eval_root}", flush=True)
    print(f"CACHE={cache}", flush=True)

    if args.eval_zip:
        download_eval_zip(eval_root, cache)

    # Prefetch independent URLs in parallel where helpful
    prefetch: list[tuple[str, Path]] = []
    if "gqa" in tasks:
        prefetch += [
            (GQA_IMAGE_PQ, cache / "gqa_testdev_balanced_images.parquet"),
            (GQA_INST_PQ, cache / "gqa_testdev_balanced_instructions.parquet"),
        ]
    if "mme" in tasks:
        for url in MME_PQS:
            prefetch.append((url, cache / url.rsplit("/", 1)[-1]))
    if "pope" in tasks:
        for name in POPE_FILES:
            prefetch.append((f"{POPE_BASE}/{name}", eval_root / "pope" / "coco" / name))
        if not args.skip_coco_images:
            prefetch.append((COCO_VAL2014, cache / "val2014.zip"))

    missing = [(u, d) for u, d in prefetch if not d.exists() or d.stat().st_size == 0]
    if missing:
        jobs = max(1, args.jobs)
        print(f"prefetch {len(missing)} files with jobs={jobs}", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_curl, u, d): (u, d) for u, d in missing}
            for fut in as_completed(futs):
                u, d = futs[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {u}: {exc}", flush=True)
                    raise

    if "gqa" in tasks:
        prepare_gqa(eval_root, cache)
        prepare_gqa_scoring(eval_root, cache)
    if "mme" in tasks:
        prepare_mme(eval_root, cache)
        prepare_mme_scoring(eval_root, cache)
    if "pope" in tasks:
        prepare_pope(eval_root, cache, skip_images=args.skip_coco_images)

    bad = verify(eval_root, tasks)
    if bad:
        print(f"DONE with {bad} missing paths", flush=True)
        return 1
    print("DONE all requested tasks ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
