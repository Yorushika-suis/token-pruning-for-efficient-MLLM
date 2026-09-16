"""Stage0 pins before a smoke/eval run."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "FastVID": "a40a109cb8df20e8c2c17c6850f8269717369990",
    "VidCom2": "944a17d14f4848ad080a65eab2b0711a760fa154",
    "VisionZip": "8f86b55c6f000eb033e6912538af2dd7dcb30502",
    "LLaVA": "c121f0432da27facab705978f83c4ada465e46fd",
}
LLAVA_VIDEO_REV = "013210b3aff822f1558b166d39c1046dd109520f"
SIGLIP_REV = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def hub_dir(repo_id: str) -> Path:
    hf = Path(os.environ.get("HF_HOME", ROOT / "weights"))
    return hf / "hub" / f"models--{repo_id.replace('/', '--')}"


def read_ref(repo_id: str) -> str | None:
    p = hub_dir(repo_id) / "refs" / "main"
    return p.read_text().strip() if p.exists() else None


def snapshot_with_config(repo_id: str) -> Path | None:
    root = hub_dir(repo_id)
    snaps = root / "snapshots"
    if not snaps.exists():
        return None
    ref = read_ref(repo_id)
    ordered = []
    if ref:
        ordered.append(snaps / ref)
    ordered.extend(sorted(snaps.iterdir()) if snaps.exists() else [])
    for cand in ordered:
        if cand.is_dir() and (cand / "config.json").exists():
            return cand
    return None


def require_snapshot(repo_id: str, errors: list[str]) -> Path | None:
    snap = snapshot_with_config(repo_id)
    print(f"snapshot {repo_id}", snap)
    if snap is None:
        errors.append(
            f"{repo_id} has refs but no config.json under snapshots/. "
            "Re-run `python utils/download_weights.py` (pinned Hub SHAs are often "
            "empty on ModelScope; the downloader must fall through to master)."
        )
    return snap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("fastvid", "vidcom2", "visionzip"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expect-args", default="", help="substring that must appear in EVAL_ARGS env")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "weights"))
    errors = []

    mapping = {
        "fastvid": ["FastVID"],
        "vidcom2": ["VidCom2"],
        "visionzip": ["VisionZip", "LLaVA"],
    }
    for name in mapping[args.method]:
        head = git_head(ROOT / "ThirdParty" / name)
        if head != EXPECTED[name]:
            errors.append(f"{name} HEAD {head} != {EXPECTED[name]}")
        print(f"{name} {head}")

    if args.method in ("fastvid", "vidcom2"):
        import llava

        llava_file = inspect.getfile(llava)
        print("llava", llava_file)
        if args.method == "fastvid" and "fastvid_llavavideo" not in llava_file:
            errors.append("llava is not FastVID LLaVA-Video: " + llava_file)
        if args.method == "vidcom2" and "VidCom2" not in llava_file.replace("\\", "/"):
            errors.append("llava is not VidCom2: " + llava_file)
        rev = read_ref("lmms-lab/LLaVA-Video-7B-Qwen2")
        print("LLaVA-Video refs/main", rev)
        if rev != LLAVA_VIDEO_REV:
            print(f"WARN LLaVA-Video refs/main {rev} != hub pin {LLAVA_VIDEO_REV} (ModelScope master fallback)")
        require_snapshot("lmms-lab/LLaVA-Video-7B-Qwen2", errors)
        sig = read_ref("google/siglip-so400m-patch14-384")
        print("siglip refs/main", sig)
        if sig != SIGLIP_REV:
            print(f"WARN siglip refs/main {sig} != hub pin {SIGLIP_REV}")
        require_snapshot("google/siglip-so400m-patch14-384", errors)
        videos = Path(os.environ["HF_HOME"]) / "videomme" / "data"
        n = len(list(videos.glob("*.mp4"))) if videos.exists() else 0
        print("videomme mp4s", n, videos)
        if n < 2:
            errors.append("need at least 2 Video-MME mp4s in $HF_HOME/videomme/data")

    if args.method == "visionzip":
        import llava

        print("llava", inspect.getfile(llava))
        if "/LLaVA/llava/" not in inspect.getfile(llava).replace("\\", "/"):
            errors.append("llava is not haotian-liu LLaVA: " + inspect.getfile(llava))
        require_snapshot("liuhaotian/llava-v1.5-7b", errors)
        require_snapshot("openai/clip-vit-large-patch14-336", errors)

    expect = args.expect_args or os.environ.get("EVAL_ARGS", "")
    if "mm_spatial_pool_mode=average" not in expect and args.method in ("fastvid", "vidcom2"):
        errors.append("EVAL_ARGS must contain mm_spatial_pool_mode=average")
    if args.method in ("fastvid", "vidcom2") and "attn_implementation=flash_attention_2" in expect:
        print("WARN flash_attention_2 on B200 is not supported; smoke uses sdpa")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run_dir / "env.pip-freeze.txt").write_text(freeze)
    sha = hashlib.sha256(freeze.encode()).hexdigest()
    print("pip freeze sha256", sha)
    pretrained = None
    if args.method in ("fastvid", "vidcom2"):
        pretrained = snapshot_with_config("lmms-lab/LLaVA-Video-7B-Qwen2")
    elif args.method == "visionzip":
        pretrained = snapshot_with_config("liuhaotian/llava-v1.5-7b")
    if pretrained is not None:
        (run_dir / "pretrained_path.txt").write_text(str(pretrained))
        print("PRETRAINED", pretrained)
    if errors:
        print("STAGE0 FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("STAGE0 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
