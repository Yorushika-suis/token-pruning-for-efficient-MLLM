"""Download Hub repos via ModelScope (this cluster cannot reach huggingface.co).

Why not `hf download`:
- huggingface.co AAAA points at Facebook (`face:b00c`), which is the SSL mismatch.
- hf-mirror.com currently 308s API/resolve traffic back to huggingface.co, so
  huggingface_hub never sees X-Repo-Commit and errors out.

Files are written into $HF_HOME/hub in the layout transformers/lmms-eval expect.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS_ENDPOINT = "https://www.modelscope.cn"

# liuhaotian/ is not on ModelScope. Original dump is preferred (llama keys +
# mm_projector.bin). AI-ModelScope/llava-v1.5-7b currently 404s; llava-hf/* is
# the Transformers conversion and will NOT load in haotian-liu/LLaVA until
# utils/convert_llava_hf_to_haotian.py is run.
MODELSCOPE_ID: dict[str, str | list[str]] = {
    "liuhaotian/llava-v1.5-7b": [
        "AI-ModelScope/llava-v1.5-7b",
        "llava-hf/llava-1.5-7b-hf",
    ],
    "openai/clip-vit-large-patch14-336": "AI-ModelScope/clip-vit-large-patch14-336",
}


class _RefuseHuggingFaceRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if "huggingface.co" in newurl:
            raise RuntimeError(
                f"refusing redirect to huggingface.co ({newurl}). "
                "That host is hijacked or unreachable on this machine."
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_RefuseHuggingFaceRedirect)


def ensure_hf_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(ROOT / "weights"))


def find_hf() -> str:
    for candidate in ("/usr/local/bin/hf", shutil.which("hf"), shutil.which("huggingface-cli")):
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "hf CLI not found. Install with: python3 -m pip install -U 'huggingface_hub[cli]'"
    )


def hub_repo_dir(repo_id: str, repo_type: str = "model") -> Path:
    prefix = {"model": "models", "dataset": "datasets", "space": "spaces"}[repo_type]
    return Path(os.environ["HF_HOME"]) / "hub" / f"{prefix}--{repo_id.replace('/', '--')}"


def pin_refs_main(repo_id: str, revision: str, repo_type: str = "model") -> None:
    refs = hub_repo_dir(repo_id, repo_type) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")
    print(f"pinned {repo_id} refs/main -> {revision}", flush=True)


def read_refs_main(repo_id: str, repo_type: str = "model") -> str | None:
    ref = hub_repo_dir(repo_id, repo_type) / "refs" / "main"
    if not ref.exists():
        return None
    return ref.read_text(encoding="utf-8").strip()


def _ms_ids(repo_id: str) -> list[str]:
    mapped = MODELSCOPE_ID.get(repo_id, repo_id)
    if isinstance(mapped, str):
        return [mapped]
    return list(mapped)


def _ms_id(repo_id: str) -> str:
    return _ms_ids(repo_id)[0]


def _ms_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "token-pruning-download/1.0"})
    with _OPENER.open(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("Code") not in (None, 200, "200"):
        raise RuntimeError(f"ModelScope API {url} -> {payload.get('Code')} {payload.get('Message')}")
    return payload


def _ms_list_files(
    repo_id: str,
    repo_type: str,
    revision: str,
    wanted: set[str] | None = None,
    ms_id: str | None = None,
) -> tuple[str, list[dict]]:
    kind = "datasets" if repo_type == "dataset" else "models"
    ms_id = ms_id or _ms_id(repo_id)
    quoted = urllib.parse.quote(ms_id, safe="/")
    if repo_type == "dataset":
        payload = _ms_json(
            f"{MS_ENDPOINT}/api/v1/datasets/{quoted}/repo/tree?Revision={urllib.parse.quote(revision)}"
        )
        files = (payload.get("Data") or {}).get("Files") or []
        blobs: list[dict] = []
        trees = []
        for item in files:
            if item.get("Type") == "tree":
                trees.append(item)
            elif item.get("Type") == "blob":
                blobs.append(item)
        if wanted:
            trees = [
                tree
                for tree in trees
                if any(
                    path == tree["Path"] or path.startswith(tree["Path"] + "/")
                    for path in wanted
                )
            ]
        for tree in trees:
            nested = _ms_json(
                f"{MS_ENDPOINT}/api/v1/datasets/{quoted}/repo/tree"
                f"?Revision={urllib.parse.quote(revision)}&Root={urllib.parse.quote(tree['Path'])}"
            )
            blobs.extend(
                item
                for item in (nested.get("Data") or {}).get("Files") or []
                if item.get("Type") == "blob"
            )
        files = blobs
        head = next((item.get("Revision") for item in files if item.get("Revision")), revision)
        return str(head), files

    payload = _ms_json(
        f"{MS_ENDPOINT}/api/v1/{kind}/{quoted}/repo/files"
        f"?Revision={urllib.parse.quote(revision)}&Recursive=true"
    )
    files = [
        item
        for item in (payload.get("Data") or {}).get("Files") or []
        if item.get("Type") == "blob"
    ]
    head = (
        ((payload.get("Data") or {}).get("LatestCommitter") or {}).get("Id")
        or next((item.get("Revision") for item in files if item.get("Revision")), revision)
    )
    return str(head), files


def _http_download(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    resume = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "token-pruning-download/1.0"}
    if resume:
        headers["Range"] = f"bytes={resume}-"
    req = urllib.request.Request(url, headers=headers)
    with _OPENER.open(req, timeout=120) as resp:
        mode = "ab" if resume and resp.status == 206 else "wb"
        if mode == "wb":
            resume = 0
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) + resume if total is not None else expected_size
        downloaded = resume
        with open(part, mode) as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = 100.0 * downloaded / total_bytes
                    print(
                        f"\r  {dest.name} {downloaded}/{total_bytes} ({pct:.1f}%)",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
        if total_bytes:
            print(file=sys.stderr)
    part.replace(dest)


def _ms_file_url(
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str,
    ms_id: str | None = None,
) -> str:
    kind = "datasets" if repo_type == "dataset" else "models"
    ms_id = ms_id or _ms_id(repo_id)
    return (
        f"{MS_ENDPOINT}/{kind}/{urllib.parse.quote(ms_id, safe='/')}"
        f"/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(filename, safe='/')}"
    )


def modelscope_download(
    repo_id: str,
    *filenames: str,
    repo_type: str = "model",
    revision: str | None = None,
) -> str:
    ensure_hf_env()
    wanted = set(filenames)
    attempts = []
    if revision:
        attempts.append(revision)
    if "master" not in attempts:
        attempts.append("master")

    last_err: Exception | None = None
    head = None
    files = None
    used_revision = None
    used_ms_id = _ms_id(repo_id)
    for ms_id in _ms_ids(repo_id):
        used_ms_id = ms_id
        for rev in attempts:
            try:
                print(f"ModelScope list {ms_id} revision={rev}", flush=True)
                head, files = _ms_list_files(
                    repo_id, repo_type, rev, wanted=wanted or None, ms_id=ms_id
                )
                if not files:
                    last_err = RuntimeError(f"empty file list at revision={rev}")
                    print(f"  list empty at {rev}", flush=True)
                    continue
                used_revision = rev
                break
            except Exception as exc:
                last_err = exc
                print(f"  list failed: {exc}", flush=True)
        if files and used_revision is not None:
            break
    if files is None or used_revision is None:
        raise SystemExit(f"ModelScope listing failed for {repo_id}: {last_err}")

    if wanted:
        files = [item for item in files if item.get("Path") in wanted]
        missing = wanted - {item.get("Path") for item in files}
        if missing:
            raise SystemExit(f"{repo_id} missing on ModelScope: {sorted(missing)}")

    snapshot_sha = head if head and len(str(head)) >= 8 else used_revision
    snapshot_dir = hub_repo_dir(repo_id, repo_type) / "snapshots" / snapshot_sha
    print(f"HF_HOME={os.environ['HF_HOME']}", flush=True)
    print(f"snapshot {snapshot_dir}", flush=True)
    if used_ms_id != repo_id:
        print(f"ModelScope id remap {repo_id} -> {used_ms_id}", flush=True)

    for item in files:
        rel = item["Path"]
        dest = snapshot_dir / rel
        expected = item.get("Size")
        if dest.exists() and expected and dest.stat().st_size == expected:
            print(f"skip {rel} ({expected} bytes)", flush=True)
            continue
        url = _ms_file_url(repo_id, rel, repo_type, used_revision, ms_id=used_ms_id)
        print(f"GET {url}", flush=True)
        _http_download(url, dest, expected_size=expected)
        print(f"saved {dest} ({dest.stat().st_size} bytes)", flush=True)

    pin_refs_main(repo_id, snapshot_sha, repo_type=repo_type)
    return str(snapshot_dir)


def hf_download(
    repo_id: str,
    *filenames: str,
    repo_type: str = "model",
    revision: str | None = None,
) -> None:
    ensure_hf_env()
    backend = os.environ.get("DOWNLOAD_BACKEND", "modelscope")
    if backend == "hf":
        cmd = [find_hf(), "download", repo_id, *filenames, "--repo-type", repo_type]
        if revision:
            cmd.extend(["--revision", revision])
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return
    modelscope_download(repo_id, *filenames, repo_type=repo_type, revision=revision)
