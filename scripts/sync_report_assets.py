#!/usr/bin/env python3
"""Copy images referenced by a report markdown into assets/<report-stem>/ and rewrite links.

Finds `![alt](url)` (and HTML src). Resolves repo-relative, absolute, and glued
paths (e.g. `/abs/foo.pngassets/bar.png`). Copies into Docs/Reports/assets/<stem>/
and replaces the URL with a relative `assets/<stem>/<filename>` link.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
MD_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMG = re.compile(r'(<img\b[^>]*\bsrc=)(["\'])([^"\']+)\2', re.I)
PATH_IN_TEXT = re.compile(
    r"(?:(?:/|[A-Za-z]:)[^\s)\"']+?|(?:assets|\.\./)*[^\s)\"']+?)\.(?:png|jpg|jpeg|gif|webp|svg)",
    re.I,
)

DIR_ALIASES = {
    "keep_a": "keep_fastvid_tokenratio",
    "keep_b": "keep_vidcom2_tokenratio",
    "keep_attn_a": "keep_attn_fastvid_tokenratio",
    "keep_attn_b": "keep_attn_vidcom2_tokenratio",
    "geom_a": "geom_fastvid_tokenratio",
    "geom_b": "geom_vidcom2_tokenratio",
}


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def alias_path(path: Path) -> Path:
    parts = list(path.parts)
    changed = False
    for i, part in enumerate(parts):
        if part in DIR_ALIASES:
            parts[i] = DIR_ALIASES[part]
            changed = True
    return Path(*parts) if changed else path


def pick_existing(hits: list[Path], hint: str) -> Path | None:
    existing = [h.resolve() for h in hits if h.is_file()]
    if not existing:
        return None
    if len(existing) == 1:
        return existing[0]
    hint_l = hint.lower()
    prefer: list[str] = []
    if "vidcom2" in hint_l:
        prefer.extend(["keep_attn_vidcom2_tokenratio", "keep_attn_b", "keep_vidcom2_tokenratio", "keep_b"])
    if "fastvid" in hint_l or "keep_attn" in hint_l:
        prefer.extend(["keep_attn_fastvid_tokenratio", "keep_attn_a", "keep_fastvid_tokenratio", "keep_a", "layers"])
    for tag in prefer:
        for h in existing:
            if tag in h.parts:
                return h
    return existing[0]


def try_file(path: Path) -> Path | None:
    for candidate in (path, alias_path(path)):
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_one(raw: str, md_path: Path) -> Path | None:
    text = raw.strip().split()[0].rstrip(").,;")
    if text.startswith("file://"):
        text = text[len("file://") :]
    candidates = []
    p = Path(text)
    if p.is_absolute():
        candidates.append(p)
    candidates.append((md_path.parent / text).resolve())
    candidates.append((ROOT / text.lstrip("./")).resolve())
    if not Path(text).is_absolute():
        name = Path(text).name
        if name:
            hits = list((ROOT / "analyse" / "out").rglob(name))
            picked = pick_existing(hits, text)
            if picked:
                candidates.append(picked)
    for c in candidates:
        hit = try_file(c)
        if hit:
            return hit
    return None


def resolve_url(url: str, md_path: Path) -> Path | None:
    hit = resolve_one(url, md_path)
    if hit:
        return hit
    found: list[Path] = []
    for m in PATH_IN_TEXT.finditer(url):
        hit = resolve_one(m.group(0), md_path)
        if hit and hit not in found:
            found.append(hit)
    if found:
        return found[0]
    return None


def dest_name(src: Path, dest_dir: Path, used: dict[str, Path]) -> str:
    parts = set(src.parts)
    name = src.name
    for old, new in DIR_ALIASES.items():
        if old in parts or new in parts:
            name = f"{new}_{src.name}"
            break
    base, ext = Path(name).stem, Path(name).suffix
    candidate = name
    n = 2
    while candidate in used and used[candidate] != src:
        same = dest_dir.joinpath(candidate).is_file() and file_digest(dest_dir / candidate) == file_digest(src)
        if same:
            break
        candidate = f"{base}_{n}{ext}"
        n += 1
    used[candidate] = src
    return candidate


def rewrite_markdown(md_path: Path, dest_dir: Path, dry_run: bool) -> int:
    text = md_path.read_text(encoding="utf-8")
    rel_prefix = dest_dir.relative_to(md_path.parent).as_posix()
    used: dict[str, Path] = {}
    n_copy = 0

    def handle(url: str) -> str:
        nonlocal n_copy
        src = resolve_url(url, md_path)
        if src is None:
            print(f"WARN missing image: {url}", file=sys.stderr)
            return url
        name = dest_name(src, dest_dir, used)
        dest = dest_dir / name
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or file_digest(dest) != file_digest(src):
                shutil.copy2(src, dest)
                n_copy += 1
                print(f"copy {src} → {dest}")
            else:
                print(f"keep {dest} (already current)")
        else:
            print(f"dry-run {src} → {dest}")
        return f"{rel_prefix}/{name}"

    def md_sub(m: re.Match[str]) -> str:
        alt, url = m.group(1), m.group(2)
        if url.strip() in {"url", "path", "..."}:
            return m.group(0)
        if not any(url.lower().rstrip().endswith(ext) for ext in IMAGE_EXT) and "assets/" not in url and "/" not in url:
            return m.group(0)
        return f"![{alt}]({handle(url)})"

    def html_sub(m: re.Match[str]) -> str:
        return f"{m.group(1)}{m.group(2)}{handle(m.group(3))}{m.group(2)}"

    new = MD_IMG.sub(md_sub, text)
    new = HTML_IMG.sub(html_sub, new)
    if new != text and not dry_run:
        md_path.write_text(new, encoding="utf-8")
        print(f"updated {md_path}")
    elif new == text:
        print(f"no markdown changes in {md_path}")
    return n_copy


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("markdown", type=Path, help="e.g. Docs/Reports/2026-09-19.md")
    p.add_argument("--assets-dir", type=Path, default=None, help="default: <md-dir>/assets/<md-stem>")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    md_path = args.markdown if args.markdown.is_absolute() else (Path.cwd() / args.markdown)
    md_path = md_path.resolve()
    if not md_path.is_file():
        raise SystemExit(f"missing markdown {md_path}")
    dest_dir = args.assets_dir
    if dest_dir is None:
        dest_dir = md_path.parent / "assets" / md_path.stem
    else:
        dest_dir = dest_dir if dest_dir.is_absolute() else (Path.cwd() / dest_dir)
    dest_dir = dest_dir.resolve()
    rewrite_markdown(md_path, dest_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
