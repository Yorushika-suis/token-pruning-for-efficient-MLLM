"""Loaded when scripts/cdpruner is first on PYTHONPATH.

Re-applies the Video-MME parquet redirect from scripts/repro, then installs the
CDPruner hook if CDPRUNER=1. Python only auto-imports the first sitecustomize.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPRO = Path(__file__).resolve().parents[1] / "repro"
if str(_REPRO) not in sys.path:
    sys.path.insert(0, str(_REPRO))

_repro_site = _REPRO / "sitecustomize.py"
if _repro_site.exists():
    spec = importlib.util.spec_from_file_location("_repro_sitecustomize", _repro_site)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def _install_later() -> None:
    if os.environ.get("CDPRUNER") != "1":
        return
    import hook_videomme

    hook_videomme.install()


_install_later()

_real_import = __import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)
    if os.environ.get("CDPRUNER") == "1" and name in (
        "llava.model.llava_arch",
        "llava.model.language_model.modeling_qwen2",
        "lmms_eval.models.llava_vid",
    ):
        import hook_videomme

        hook_videomme.install()
    return mod


import builtins

builtins.__import__ = _import
