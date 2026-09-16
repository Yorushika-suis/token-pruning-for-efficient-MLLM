"""Function identity for hooked official bindings."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path


def identity(fn) -> dict:
    obj = getattr(fn, "__func__", fn)
    filename = inspect.getsourcefile(obj)
    if filename is None:
        raise ValueError("cannot resolve source file for " + repr(fn))
    data = Path(filename).read_bytes()
    return {
        "module": obj.__module__,
        "qualname": obj.__qualname__,
        "source_file": filename,
        "source_file_sha256": hashlib.sha256(data).hexdigest(),
        "first_line": inspect.getsourcelines(obj)[1],
    }
