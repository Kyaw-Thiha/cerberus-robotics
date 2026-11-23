"""Atomic writes for units of work (stills, clips, per-trial JSON, checkpoints written by us).

A unit that is visible is a unit that finished: write to ``<name>.part`` in the same directory, flush, fsync, then
``os.replace``. A killed Kit process therefore leaves either no unit or a complete one, which is what the
``run_until_complete`` relaunch rule counts.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_bytes_atomic(path: Path | str, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def write_text_atomic(path: Path | str, text: str) -> Path:
    return write_bytes_atomic(path, text.encode("utf-8"))


def write_json_atomic(path: Path | str, payload: Any, *, indent: int | None = 2) -> Path:
    return write_text_atomic(path, json.dumps(payload, indent=indent, sort_keys=True, default=str) + "\n")
