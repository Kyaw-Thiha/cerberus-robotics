"""Find, copy and parse Kit's own log.

Kit prints its log path at startup, under ``logs/Kit/Isaac-Sim/<version>/kit_<stamp>.log``. The exact root differs by
image (``/isaac-sim/kit/logs`` when the Isaac Sim runtime is mounted there, ``~/.nvidia-omniverse/logs`` when Kit uses
the home config), so this searches a candidate list and falls back to any ``kit_*.log`` mentioned by the launcher
output. Everything is read-only and safe to call when no log exists.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

CANDIDATE_LOG_ROOTS = (
    "/isaac-sim/kit/logs/Kit/Isaac-Sim",
    "/isaac-sim/kit/logs",
    "~/.nvidia-omniverse/logs/Kit/Isaac-Sim",
    "~/.nvidia-omniverse/logs",
    "/root/.nvidia-omniverse/logs",
    "/workspace/.nvidia-omniverse/logs",
    "/tmp/kit-logs",
)

TIMELINE_MARKERS = {
    "app_starting": r"app\s+start",
    "raytracing_shader_compile": r"waiting for compilation of ray tracing shaders",
    "app_ready": r"app\s+ready",
    "startup_complete": r"startup\s+complete",
}
LOG_LINE_TS = re.compile(r"^\[?(\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?)\]?")


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path))


def find_kit_log(extra_roots: list[str | Path] | None = None) -> Path | None:
    """Newest ``kit_*.log`` under the candidate roots, or ``None``."""
    candidates: list[Path] = []
    for root in list(CANDIDATE_LOG_ROOTS) + [str(p) for p in (extra_roots or [])]:
        base = _expand(root)
        try:
            if not base.exists():
                continue
            candidates.extend(p for p in base.rglob("kit_*.log") if p.is_file())
        except (PermissionError, OSError):
            continue  # an unreadable root is not a failure: try the next candidate
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def log_from_launcher_output(text: str) -> Path | None:
    """The ``.../logs/Kit/.../kit_*.log`` path the launcher may have printed, if any."""
    for match in re.finditer(r"(/[^\s\"']*logs/[^\s\"']*kit_[^\s\"']*\.log)", text):
        path = Path(match.group(1))
        if path.is_file():
            return path
    return None


def parse_timeline(log_text: str) -> dict:
    """Stage markers with the log's own timestamps, plus seconds from the first marker."""
    lines = log_text.splitlines()
    found: dict[str, dict] = {}
    for index, line in enumerate(lines):
        stamp = None
        match = LOG_LINE_TS.match(line.strip())
        if match:
            raw = match.group(1).replace(" ", "T")
            if not raw.endswith("Z") and "+" not in raw:
                raw += "Z"
            try:
                stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                stamp = None
        for name, pattern in TIMELINE_MARKERS.items():
            if name in found:
                continue
            if re.search(pattern, line, re.IGNORECASE):
                found[name] = {"line": index, "ts": stamp.isoformat() if stamp else None}
    stamps = [datetime.fromisoformat(v["ts"]) for v in found.values() if v["ts"]]
    origin = min(stamps) if stamps else None
    for value in found.values():
        if origin and value["ts"]:
            value["elapsed_s"] = round((datetime.fromisoformat(value["ts"]) - origin).total_seconds(), 1)
    return {"markers": found, "log_lines": len(lines)}


def copy_kit_log(out_dir: str | Path, *, stage: str = "probe", stdout_text: str = "",
                 extra_roots: list[str | Path] | None = None) -> dict:
    """Copy the newest Kit log and the launcher stdout into ``out_dir``; always call this, whatever the verdict."""
    out = Path(out_dir) / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{stage}"
    out.mkdir(parents=True, exist_ok=True)
    source = find_kit_log(extra_roots) or log_from_launcher_output(stdout_text)
    copied = None
    text = ""
    if source is not None:
        copied = out / source.name
        shutil.copy2(source, copied)
        text = source.read_text(encoding="utf-8", errors="replace")
    (out / "stdout.log").write_text(stdout_text, encoding="utf-8", errors="replace")
    timeline = parse_timeline(text) if text else {"markers": {}, "log_lines": 0}
    timeline.update({"kit_log": str(copied) if copied else None,
                     "kit_log_source": str(source) if source else None,
                     "stdout_bytes": len(stdout_text)})
    (out / "launch_timeline.json").write_text(
        __import__("json").dumps(timeline, indent=2) + "\n", encoding="utf-8")
    return timeline
