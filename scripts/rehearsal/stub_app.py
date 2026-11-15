#!/usr/bin/env python3
"""Stub Isaac app for rehearsal: writes the sample artifacts the real stages would write."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _parse(argv: list[str]) -> tuple[str, Path, dict]:
    stage = "unknown"
    out = Path("outputs/rehearsal")
    opts: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in ("-p", "--python"):
            stage = Path(argv[i + 1]).stem
            i += 2
            continue
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            value = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else "1"
            opts[key] = value
            i += 2
            continue
        i += 1
    if "out_dir" in opts:
        out = Path(opts["out_dir"])
    return stage, out, opts


def _clip(path: Path) -> None:
    """A tiny placeholder MP4 (ftyp box) so the manifest has a clip entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 64)


def main(argv: list[str] | None = None) -> int:
    stage, out, opts = _parse(list(sys.argv[1:] if argv is None else argv))
    out.mkdir(parents=True, exist_ok=True)
    if stage == "scene_eval":
        (out / "assertions.json").write_text(json.dumps(
            {"passed": True, "checks": [
                {"name": "joint_order", "passed": True},
                {"name": "observation_dim", "passed": True, "value": 235},
                {"name": "height_scan_flat", "passed": True, "value": 0.0},
                {"name": "nan_free", "passed": True}]}, indent=2) + "\n")
        (out / "build_timings.json").write_text(json.dumps({"stand_s": 2.0, "steps": 200}, indent=2) + "\n")
        _clip(out / "rehearsal_0_success.mp4")
        (out / "rehearsal_0_success.json").write_text(json.dumps({"outcome": "success", "mock": True}) + "\n")
    elif stage == "nav_eval":
        rows = [{"pair": i, "success": i % 2 == 0, "spl": 0.8 if i % 2 == 0 else 0.0,
                 "time_s": 12.0 + i, "fall": False} for i in range(int(opts.get("pairs", "2")))]
        (out / "nav_metrics.json").write_text(json.dumps({"scene": opts.get("scene", "rehearsal"),
                                                          "pairs": rows}, indent=2) + "\n")
        _clip(out / "rehearsal_0_success.mp4")
        _clip(out / "rehearsal_1_failure.mp4")
    print(f"[stub] stage={stage} out={out} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
