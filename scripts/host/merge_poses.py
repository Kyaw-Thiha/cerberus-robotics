#!/usr/bin/env python3
"""Merge the per-segment pose logs of a segmented clip into one, so the clip QA gate can judge the stitched file.

A clip rendered in segments (one Kit launch each, resuming from the saved robot state) writes one
``<name>_segN.poses.json`` per segment. The gate wants one log for the whole clip, with frames in order.

Usage::

    python3 scripts/host/merge_poses.py --segments-dir clips/arm_chase_seg0 clips/arm_chase_seg1 \\
        --name arm_chase --out clips/arm_chase.poses.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def merge(segment_dirs: list[Path], name: str) -> dict:
    frames: list[dict] = []
    camera = None
    width = height = None
    hfov = None
    for index, directory in enumerate(segment_dirs):
        path = directory / f"{name}_seg{index}.poses.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing pose log {path}")
        data = json.loads(path.read_text())
        camera = data.get("camera", camera)
        width = data.get("width", width)
        height = data.get("height", height)
        hfov = data.get("hfov_deg", hfov)
        frames.extend(data["frames"])
    return {"schema": "cerberus.poses.v1", "camera": camera, "width": width, "height": height,
            "hfov_deg": hfov, "frames": frames, "segments": len(segment_dirs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--segments-dir", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    merged = merge([Path(p) for p in args.segments_dir], args.name)
    Path(args.out).write_text(json.dumps(merged) + "\n")
    print(f"merged {len(merged['frames'])} frames from {merged['segments']} segments -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
