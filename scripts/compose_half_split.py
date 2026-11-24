#!/usr/bin/env python3
"""Compose two native halves into one 1920x1080 clip: left = CHASE (or any view), right = a modality panel.

Both inputs must already be the half size (default 960x1080), same fps and duration; nothing is cropped or rescaled.
A caption is burned into the top-left of each half in DejaVu Sans (redistributable), and the audio stream is absent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    num, _, den = str(stream["r_frame_rate"]).partition("/")
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": float(num) / float(den or 1), "frames": int(stream.get("nb_frames") or 0),
            "duration_s": float(data["format"]["duration"])}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")


def compose(left: Path, right: Path, out: Path, *, left_text: str, right_text: str,
            crf: int = 20, preset: str = "veryfast", font: str = FONT) -> dict:
    a, b = probe(left), probe(right)
    problems = []
    if (a["width"], a["height"]) != (b["width"], b["height"]):
        problems.append(f"half sizes differ: {a['width']}x{a['height']} vs {b['width']}x{b['height']}")
    if abs(a["fps"] - b["fps"]) > 0.01:
        problems.append(f"fps differ: {a['fps']} vs {b['fps']}")
    if abs(a["duration_s"] - b["duration_s"]) > 0.1:
        problems.append(f"duration differs: {a['duration_s']:.2f}s vs {b['duration_s']:.2f}s")
    if a["frames"] != b["frames"]:
        problems.append(f"frame counts differ: {a['frames']} vs {b['frames']}")
    if problems:
        raise SystemExit("; ".join(problems))
    size = 22
    caption = (f"drawtext=fontfile={font}:text='{{text}}':x=18:y=26:fontsize={size}:"
               "fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=6")
    filt = (f"[0:v]{caption.format(text=_escape(left_text))}[l];"
            f"[1:v]{caption.format(text=_escape(right_text))}[r];[l][r]hstack=inputs=2[v]")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(left), "-i", str(right), "-filter_complex", filt,
           "-map", "[v]", "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    result = {"out": str(out), "size": [2 * a["width"], a["height"]], "fps": a["fps"],
              "frames": a["frames"], "duration_s": a["duration_s"], "left": left_text, "right": right_text}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--left-text", default="Chase camera (render)")
    parser.add_argument("--right-text", default="Data panel")
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args(argv)
    result = compose(Path(args.left), Path(args.right), Path(args.out),
                     left_text=args.left_text, right_text=args.right_text, crf=args.crf)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
