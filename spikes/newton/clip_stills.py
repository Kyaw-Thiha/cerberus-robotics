"""Cut stills out of the rollout clip: first frame, the frame at the first fall, and the last frame.

    python -m spikes.newton.clip_stills --clip <dir>/videos/clip_0000.mp4 --report <dir>/rollout_clip.json \
        --out <dir>/stills
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.report).read_text())
    falls = data.get("fall_definition", {}).get("first_fall_steps", {}) or {}
    first_fall = min(falls.values()) if falls else None

    from PIL import Image
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(args.clip)
    n_frames = round(clip.duration * (clip.fps or 1))
    picks = {"frame0": 0}
    if first_fall is not None:
        picks["first_fall"] = min(first_fall, max(n_frames - 1, 0))
    else:
        picks["mid"] = max(n_frames // 2, 0)
    picks["last"] = max(n_frames - 1, 0)
    written = {}
    for name, index in picks.items():
        t = index / float(clip.fps or 1)
        frame = clip.get_frame(min(t, max(clip.duration - 1e-3, 0.0)))
        path = out / f"{name}.png"
        Image.fromarray(frame).save(path)
        written[name] = {"step": index, "file": path.name, "bytes": path.stat().st_size}
        print(f"[stills] {name} step={index} -> {path} ({path.stat().st_size} bytes)", flush=True)
    meta = {"clip": str(args.clip), "duration_s": round(float(clip.duration), 2), "fps": clip.fps,
            "frames": n_frames, "first_fall_step": first_fall, "stills": written}
    (out / "stills.json").write_text(json.dumps(meta, indent=2) + "\n")
    clip.close()
    return 0


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
