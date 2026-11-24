#!/usr/bin/env python3
"""Build the showcase film, its teaser and a GIF from finished clips.

The plan is a small JSON (or YAML) file::

    {"title": "Cerberus: ...", "subtitle": "...",
     "chapters": [{"clip": "01_chase_map.mp4", "title": "Exploration",
                   "desc": "chained A* goals, PPO", "caption": "Chase | occupancy grid (ground truth)",
                   "start": 2.0, "duration": 18.0}, ...],
     "credits": ["GRScenes-100 (CC BY-NC-SA 4.0, non-commercial)", "Isaac Sim / Isaac Lab", "JEPLO (authors)"],
     "out_dir": "outputs/videos/2026-11-24"}

Outputs: ``00_cerberus_showcase.mp4`` (1920x1080, 30 fps, H.264 yuv420p crf 18, faststart, no audio),
``00_cerberus_teaser.mp4`` (60 s) and ``00_cerberus_teaser.gif`` (10 s, <= 8 MB). Chapter cards are 2 s; content
is trimmed to 15-25 s; chapters are joined with 0.5 s cross-dissolves. A missing or broken clip skips its chapter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
W, H, FPS = 1920, 1080, 30
FADE = 0.5
TITLE_S = 4.0
CARD_S = 2.0
END_S = 6.0


def font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    s = data["streams"][0]
    num, _, den = s["r_frame_rate"].partition("/")
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": float(num) / float(den or 1),
            "duration": float(data["format"]["duration"])}


def _card(path: Path, *, number: str, title: str, desc: str, background: Path | None = None,
          darken: float = 0.55) -> None:
    if background is not None and background.is_file():
        frame = Image.open(background).convert("RGB").resize((W, H), Image.LANCZOS)
        frame = Image.eval(frame, lambda v: int(v * (1.0 - darken)))
        img = frame
    else:
        img = Image.new("RGB", (W, H), (18, 20, 26))
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, H // 2 - 170, W, H // 2 + 190], fill=(10, 12, 16, 150))
    if number:
        draw.text((120, H // 2 - 150), number, font=font(64), fill=(235, 180, 90, 255))
    draw.text((120, H // 2 - 60), title, font=font(72), fill=(245, 245, 245, 255))
    draw.text((120, H // 2 + 60), desc, font=font(34), fill=(200, 200, 200, 255))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img.save(path)


def _card_clip(png: Path, out: Path, seconds: float) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(png), "-t", f"{seconds:.2f}",
                    "-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", str(out)], check=True)


def _chapter_clip(source: Path, out: Path, *, start: float, duration: float, caption: str,
                  tick: str) -> bool:
    info = _probe(source)
    if info["fps"] < 20:
        return False
    duration = min(duration, max(0.5, info["duration"] - start))
    # a typographic apostrophe avoids ffmpeg drawtext's single-quote parsing entirely
    cap = (caption.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019").replace("%", "\\%"))
    draw = (f"drawtext=fontfile={FONT}:text='{cap}':x=40:y=h-96:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=10:enable='between(t,0.5,3.6)',"
            f"drawtext=fontfile={FONT}:text='{tick}':x=w-260:y=24:fontsize=26:fontcolor=white:"
            "box=1:boxcolor=black@0.45:boxborderw=8")
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"fps={FPS},{draw},format=yuv420p")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{duration:.2f}",
                    "-i", str(source), "-vf", vf, "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    str(out)], check=True)
    return True


def _xfade(segments: list[Path], out: Path, fade: float = FADE) -> None:
    if len(segments) == 1:
        shutil.copy(segments[0], out)
        return
    durations = [_probe(p)["duration"] for p in segments]
    fade = min(fade, min(durations) * 0.4)
    inputs = []
    for p in segments:
        inputs += ["-i", str(p)]
    chain = []
    offset = durations[0] - fade
    previous = "[0:v]"
    for i in range(1, len(segments)):
        label = f"[v{i}]"
        chain.append(f"{previous}[{i}:v]xfade=transition=fade:duration={fade}:offset={offset:.3f}{label}")
        previous = label
        offset += durations[i] - fade
    filt = ";".join(chain)
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", filt, "-map", previous,
                    "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)], check=True)


def build(plan: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    chapters = [c for c in plan.get("chapters", []) if Path(c["clip"]).is_file()]
    if not chapters:
        raise SystemExit("no chapter clips exist")
    timing = plan.get("timing") or {}
    title_s = float(timing.get("title_s", TITLE_S))
    card_s = float(timing.get("card_s", CARD_S))
    end_s = float(timing.get("end_s", END_S))
    fade = float(timing.get("fade", FADE))
    work = Path(tempfile.mkdtemp(prefix="showcase_"))
    segments: list[Path] = []
    used: list[str] = []

    first = Path(chapters[0]["clip"])
    frame = work / "title_bg.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(first), "-frames:v", "1", str(frame)],
                   check=True)
    title_png = work / "title.png"
    _card(title_png, number="", title=plan.get("title", "Cerberus"),
          desc=plan.get("subtitle", "a modular quadruped learning stack"), background=frame, darken=0.6)
    title_clip = work / "seg_title.mp4"
    _card_clip(title_png, title_clip, title_s)
    segments.append(title_clip)

    total = len(chapters)
    for i, chapter in enumerate(chapters, start=1):
        card_png = work / f"card_{i}.png"
        _card(card_png, number=f"{i:02d}", title=chapter["title"], desc=chapter.get("desc", ""))
        card_clip = work / f"seg_card_{i}.mp4"
        _card_clip(card_png, card_clip, card_s)
        body_clip = work / f"seg_body_{i}.mp4"
        ok = _chapter_clip(Path(chapter["clip"]), body_clip, start=float(chapter.get("start", 0.0)),
                           duration=float(chapter.get("duration", 18.0)),
                           caption=chapter.get("caption", ""), tick=f"chapter {i}/{total}")
        if not ok:
            continue
        segments += [card_clip, body_clip]
        used.append(chapter["title"])

    credits = plan.get("credits") or []
    end_png = work / "end.png"
    img = Image.new("RGB", (W, H), (12, 14, 18))
    draw = ImageDraw.Draw(img)
    draw.text((140, 260), "Credits", font=font(64), fill=(235, 180, 90))
    y = 380
    for line in credits + ["ground truth / sensed / scripted as captioned"]:
        draw.text((140, y), line, font=font(30), fill=(220, 220, 220))
        y += 56
    img.save(end_png)
    end_clip = work / "seg_end.mp4"
    _card_clip(end_png, end_clip, end_s)
    segments.append(end_clip)

    film = out_dir / "00_cerberus_showcase.mp4"
    _xfade(segments, film, fade)
    teaser = out_dir / "00_cerberus_teaser.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-t", "60", "-i", str(film), "-an", "-c:v", "libx264",
                    "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(teaser)], check=True)
    gif = out_dir / "00_cerberus_teaser.gif"
    palette = work / "palette.png"
    film_duration = _probe(film)["duration"]
    gif_start = min(10.0, max(0.0, film_duration * 0.3))
    gif_len = max(1.0, min(10.0, film_duration - gif_start))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{gif_start:.2f}", "-t", f"{gif_len:.2f}",
                    "-i", str(film), "-vf", "fps=10,scale=640:-1:flags=lanczos,palettegen=max_colors=128",
                    str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{gif_start:.2f}", "-t", f"{gif_len:.2f}",
                    "-i", str(film), "-i", str(palette),
                    "-lavfi", "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
                    str(gif)], check=True)
    return {"film": str(film), "teaser": str(teaser), "gif": str(gif),
            "chapters": used, "size_bytes": film.stat().st_size,
            "gif_bytes": gif.stat().st_size, "duration_s": _probe(film)["duration"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, help="JSON plan (see the docstring)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    out_dir = Path(args.out_dir or plan.get("out_dir") or "outputs/videos/showcase")
    print(json.dumps(build(plan, out_dir), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
