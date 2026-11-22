#!/usr/bin/env python3
"""Minimal rclone stand-in for the channel tests: maps cerberus_r2:<bucket>/<path> to $FAKE_BUCKET/<bucket>/<path>."""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ["FAKE_BUCKET"])
FLAGS_WITH_VALUE = {"--max-depth", "--min-size", "--transfers", "--include", "--exclude"}

def to_local(arg: str) -> tuple[Path, bool]:
    if arg.startswith("cerberus_r2:"):
        rest = arg.split(":", 1)[1]
        return ROOT / rest, True
    return Path(arg), False

def main(argv):
    args, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in FLAGS_WITH_VALUE:
            i += 2; continue
        if a.startswith("-"):
            i += 1; continue
        args.append(a); i += 1
    if not args:
        return 1
    sub = args[0]
    if sub == "copyto" and len(args) == 3:
        src, _ = to_local(args[1]); dst, _ = to_local(args[2])
        if not src.is_file(): return 3
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst); return 0
    if sub == "cat" and len(args) == 2:
        p, _ = to_local(args[1])
        if not p.is_file(): return 3
        sys.stdout.write(p.read_text()); return 0
    if sub == "lsf" and len(args) == 2:
        p, _ = to_local(args[1])
        if not p.is_dir(): return 0
        for f in sorted(p.iterdir()):
            if f.is_file(): print(f.name)
        return 0
    if sub == "copy" and len(args) == 3:
        src, _ = to_local(args[1]); dst, _ = to_local(args[2])
        if not src.exists(): return 3
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                rel = f.relative_to(src); (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst / rel)
        return 0
    return 2

sys.exit(main(sys.argv[1:]))
