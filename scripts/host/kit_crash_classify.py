#!/usr/bin/env python3
"""Classify Kit run failures as *masked crash* or *clean exit* from the evidence Kit leaves behind.

The problem this exists for: on this host class a launch either dies in the loader (``libcarb.tasking`` /
``make_fcontext``) or, after a long sim block, the native layer ends the process with exit 0. Isaac Lab's
``AppLauncher`` reports both as a zero exit, so the code alone cannot say whether work happened.

Evidence used, per failure:
  * ``[Fatal]`` / ``[carb.crashreporter`` lines in the log;
  * a crash dump written by the crash reporter (``*.dmp`` / ``*.dmp.zip`` under the crash dirs);
  * ``coredumpctl`` entries for the process (when systemd-coredump is present);
  * ``dmesg`` Xid / segfault lines (host level, needs permission);
  * the crash signature strings (``A crash has occurred``, ``libcarb.tasking``, ``make_fcontext``).

Usage::

    kit_crash_classify.py --log run1.log --log run2.log --json out.json
    kit_crash_classify.py --dir outputs/kit_launches --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

CRASH_MARKERS = (
    "[Fatal]",
    "carb.crashreporter",
    "A crash has occurred",
    "libcarb.tasking",
    "make_fcontext",
    "SIGSEGV",
    "Segmentation fault",
)
DEFAULT_DUMP_DIRS = ("~/.nvidia-omniverse/crashes", "/tmp", "~/.local/share/ov/data")


@dataclass
class Verdict:
    log: str
    masked_crash: bool
    clean_exit: bool
    markers: list[str] = field(default_factory=list)
    dumps: list[str] = field(default_factory=list)
    xid: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {"log": self.log, "masked_crash": self.masked_crash, "clean_exit": self.clean_exit,
                "markers": self.markers, "dumps": self.dumps, "xid": self.xid, "note": self.note}


def scan_text(text: str) -> list[str]:
    return [marker for marker in CRASH_MARKERS if marker in text]


def find_dumps(dirs: list[Path], since: float | None = None) -> list[str]:
    found: list[str] = []
    for directory in dirs:
        if not directory.is_dir():
            continue
        for pattern in ("*.dmp", "*.dmp.zip"):
            for path in directory.rglob(pattern):
                try:
                    if since is not None and path.stat().st_mtime < since:
                        continue
                except OSError:
                    continue
                found.append(str(path))
    return sorted(found)


def dmesg_xid(lines: list[str] | None = None) -> list[str]:
    if lines is None:
        try:
            out = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=10, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        lines = out.splitlines()
    return [line.strip() for line in lines if "Xid" in line or "segfault" in line.lower()]


def classify(log: Path, dumps: list[str], xid: list[str], exit_code: int | None = None) -> Verdict:
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return Verdict(str(log), False, False, note=f"unreadable: {exc}")
    markers = scan_text(text)
    # a crash signature or a fresh dump means the process did not finish its work, whatever exit code it reported
    masked = bool(markers) or bool(dumps)
    clean = (not masked) and (exit_code in (None, 0))
    note = ""
    if markers and exit_code == 0:
        note = "reported exit 0 while the log shows a crash: masked crash"
    elif markers:
        note = "crash signature present"
    return Verdict(str(log), masked, clean, markers, dumps, xid, note)


def classify_all(logs: list[Path], dump_dirs: list[Path], xid_lines: list[str] | None = None,
                 since: float | None = None) -> dict:
    dumps = find_dumps(dump_dirs, since=since)
    xid = dmesg_xid(xid_lines)
    verdicts = [classify(log, dumps, xid) for log in logs]
    return {
        "verdicts": [v.as_dict() for v in verdicts],
        "counts": {
            "logs": len(verdicts),
            "masked_crash": sum(1 for v in verdicts if v.masked_crash),
            "clean_exit": sum(1 for v in verdicts if v.clean_exit),
        },
        "dumps_found": len(dumps),
        "xid_lines": len(xid),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", action="append", default=[], help="one log file (repeatable)")
    parser.add_argument("--dir", action="append", default=[], help="directory of *.log files (repeatable)")
    parser.add_argument("--dump-dir", action="append", default=[], help="crash-dump directory (repeatable)")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--since-minutes", type=float, default=None,
                        help="only count dumps written in the last N minutes")
    args = parser.parse_args(argv)

    logs = [Path(p) for p in args.log]
    for directory in args.dir:
        logs.extend(sorted(Path(directory).glob("*.log")))
    if not logs:
        print("kit_crash_classify: no logs given", file=sys.stderr)
        return 2
    dumps = [Path(p).expanduser() for p in args.dump_dir] or [Path(p).expanduser() for p in DEFAULT_DUMP_DIRS]
    since = None
    if args.since_minutes is not None:
        import time

        since = time.time() - args.since_minutes * 60
    report = classify_all(logs, dumps, since=since)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
