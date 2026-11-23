#!/usr/bin/env python3
"""Relaunch a Kit job until its *outputs* exist, never trusting the exit code.

Evidence (2026-11-23): Kit on this host class dies with a native ``exit(0)`` at the first Python boundary after a
long sim block, and the loader crashes about nine launches in ten. Isaac Lab's ``AppLauncher`` maps both to a zero
exit, so an exit code cannot say whether the work was done.

Contract:

* the job declares how many units of work it produces (``--expect``) and where they land (``--units-dir``);
* every unit is written atomically (``atomic_io.write_json_atomic``), so a present, parseable unit is a finished
  unit, and a half-written one is invisible;
* after each launch the wrapper counts the valid units; anything short of ``--expect`` relaunches -- including a
  launch that exited 0 ("clean exit without its marker" is a crash);
* ``--resume-flag``/``--resume-glob`` append the newest checkpoint to the command of launch 2+, so training
  continues instead of restarting;
* at most ``--max-launches`` launches (default 60); exit 76 means the units never all arrived.

Usage::

    run_until_complete.py --name nav10 --units-dir out/nav10 --expect 10 --glob '*.json' -- \
        python3 run_session.py --stages nav

    run_until_complete.py --name retrain --units-dir ckpt/go2_arx5 --expect 1 --glob 'model_*.pt' \
        --resume-args --resume --resume-glob 'model_*.pt' -- \
        bash -c 'isaaclab ... ${CERBERUS_RESUME_ARGS:-}'
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DONE_EXIT = 0
EXHAUSTED_EXIT = 76


def valid_unit(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, ValueError):
        return False


def count_units(units_dir: Path, pattern: str) -> int:
    if not units_dir.is_dir():
        return 0
    return sum(1 for path in units_dir.glob(pattern) if valid_unit(path))


def newest_checkpoint(units_dir: Path, pattern: str) -> Path | None:
    if not units_dir.is_dir():
        return None
    rows = [path for path in units_dir.glob(pattern) if path.is_file() and path.stat().st_size > 0]
    return max(rows, key=lambda p: p.stat().st_mtime) if rows else None


def launch(command: list[str], env: dict, log_path: Path) -> tuple[int, float]:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as out:
        out.write(f"\n===== launch at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: "
                  f"{shlex.join(command)}\n")
        out.flush()
        proc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=env, check=False)
    return proc.returncode, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        split = argv.index("--")
        head, command = argv[:split], argv[split + 1:]
    else:
        head, command = argv, []

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--units-dir", required=True)
    parser.add_argument("--expect", type=int, required=True)
    parser.add_argument("--glob", default="*")
    parser.add_argument("--max-launches", type=int, default=int(os.environ.get("CERBERUS_MAX_LAUNCHES", "60")))
    parser.add_argument("--min-launch-s", type=float, default=0.0)
    parser.add_argument("--resume-args", default="--resume", help="flag appended before the checkpoint path")
    parser.add_argument("--resume-glob")
    parser.add_argument("--log-dir", default="outputs/unit_runs")
    parser.add_argument("--launch-env", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(head)

    if not command:
        print("run_until_complete: no command after --", file=sys.stderr)
        return 2

    units_dir = Path(args.units_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    job_log = log_dir / f"{args.name}.log"
    csv_path = log_dir / f"{args.name}.csv"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for item in args.launch_env:
        key, _, value = item.partition("=")
        env[key] = value

    units = count_units(units_dir, args.glob)
    print(f"[until-complete] {args.name}: {units}/{args.expect} units present before launch", flush=True)

    for number in range(1, args.max_launches + 1):
        if units >= args.expect:
            break
        cmd = list(command)
        resume_from = None
        if args.resume_glob and number > 1:
            resume_from = newest_checkpoint(units_dir, args.resume_glob)
            if resume_from is not None:
                env["CERBERUS_RESUME_FROM"] = str(resume_from)
                cmd = cmd + [args.resume_args, str(resume_from)]
        rc, elapsed = launch(cmd, env, job_log)
        units = count_units(units_dir, args.glob)
        if units >= args.expect:
            verdict = "DONE"
        elif elapsed < args.min_launch_s:
            verdict = "SHORT"
        elif rc == 0:
            verdict = "CRASH_NO_MARKER"  # a "clean" exit without the outputs is the masked crash
        else:
            verdict = "CRASH"
        verified = bool(args.resume_args) and resume_from is not None
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "name": args.name, "launch": number, "exit": rc, "elapsed_s": round(elapsed, 1),
            "units": units, "expect": args.expect, "verdict": verdict,
            "resumed": verified,
        }
        new_file = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"[until-complete] {args.name} launch {number}/{args.max_launches} exit={rc} "
              f"elapsed={elapsed:.0f}s units={units}/{args.expect} {verdict}", flush=True)
        if units >= args.expect:
            return DONE_EXIT

    units = count_units(units_dir, args.glob)
    if units >= args.expect:
        return DONE_EXIT
    print(f"[until-complete] {args.name}: gave up with {units}/{args.expect} units", file=sys.stderr)
    return EXHAUSTED_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
