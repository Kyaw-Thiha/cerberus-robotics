#!/usr/bin/env python3
"""Launch Kit until one run reaches Startup Complete, then hand the host to the physics plan.

Failures are per launch and random on driver 595.x, and a started run is stable, so the gate needs only one start:
launch, record the attempt (elapsed, outcome, the window that ended it, crash signature, hygiene log), clean up, and
try again. Abandon the host after ``--max-consecutive-failures`` failures (default 10) or ``--cap-minutes`` minutes
(default 15) in the launch stage.

    python3 scripts/launch_gate.py --max-consecutive-failures 10 --cap-minutes 15 -- "$ISAACLAB_PATH/isaaclab.sh" \
        -p scripts/pod_launch_probe_env.py --headless --verbose-kit
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ABANDON_EXIT = 75


def main(argv: list[str] | None = None) -> int:
    from kit_retry import ATTEMPT_DIR, CSV_PATH, launch_once
    from pod_driver_gate import append_probe_row, collect_host_info

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-consecutive-failures", type=int, default=10)
    parser.add_argument("--cap-minutes", type=float, default=15.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("launch_gate: no command (usage: ... -- <cmd...>)", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env.setdefault("CERBERUS_ALLOW_BAD_DRIVER_RENDER", "1")
    env["PYTHONUNBUFFERED"] = "1"
    info = collect_host_info()
    started_total = failed_total = consecutive = 0
    gate_start = time.monotonic()
    while True:
        attempt = started_total + failed_total + 1
        attempt_log = ATTEMPT_DIR / f"gate_{attempt:02d}.log"
        result = launch_once(command, env=env, attempt_log=attempt_log, window_s=300)
        hygiene = ""
        if not result.started:
            path = REPO / "outputs" / "launch_hygiene.log"
            subprocess.run(["bash", str(REPO / "scripts" / "launch_hygiene.sh"), str(path)],
                           capture_output=True, text=True)
            hygiene = str(path.relative_to(REPO))
        if result.started:
            started_total += 1
        else:
            failed_total += 1
            consecutive += 1
        append_probe_row(CSV_PATH, info, stage="launch",
                         verdict="STARTED" if result.started else "LAUNCH_FAIL",
                         note=(f"attempt={attempt} elapsed_s={result.elapsed_s:.1f} exit={result.rc} "
                               f"sig={result.signature or '-'} window={result.window or 'startup'} "
                               f"hygiene={hygiene or '-'} last={result.last_line[:110]}"))
        print(f"[launch_gate] attempt {attempt}: {'STARTED' if result.started else 'FAIL'} "
              f"elapsed={result.elapsed_s:.1f}s window={result.window or 'startup'} exit={result.rc}", flush=True)
        if result.started:
            return 0
        gate_minutes = (time.monotonic() - gate_start) / 60.0
        if consecutive >= args.max_consecutive_failures or gate_minutes >= args.cap_minutes:
            print(f"[launch_gate] ABANDON after {consecutive} consecutive failures, {gate_minutes:.1f} min, "
                  f"started={started_total} failed={failed_total}", flush=True)
            return ABANDON_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
