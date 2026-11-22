#!/usr/bin/env python3
"""Launch probe: does Kit start on this host at all, within the launch cap?

The probe is now a thin wrapper around :mod:`kit_retry`, because a pre-startup crash or stall is a *launch* failure to
retry, not a host verdict (evidence: 2026-11-13, ~1 launch in 4 reaches Startup Complete on driver 595.91.07 and runs
are stable once it does).

* ``kit_retry`` performs up to ``CERBERUS_MAX_LAUNCHES`` (10) launches, each waiting ``CERBERUS_STARTUP_WINDOW_S``
  (60 s) for ``Simulation App Startup Complete``;
* the probe classifies the entrypoint as ``OK`` (exit 0 with ``PROBE_OK``) or ``LAUNCH_FAIL``;
* Kit's log and the attempt logs are always copied to ``outputs/kit_logs/<stamp>_probe/`` with
  ``launch_timeline.json``; every launch is a row in ``outputs/launch_probe.csv`` (stage ``launch``);
* a host is abandoned after 10 consecutive ``LAUNCH_FAIL``s or when the launch-stage cap (15 min) is reached. There is
  no cross-host "two non-OK" rule any more.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "outputs" / "launch_probe.csv"
LOG_DIR = REPO_ROOT / "outputs" / "kit_logs"
STDOUT_LOG = REPO_ROOT / "outputs" / "probe_last.log"
LAUNCH_CAP_S = 15 * 60
EXIT_NON_OK = 4


def launch_counts(csv_path: Path, host: str) -> tuple[int, int]:
    """``(started, total)`` launch rows recorded for ``host`` (stage ``launch``)."""
    import csv

    if not csv_path.is_file():
        return 0, 0
    started = total = 0
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("stage") == "launch" and (row.get("host") or "") == host:
                total += 1
                started += int(row.get("verdict") == "STARTED")
    return started, total


def main(argv: list[str] | None = None) -> int:
    import argparse

    from driver_gate import append_probe_row, collect_host_info

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=float, default=float(os.environ.get("CERBERUS_STARTUP_WINDOW_S", "60")))
    parser.add_argument("--max-launches", type=int, default=int(os.environ.get("CERBERUS_MAX_LAUNCHES", "10")))
    parser.add_argument("--cache-restored", action="store_true",
                        help="record that a cache bundle was restored; does not change the launch logic")
    args = parser.parse_args(argv)

    lab = os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab").rstrip("/") + "/isaaclab.sh"
    info = collect_host_info()
    if not Path(lab).is_file():
        print(f"LAUNCH PROBE: LAUNCH_FAIL isaaclab.sh not found at {lab}")
        append_probe_row(CSV_PATH, info, stage="probe", verdict="LAUNCH_FAIL", note="isaaclab.sh missing",
                         cache=str(args.cache_restored).lower())
        return EXIT_NON_OK

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CERBERUS_ALLOW_BAD_DRIVER_RENDER", "1")
    env["CERBERUS_STARTUP_WINDOW_S"] = str(args.window)
    env["CERBERUS_MAX_LAUNCHES"] = str(args.max_launches)
    cmd = ["bash", str(REPO_ROOT / "scripts" / "host" / "kit_retry.sh"), "--", lab, "-p",
           str(REPO_ROOT / "scripts" / "host" / "launch_probe_env.py"), "--headless", "--verbose-kit"]
    STDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"LAUNCH PROBE: starting (window={args.window:.0f}s, max_launches={args.max_launches}, "
          f"cap={LAUNCH_CAP_S}s, cache_restored={args.cache_restored})", flush=True)
    rc, output = None, ""
    try:
        with STDOUT_LOG.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True,
                                  timeout=LAUNCH_CAP_S + args.window * args.max_launches, env=env, check=False)
        rc = proc.returncode
        output = STDOUT_LOG.read_text(encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        rc = 124
    except OSError as e:
        rc, output = 1, f"{type(e).__name__}: {e}"

    launch_dir = REPO_ROOT / "outputs" / "kit_launches"
    if launch_dir.is_dir():
        for log in sorted(launch_dir.glob("attempt_*.log")):
            try:
                output += log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    verdict = "OK" if rc == 0 and "PROBE_OK" in output else "LAUNCH_FAIL"
    started, total = launch_counts(CSV_PATH, info.host)
    note = f"launches={started}/{total} rc={rc} window={args.window:.0f}s max={args.max_launches}"
    from kit_logs import copy_kit_log

    timeline = copy_kit_log(LOG_DIR, stage="probe", stdout_text=output)
    print(f"LAUNCH PROBE: {verdict} driver={info.driver} gpu={info.gpu} host={info.host} {note}")
    print(f"LAUNCH PROBE: timeline -> {timeline.get('kit_log')}")
    append_probe_row(CSV_PATH, info, stage="probe", verdict=verdict, note=note,
                     cache=str(args.cache_restored).lower())
    return 0 if verdict == "OK" else EXIT_NON_OK


if __name__ == "__main__":
    raise SystemExit(main())
