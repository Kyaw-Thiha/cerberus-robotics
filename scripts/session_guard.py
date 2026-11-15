#!/usr/bin/env python3
"""Laptop-side detached guard: stop the Vast.ai instance at the wall clock or the USD cap.

This runs on the laptop, in its own session (``--detach`` uses ``setsid`` semantics via ``start_new_session``), so it
survives a lost shell and does not depend on anything inside the instance. It re-checks every 30 s, and once the stop
condition is met it stops the instance and retries until it succeeds or the absolute hard wall is reached.

    python3 scripts/session_guard.py --detach --id 12345 --rate 0.55 --started 1763... \\
        --wall 5400 --hard-wall 10800 --usd 3.0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE = REPO_ROOT / "outputs" / "session_guard.json"
LOG = REPO_ROOT / "outputs" / "session_guard.log"
TICK_S = 30


def write_state(**data) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    data["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATE.write_text(json.dumps(data, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", default=None)
    parser.add_argument("--rate", type=float, default=0.0, help="USD per hour")
    parser.add_argument("--targets", default="", help="comma list id:rate for several live instances")
    parser.add_argument("--provider", choices=["vast", "runpod"], default="vast")
    parser.add_argument("--started", type=float, default=None, help="creation epoch (default: now)")
    parser.add_argument("--wall", type=float, default=90 * 60)
    parser.add_argument("--hard-wall", type=float, default=3 * 3600)
    parser.add_argument("--usd", type=float, default=3.0)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if args.started is None:
        args.started = time.time()
    targets = []
    if args.targets:
        for item in args.targets.split(","):
            iid, _, rate = item.partition(":")
            if iid:
                targets.append((iid, float(rate or 0)))
    elif args.id:
        targets.append((args.id, args.rate))
    if not targets:
        print("session_guard: no targets", file=sys.stderr)
        return 2
    if args.detach:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        raw = sys.argv[1:] if len(sys.argv) > 1 else list(argv or [])
        forward = [item for item in raw if item != "--detach"]
        with LOG.open("a", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *forward],
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        print(f"session_guard: detached pid {child.pid} (log {LOG})")
        return 0

    from vast_provider import guard_decision

    key_env = "RUNPOD_API_KEY" if args.provider == "runpod" else "VAST_API_KEY"
    key = os.environ.get(key_env, "")
    if not key:
        print(f"session_guard: {key_env} is not set", file=sys.stderr)
        write_state(status="error", reason=f"no {key_env}")
        return 2

    stopping = False
    while True:
        now = time.time()
        blended_rate = sum(rate for _, rate in targets)
        decision = guard_decision(started_epoch=args.started, now_epoch=now, rate_per_hour=blended_rate,
                                  wall_s=args.wall, usd_cap=args.usd)
        hard_left = args.started + args.hard_wall - now
        write_state(status="stopping" if stopping else "waiting", targets=[i for i, _ in targets],
                    remaining_s=round(decision.remaining_s), spend_usd=decision.spend_usd,
                    hard_wall_remaining_s=round(hard_left), reason=decision.reason)
        if not stopping and decision.action == "stop":
            stopping = True
            print(f"session_guard: STOP {args.id}: {decision.reason}", flush=True)
        if stopping:
            failed = []
            for iid, _rate in targets:
                if args.provider == "runpod":
                    from runpod_provider import stop_pod
                    status, text = stop_pod(key, iid)
                else:
                    from vast_provider import stop_instance
                    status, text = stop_instance(key, iid)
                write_state(status="stop_attempt", instance=iid, http=status, body=text[:200],
                            spend_usd=decision.spend_usd)
                print(f"session_guard: stop {iid} -> HTTP {status}", flush=True)
                if status >= 400:
                    failed.append(iid)
            if not failed:
                write_state(status="stopped", targets=[i for i, _ in targets], spend_usd=decision.spend_usd)
                return 0
            if hard_left <= 0:
                write_state(status="stop_failed", targets=failed)
                print("session_guard: hard wall reached and the stop keeps failing", file=sys.stderr)
                return 1
        if hard_left <= 0 and not stopping:
            write_state(status="hard_wall", instance=args.id, spend_usd=decision.spend_usd)
            return 1
        time.sleep(TICK_S)


if __name__ == "__main__":
    raise SystemExit(main())
