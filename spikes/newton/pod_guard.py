#!/usr/bin/env python3
"""Laptop-side detached guard for a RunPod job pod: stop it at the wall clock or the USD cap.

Why this exists instead of a bare ``session_guard.py --provider runpod`` call: that path reads the *Vast*
key before it looks at the provider and then passes that same key to the RunPod stop, so on a RunPod
target it either refuses to start (no ``VAST_API_KEY``) or stops with the wrong credential. The guard's
decision logic is the same pure function, imported read-only; only the stop call and the credential are
RunPod's. ``session_guard.write_state`` and its state file are reused so both guards report identically.

    python3 spikes/newton/pod_guard.py --detach --id <pod> --rate 0.89 --wall 7200 --usd 2.5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

TICK_S = 30


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="RunPod pod id")
    parser.add_argument("--rate", type=float, required=True, help="pod USD/hour")
    parser.add_argument("--started", type=float, default=None, help="creation epoch (default: now)")
    parser.add_argument("--wall", type=float, default=7200.0)
    parser.add_argument("--hard-wall", type=float, default=9000.0)
    parser.add_argument("--usd", type=float, default=2.5)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if args.started is None:
        args.started = time.time()

    if args.detach:
        log = REPO / "outputs" / "newton_pod_guard.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        forward = [a for a in (sys.argv[1:] if len(sys.argv) > 1 else list(argv or [])) if a != "--detach"]
        with log.open("a", encoding="utf-8") as handle:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *forward],
                                     stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        print(f"pod_guard: detached pid {child.pid} (log {log})")
        return 0

    from session_guard import write_state  # read-only import: the repo's shared state file
    from vast_provider import guard_decision
    from runpod_provider import stop_pod

    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        write_state(status="error", reason="no RUNPOD_API_KEY")
        print("pod_guard: RUNPOD_API_KEY is not set", file=sys.stderr)
        return 2

    stopping = False
    while True:
        now = time.time()
        decision = guard_decision(started_epoch=args.started, now_epoch=now, rate_per_hour=args.rate,
                                  wall_s=args.wall, usd_cap=args.usd)
        hard_left = args.started + args.hard_wall - now
        write_state(status="stopping" if stopping else "waiting", targets=[args.id],
                    remaining_s=round(decision.remaining_s), spend_usd=decision.spend_usd,
                    hard_wall_remaining_s=round(hard_left), reason=decision.reason, provider="runpod")
        if not stopping and decision.action == "stop":
            stopping = True
            print(f"pod_guard: STOP {args.id}: {decision.reason}", flush=True)
        if stopping:
            status, text = stop_pod(key, args.id)
            # the stop response echoes the pod env, so record the status only, never the body
            write_state(status="stop_attempt", instance=args.id, http=status,
                        spend_usd=decision.spend_usd, provider="runpod")
            print(f"pod_guard: stop {args.id} -> HTTP {status}", flush=True)
            if 200 <= status < 400:
                write_state(status="stopped", targets=[args.id], spend_usd=decision.spend_usd,
                            provider="runpod")
                return 0
            if hard_left <= 0:
                write_state(status="stop_failed", targets=[args.id])
                return 1
        if hard_left <= 0 and not stopping:
            write_state(status="hard_wall", instance=args.id, spend_usd=decision.spend_usd)
            return 1
        time.sleep(TICK_S)


if __name__ == "__main__":
    raise SystemExit(main())
