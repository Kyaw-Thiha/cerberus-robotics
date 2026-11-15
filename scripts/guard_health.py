#!/usr/bin/env python3
"""Check the guard's tick file at the start of every agent turn; stop the pod when ticks are stale.

The guard writes ``outputs/guard_tick.json`` every 30 s. If the newest tick is older than ``--max-gap`` seconds (the
laptop slept, the service died, the agent moved on), the pod is stopped immediately and the reason is printed — a live
pod with no guard is exactly the failure this project has hit before.

    python3 scripts/guard_health.py --provider runpod --pod-id <id> --max-gap 120
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TICK = REPO / "outputs" / "guard_tick.json"


def tick_age(now: float, tick_epoch: float | None) -> float | None:
    return None if tick_epoch is None else max(0.0, now - tick_epoch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-gap", type=float, default=120.0)
    parser.add_argument("--provider", choices=["runpod", "vast"], default="runpod")
    parser.add_argument("--pod-id", action="append", default=[])
    parser.add_argument("--tick", default=str(TICK))
    args = parser.parse_args(argv)
    path = Path(args.tick)
    data = json.loads(path.read_text()) if path.is_file() else {}
    age = tick_age(time.time(), data.get("epoch"))
    print(json.dumps({"tick_age_s": None if age is None else round(age, 1),
                      "max_gap_s": args.max_gap, "status": data.get("status")}))
    if age is not None and age <= args.max_gap:
        return 0
    print(f"GUARD TICK STALE: last tick {'missing' if age is None else f'{age:.0f}s old'}; stopping the pod now",
          file=sys.stderr)
    if not args.pod_id:
        print("guard_health: no --pod-id to stop", file=sys.stderr)
        return 2
    sys.path.insert(0, str(REPO / "scripts"))
    for pod in args.pod_id:
        if args.provider == "runpod":
            from runpod_provider import stop_pod
            key_env = "RUNPOD_API_KEY"
        else:
            from vast_provider import stop_instance as stop_pod
            key_env = "VAST_API_KEY"
        import os
        status, _ = stop_pod(os.environ[key_env], pod)
        print(f"guard_health: stop {pod} -> HTTP {status}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
