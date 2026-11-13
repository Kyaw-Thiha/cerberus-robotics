#!/usr/bin/env python3
"""Session spend estimate and per-stage time caps for a pod run.

Spend is estimated from the pod's hourly price and the elapsed wall time since creation; a wasted image pull is part of
that time, so it counts. The session caps are 6 USD and 3 pod creations; the estimate is printed at every stage and
the driver must stop the pod when it approaches the cap.

    python3 scripts/pod_budget.py --rate 0.89 --start 1762897088
"""

from __future__ import annotations

import argparse
import time

SPEND_CAP_USD = 6.0
WARN_AT_USD = 5.0
MAX_POD_CREATIONS = 3
STAGE_CAPS_MIN = {
    "provision": 25,
    "credentials": 5,
    "fetch": 40,
    "grids": 25,
    "gate_g": 15,
    "stand": 20,
    "scripted": 15,
    "navigation": 90,
    "wrap_up": 15,
}


def estimate(rate_per_hour: float, started_epoch: float, now_epoch: float | None = None, extra_usd: float = 0.0) -> dict:
    """``{elapsed_min, usd, remaining_usd, over_warn}`` for the session so far."""
    now = time.time() if now_epoch is None else now_epoch
    elapsed_min = max(0.0, (now - started_epoch) / 60.0)
    usd = elapsed_min / 60.0 * rate_per_hour + extra_usd
    return {"elapsed_min": round(elapsed_min, 1), "usd": round(usd, 2),
            "remaining_usd": round(SPEND_CAP_USD - usd, 2), "over_warn": usd >= WARN_AT_USD}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate", type=float, required=True, help="pod price per hour")
    parser.add_argument("--start", type=float, required=True, help="session start, epoch seconds")
    parser.add_argument("--extra", type=float, default=0.0, help="already-spent extras (deleted duplicates, storage)")
    parser.add_argument("--stage", default="", help="stage name, for the printed line and its cap")
    args = parser.parse_args(argv)
    est = estimate(args.rate, args.start, extra_usd=args.extra)
    cap = STAGE_CAPS_MIN.get(args.stage)
    cap_note = f" stage_cap={cap}min" if cap else ""
    print(f"[spend] {args.stage or 'session'} elapsed={est['elapsed_min']}min est=${est['usd']:.2f} "
          f"cap=$6.00{cap_note}", flush=True)
    return 0 if est["usd"] < WARN_AT_USD else 3


if __name__ == "__main__":
    raise SystemExit(main())
