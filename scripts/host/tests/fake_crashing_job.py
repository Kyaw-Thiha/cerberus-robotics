#!/usr/bin/env python3
"""Fake Kit job for testing the relaunch rule: writes units atomically, crashes at random (with exit 0 half the time).

usage: fake_crashing_job.py --units-dir DIR --total 8 [--crash-p 0.5] [--seed 0] [--sleep 0]

Each run writes as many units as it gets through before the simulated crash; a restart continues from what exists.
Half of the crashes exit 1, half exit 0 (the masked crash that started all this).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.cerberus_eval.atomic_io import write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units-dir", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--crash-p", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    units = Path(args.units_dir)
    units.mkdir(parents=True, exist_ok=True)
    # the seed moves with every file left behind, so repeated launches do not replay the same crashes
    rng = random.Random(args.seed * 7919 + len(list(units.iterdir())))
    for index in range(args.total):
        target = units / f"unit_{index:02d}.json"
        if target.is_file():
            continue
        if args.sleep:
            import time
            time.sleep(args.sleep)
        if rng.random() < args.crash_p:
            crashes = len(list(units.glob("_crash_*.json")))
            write_json_atomic(units / f"_crash_{crashes:02d}.json", {"crashed": True, "index": index})
            # alternate: every other crash exits 0, the masked-crash shape the wrapper must not trust
            return 0 if crashes % 2 == 0 else 1
        write_json_atomic(target, {"index": index, "done": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
