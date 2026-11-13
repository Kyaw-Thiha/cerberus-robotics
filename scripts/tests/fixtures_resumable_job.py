"""Resumable test job: writes pairs through run_pairs_resumable and crashes once mid-run."""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common.cerberus_eval.nav_loop import run_pairs_resumable  # noqa: E402
from planning.sampling import StartGoalPair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--crash-after", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / ".crashed"
    calls = {"n": 0}

    def run_one(index, _pair):
        calls["n"] += 1
        if not marker.exists() and calls["n"] >= args.crash_after:
            marker.write_text("1")
            os._exit(3)
        return {"outcome": "success", "spl": 1.0, "time_s": 1.0}

    pairs = [StartGoalPair((0.0, 0.0), (1.0, 0.0), 1.0) for _ in range(args.pairs)]
    run_pairs_resumable(out, "toy", pairs, run_one)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
