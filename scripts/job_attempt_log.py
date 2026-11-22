#!/usr/bin/env python3
"""Append one long-job attempt row to outputs/launch_probe.csv (stage ``job``)."""
from __future__ import annotations

import sys

from pod_driver_gate import CSV_PATH, append_probe_row, collect_host_info


def main(argv: list[str]) -> int:
    name, attempt, rc, elapsed = argv[0], int(argv[1]), int(argv[2]), float(argv[3])
    verdict = "OK" if rc == 0 else ("GAVE_UP" if attempt < 0 else "RESTART")
    append_probe_row(CSV_PATH, collect_host_info(), stage="job", verdict=verdict,
                     note=f"job={name} attempt={attempt} exit={rc} elapsed_s={elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
