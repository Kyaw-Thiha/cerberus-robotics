#!/usr/bin/env python3
"""Minimal Kit launch probe: start the app headless with cameras, step the app loop, print markers, exit 0.

Used by the K2 A/B matrix: the only thing that matters is whether *this* launches, so it builds no env at all.
Markers: ``PROBE STARTED`` once the app object exists and five updates ran; ``PROBE DONE`` before exit.
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    args, _ = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    t0 = time.perf_counter()
    launcher = AppLauncher(args)
    app = launcher.app
    for _ in range(5):
        app.update()
    print(f"PROBE STARTED after {time.perf_counter() - t0:.1f}s", flush=True)
    deadline = time.perf_counter() + args.seconds
    while time.perf_counter() < deadline:
        app.update()
    print("PROBE DONE", flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
