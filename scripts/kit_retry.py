#!/usr/bin/env python3
"""Retry a Kit launch that dies with the known startup SIGSEGV.

Isaac Sim 6.x on some 595.x hosts segfaults in ``libcarb.tasking`` (exit 139) in roughly one launch in two, always
within the first minutes. That is a flaky launch, not a dead host, so every Kit entrypoint runs through this wrapper:

* retry only when the exit code is **139** and the attempt lasted **less than 180 s** (configurable);
* at most **3 launches** in total (configurable);
* any other exit code, and any crash after the window, passes through unchanged;
* every attempt is logged to ``outputs/launch_probe.csv`` (host, attempt, elapsed, exit code).

    bash scripts/kit_retry.sh -- /workspace/isaaclab/isaaclab.sh -p scripts/thing.py --headless
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "outputs" / "launch_probe.csv"
SIGSEGV = 139
#: ``isaaclab.sh -p`` turns the child's SIGSEGV into exit 1, so the crash has to be recognised from its own log line.
CRASH_MARKERS = ("exit code 139", "terminating this process with exit code 139")
DEFAULT_WINDOW_S = 180.0
DEFAULT_MAX_LAUNCHES = 3


def looks_like_crash(returncode: int, output: str) -> bool:
    return returncode == SIGSEGV or any(marker in output for marker in CRASH_MARKERS)


def should_retry(returncode: int, elapsed_s: float, window_s: float, output: str = "") -> bool:
    return looks_like_crash(returncode, output) and elapsed_s < window_s


def run(
    command: list[str],
    *,
    window_s: float = DEFAULT_WINDOW_S,
    max_launches: int = DEFAULT_MAX_LAUNCHES,
    runner=None,
    clock=time.monotonic,
    log=None,
) -> int:
    """Run ``command`` with the retry rule; returns the final exit code."""
    def default_runner(command):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".kit_retry.log", delete=False) as fh:
            path = fh.name
        with open(path, "w", encoding="utf-8") as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        try:
            Path(path).unlink()
        except OSError:
            pass
        return rc, text

    runner = runner or default_runner
    info = log_info = None
    try:
        from pod_driver_gate import append_probe_row, collect_host_info

        info = collect_host_info()
        log_info = lambda attempt, elapsed, rc, final: append_probe_row(  # noqa: E731
            CSV_PATH, info, stage="retry",
            verdict=("FINAL" if final else "RETRY"),
            note=f"attempt={attempt} elapsed_s={elapsed:.1f} exit={rc} cmd={' '.join(Path(c).name for c in command[:4])}",
        )
        log = log or log_info
    except Exception:  # noqa: BLE001 - logging must never block a launch
        log = log or (lambda *a, **k: None)

    returncode = 1
    for attempt in range(1, max_launches + 1):
        started = clock()
        result = runner(command)
        returncode, output = (result if isinstance(result, tuple) else (int(result), ""))
        elapsed = clock() - started
        if output:
            print("\n".join(output.strip().splitlines()[-6:]), flush=True)
        final = not should_retry(returncode, elapsed, window_s, output) or attempt == max_launches
        log(attempt, elapsed, returncode, final)
        print(f"[kit_retry] attempt {attempt}/{max_launches} exit={returncode} elapsed={elapsed:.1f}s"
              f"{' (final)' if final else ' -> retrying'}", flush=True)
        if final:
            return returncode
    return returncode


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("kit_retry: no command given (usage: kit_retry.sh -- <cmd...>)", file=sys.stderr)
        return 2
    window = float(os.environ.get("CERBERUS_RETRY_WINDOW_S", DEFAULT_WINDOW_S))
    attempts = int(os.environ.get("CERBERUS_RETRY_MAX", DEFAULT_MAX_LAUNCHES))
    return run(argv, window_s=window, max_launches=attempts)


if __name__ == "__main__":
    raise SystemExit(main())
