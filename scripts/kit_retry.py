#!/usr/bin/env python3
"""Relaunch Kit until it actually starts.

Evidence (2026-11-13): on driver 595.91.07 a launch either crashes at the loader within ~1.6 s
(``libcarb.tasking`` / ``make_fcontext``, exit 139) or stalls before the log advances; roughly one launch in four
reaches Startup Complete, and once there the run is stable. So a pre-startup failure is a **launch** failure, not a
host verdict:

* a launch counts as started when the Kit log (or the attempt's own stdout) shows
  ``Simulation App Startup Complete`` within ``CERBERUS_STARTUP_WINDOW_S`` (default 60 s);
* otherwise the whole process tree is killed and the launch is retried, up to ``CERBERUS_MAX_LAUNCHES`` (default 10);
* **after Startup Complete there is no watchdog** -- the command runs to its own exit;
* every attempt is logged (elapsed, exit code, last log line, crash signature) to ``outputs/launch_probe.csv``.

Exit code 75 means no launch started within the cap; anything else is the command's own exit code.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "outputs" / "launch_probe.csv"
ATTEMPT_DIR = REPO_ROOT / "outputs" / "kit_launches"
STDOUT_TAIL = REPO_ROOT / "outputs" / "kit_retry.log"

STARTUP_MARKER = "Simulation App Startup Complete"
# A launch is not cut off at a fixed 60 s: it may take up to five minutes to reach Startup Complete, and it is
# declared hung early only when the log stops growing AND the CPU and GPU are idle (see classify_window).
STARTUP_WINDOW_S = float(os.environ.get("CERBERUS_STARTUP_WINDOW_S", "300"))
LIVENESS_S = float(os.environ.get("CERBERUS_LIVENESS_S", "240"))
CPU_IDLE_PCT = float(os.environ.get("CERBERUS_CPU_IDLE_PCT", "5"))
GPU_IDLE_PCT = float(os.environ.get("CERBERUS_GPU_IDLE_PCT", "5"))
MAX_LAUNCHES = int(os.environ.get("CERBERUS_MAX_LAUNCHES", "10"))
POLL_S = float(os.environ.get("CERBERUS_LAUNCH_POLL_S", "0.5"))
NO_START_EXIT = 75
CRASH_PATTERNS = ("libcarb.tasking", "make_fcontext")


@dataclass
class Attempt:
    started: bool
    rc: int | None
    elapsed_s: float
    last_line: str
    signature: str
    window: str = ""


def last_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()[:240]
    return ""


def crash_signature(text: str) -> str:
    hits = [name for name in CRASH_PATTERNS if name in text]
    return "/".join(hits)


def classify_window(*, elapsed_s: float, silent_s: float, cpu_pct: float | None, gpu_idle: bool | None,
                   max_wait_s: float, liveness_s: float = LIVENESS_S) -> str | None:
    """``"liveness"`` (hung), ``"5min-cap"`` (slow), or None (keep waiting).

    The liveness rule is the only early kill: the log must have been silent for ``liveness_s`` **and** the CPU below
    5% **and** the GPU idle. A probe that cannot answer counts as "not idle", so a launch is never killed on a guess.
    """
    if (
        silent_s >= liveness_s
        and cpu_pct is not None
        and gpu_idle is not None
        and cpu_pct < CPU_IDLE_PCT
        and gpu_idle
    ):
        return "liveness"
    if elapsed_s >= max_wait_s:
        return "5min-cap"
    return None


def _group_cpu(pgid: int) -> float | None:
    try:
        out = subprocess.run(["ps", "-o", "pcpu=", "--pgid", str(pgid)], capture_output=True, text=True,
                             timeout=5, check=False).stdout
        values = [float(v) for v in out.split() if v.strip()]
        return sum(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _gpu_idle() -> bool | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=False).stdout
        values = [float(v) for v in out.replace("%", "").split() if v.strip()]
        return max(values) < GPU_IDLE_PCT if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def _read(path: Path | None, offset: int = 0) -> str:
    if path is None:
        return ""
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def launch_once(command: list[str], *, env: dict, attempt_log: Path, window_s: float,
                spawn=None, clock=time.monotonic, find_log=None, cpu_probe=None, gpu_probe=None,
                sleep=time.sleep) -> Attempt:
    """One launch. Returns as soon as Startup Complete is seen, the process exits, or the window closes."""
    from kit_logs import find_kit_log

    spawn = spawn or (lambda cmd, log: subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True,
                                                        env=env, start_new_session=True))
    override = env.get("CERBERUS_KIT_LOG")
    find_log = find_log or (lambda: Path(override) if override and Path(override).exists() else find_kit_log())
    attempt_log.parent.mkdir(parents=True, exist_ok=True)
    baseline = find_log()
    base_size = baseline.stat().st_size if baseline and baseline.is_file() else 0
    cpu_probe = cpu_probe or _group_cpu
    gpu_probe = gpu_probe or _gpu_idle
    started_at = clock()
    last_growth = started_at
    seen_len = 0
    last_probe = 0.0
    cpu_pct, gpu_idle = None, None
    with attempt_log.open("w", encoding="utf-8") as out:
        proc = spawn(command, out)
        text = ""
        while True:
            rc = proc.poll()
            text = _read(attempt_log)
            current = find_log()
            if current is not None:
                offset = base_size if (baseline is not None and current == baseline) else 0
                text += _read(current, offset)
            elapsed = clock() - started_at
            if len(text) > seen_len:
                seen_len = len(text)
                last_growth = clock()
            if elapsed - last_probe >= 5.0:
                last_probe = elapsed
                try:
                    pgid = os.getpgid(proc.pid)
                except ProcessLookupError:
                    pgid = proc.pid
                cpu_pct, gpu_idle = cpu_probe(pgid), gpu_probe()
            if STARTUP_MARKER in text:
                rc = proc.wait()  # started: no watchdog from here on
                return Attempt(True, rc, clock() - started_at, last_line(text), crash_signature(text))
            if rc is not None:
                return Attempt(False, rc, elapsed, last_line(text), crash_signature(text))
            window = classify_window(elapsed_s=elapsed, silent_s=clock() - last_growth, cpu_pct=cpu_pct,
                                     gpu_idle=gpu_idle, max_wait_s=window_s,
                                     liveness_s=min(LIVENESS_S, window_s))
            if window:
                _kill_tree(proc)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                text += _read(attempt_log)
                return Attempt(False, proc.returncode, elapsed, last_line(text), crash_signature(text), window)
            sleep(POLL_S)


def run(command: list[str], *, max_launches: int | None = None, window_s: float | None = None,
        env: dict | None = None, log=None) -> int:
    """Launch until started (returning the command's exit code), or return 75 after the launch cap."""

    max_launches = MAX_LAUNCHES if max_launches is None else max_launches
    window_s = STARTUP_WINDOW_S if window_s is None else window_s
    env = os.environ.copy() if env is None else env
    env["PYTHONUNBUFFERED"] = "1"
    info = None
    try:
        from pod_driver_gate import append_probe_row, collect_host_info

        info = collect_host_info()
        log = log or (lambda **row: append_probe_row(CSV_PATH, info, **row))
    except Exception:  # noqa: BLE001 - logging must never block a launch
        log = log or (lambda **row: None)

    for attempt in range(1, max_launches + 1):
        attempt_log = ATTEMPT_DIR / f"attempt_{attempt:02d}.log"
        result = launch_once(command, env=env, attempt_log=attempt_log, window_s=window_s)
        log(stage="launch",
            verdict="STARTED" if result.started else "LAUNCH_FAIL",
            note=(f"attempt={attempt}/{max_launches} elapsed_s={result.elapsed_s:.1f} exit={result.rc} "
                  f"sig={result.signature or '-'} window={result.window or 'startup'} "
                  f"last={result.last_line[:120]}"))
        print(f"[kit_retry] launch {attempt}/{max_launches} "
              f"{'STARTED' if result.started else 'LAUNCH_FAIL'} elapsed={result.elapsed_s:.1f}s "
              f"exit={result.rc} sig={result.signature or '-'}", flush=True)
        if result.started:
            # the entrypoint's stdout is captured per attempt; replay it so callers (the probe) can see markers
            try:
                print(attempt_log.read_text(encoding="utf-8", errors="replace"), flush=True)
            except OSError:
                pass
            return 0 if result.rc is None else int(result.rc)
    return NO_START_EXIT


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("kit_retry: no command given (usage: kit_retry.sh -- <cmd...>)", file=sys.stderr)
        return 2
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
