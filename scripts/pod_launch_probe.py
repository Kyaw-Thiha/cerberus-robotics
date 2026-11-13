#!/usr/bin/env python3
"""Liveness-based launch probe: cold Kit starts are slow, so "no output in 300 s" is not by itself a verdict.

Every 15 s this samples the Kit log size, the CPU of the process tree and GPU utilisation, and only calls a **HANG**
when the log has not grown for four consecutive minutes *and* CPU is below 5 % *and* the GPU is idle. Otherwise it
keeps waiting up to the wall cap: **20 minutes when the Kit cache is cold, 5 minutes when a cache bundle was
restored**. Verdicts:

* ``OK``                     exit 0 with the probe marker, within 5 minutes
* ``SLOW_OK``                same, but ready after more than 5 minutes (elapsed recorded)
* ``SEGFAULT``               exit 139, or the crash marker in the wrapped output
* ``HANG``                   the liveness rule fired
* ``TIMEOUT_WITH_ACTIVITY``  wall cap reached while the process was demonstrably doing work
* ``OTHER``                  any other non-zero exit

Only ``SEGFAULT`` and ``HANG`` count toward the "two non-OK verdicts on different hosts" stop rule.

Whatever the verdict, Kit's own log and this run's stdout are copied to ``outputs/kit_logs/<stamp>_probe/`` and the
stage markers (app starting, ray-tracing shader compilation, app ready, startup complete) are written to
``launch_timeline.json``. That is the evidence a cold-start hypothesis needs.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "outputs" / "launch_probe.csv"
LOG_DIR = REPO_ROOT / "outputs" / "kit_logs"
STDOUT_LOG = REPO_ROOT / "outputs" / "probe_last.log"

SAMPLE_S = float(os.environ.get("CERBERUS_PROBE_SAMPLE_S", "15"))
HANG_WINDOW_S = float(os.environ.get("CERBERUS_PROBE_HANG_WINDOW_S", "240"))
CPU_IDLE_PCT = 5.0
MAX_WALL_COLD_S = 20 * 60
MAX_WALL_WARM_S = 5 * 60
FAST_OK_S = 5 * 60
EXIT_NON_OK = 4

NON_OK = ("SEGFAULT", "HANG")
"""Only these two count toward the two-host stop rule; SLOW_OK and TIMEOUT_WITH_ACTIVITY do not."""

CRASH_MARKERS = ("exit code 139", "terminating this process with exit code 139")


@dataclass
class Sample:
    elapsed_s: float
    log_size: int
    cpu_pct: float
    gpu_util: float | None


def tree_cpu_seconds(root_pid: int) -> float:
    """utime+stime of ``root_pid`` and all its descendants, in seconds (Linux /proc)."""
    hz = os.sysconf("SC_CLK_TCK")
    parents: dict[int, int] = {}
    times: dict[int, float] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().split()
        except OSError:
            continue
        pid = int(entry.name)
        try:
            times[pid] = (int(stat[13]) + int(stat[14])) / hz
            parents[pid] = int(stat[3])
        except (IndexError, ValueError):
            continue
    children: dict[int, list[int]] = {}
    for pid, parent in parents.items():
        children.setdefault(parent, []).append(pid)
    total, stack = 0.0, [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += times.get(pid, 0.0)
        stack.extend(children.get(pid, []))
    return total


def gpu_utilization() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def log_size(path: Path | None) -> int:
    try:
        return path.stat().st_size if path else 0
    except OSError:
        return 0


def hang_detected(samples: list[Sample], window_s: float = HANG_WINDOW_S) -> bool:
    """True when the last ``window_s`` shows no log growth, low CPU and an idle GPU."""
    if not samples:
        return False
    last = samples[-1]
    window = [s for s in samples if s.elapsed_s >= last.elapsed_s - window_s]
    # one sampling interval of tolerance: a window filtered by "last - window_s" spans at most window_s - SAMPLE_S
    if len(window) < 2 or (last.elapsed_s - window[0].elapsed_s) < window_s - SAMPLE_S:
        return False
    grew = last.log_size > window[0].log_size
    busy_cpu = any(s.cpu_pct >= CPU_IDLE_PCT for s in window)
    busy_gpu = any((s.gpu_util or 0) > 0 for s in window)
    return not (grew or busy_cpu or busy_gpu)


def activity_recent(samples: list[Sample], window_s: float = HANG_WINDOW_S) -> bool:
    if not samples:
        return False
    last = samples[-1]
    window = [s for s in samples if s.elapsed_s >= last.elapsed_s - window_s]
    return any(s.log_size > window[0].log_size or s.cpu_pct >= CPU_IDLE_PCT or (s.gpu_util or 0) > 0
               for s in window) if window else False


def decide(*, exited: bool, rc: int, output: str, elapsed_s: float, killed_for_hang: bool,
           samples: list[Sample]) -> str:
    if exited:
        if rc == 0 and "PROBE_OK" in output:
            return "SLOW_OK" if elapsed_s > FAST_OK_S else "OK"
        if rc == 139 or any(marker in output for marker in CRASH_MARKERS):
            return "SEGFAULT"
        return "OTHER"
    if killed_for_hang:
        return "HANG"
    return "TIMEOUT_WITH_ACTIVITY" if activity_recent(samples) else "HANG"


def non_ok_hosts(csv_path: Path) -> set[str]:
    """Hosts that produced a hard failure (SEGFAULT or HANG)."""
    import csv

    if not csv_path.is_file():
        return set()
    hosts = set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("stage") == "probe" and row.get("verdict") in NON_OK:
                hosts.add(row.get("host") or "unknown")
    return hosts


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def run_probe(command: list[str], *, env: dict, max_wall_s: float, cache_restored: bool) -> dict:
    """Launch, sample, decide; returns ``{verdict, elapsed_s, rc, output, samples, timeline}``."""
    from kit_logs import copy_kit_log, find_kit_log

    STDout = STDOUT_LOG
    STDout.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    samples: list[Sample] = []
    killed_for_hang = False
    with STDout.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env,
                                start_new_session=True)
        rc = None
        while True:
            time.sleep(SAMPLE_S)
            elapsed = time.monotonic() - started
            kit_log = find_kit_log()
            size = log_size(kit_log) or log_size(STDout)
            cpu0 = tree_cpu_seconds(proc.pid)
            time.sleep(0.5)
            cpu_pct = 100.0 * (tree_cpu_seconds(proc.pid) - cpu0) / (0.5 * max(len(os.sched_getaffinity(0)), 1))
            samples.append(Sample(elapsed, size, cpu_pct, gpu_utilization()))
            if proc.poll() is not None:
                rc = proc.returncode
                break
            if hang_detected(samples):
                killed_for_hang = True
                _kill_tree(proc)
                proc.wait(timeout=30)
                rc = proc.returncode
                break
            if elapsed >= max_wall_s:
                _kill_tree(proc)
                proc.wait(timeout=30)
                rc = proc.returncode
                break
    elapsed = time.monotonic() - started
    output = STDout.read_text(encoding="utf-8", errors="replace")
    verdict = decide(exited=not killed_for_hang and rc is not None, rc=rc if rc is not None else 1,
                     output=output, elapsed_s=elapsed, killed_for_hang=killed_for_hang, samples=samples)
    timeline = copy_kit_log(LOG_DIR, stage="probe", stdout_text=output)
    return {"verdict": verdict, "elapsed_s": elapsed, "rc": rc, "output": output,
            "samples": [s.__dict__ for s in samples], "timeline": timeline,
            "cache_restored": cache_restored}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from pod_driver_gate import append_probe_row, collect_host_info
    from pod_guard import require_pod

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-restored", action="store_true",
                        help="a Kit cache bundle was restored for this driver/GPU (5-minute wall cap)")
    parser.add_argument("--max-wall-cold", type=float, default=MAX_WALL_COLD_S)
    parser.add_argument("--max-wall-warm", type=float, default=MAX_WALL_WARM_S)
    args = parser.parse_args(argv)
    require_pod("pod_launch_probe.sh")

    lab = os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab").rstrip("/") + "/isaaclab.sh"
    info = collect_host_info()
    if not Path(lab).is_file():
        print(f"LAUNCH PROBE: OTHER isaaclab.sh not found at {lab}")
        append_probe_row(CSV_PATH, info, stage="probe", verdict="OTHER", note="isaaclab.sh missing",
                         cache=str(args.cache_restored).lower())
        return EXIT_NON_OK

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # keep stdout when the process is killed
    env.setdefault("CERBERUS_ALLOW_BAD_DRIVER_RENDER", "1")
    cmd = ["bash", str(REPO_ROOT / "scripts" / "kit_retry.sh"), "--", lab, "-p",
           str(REPO_ROOT / "scripts" / "pod_launch_probe_env.py"), "--headless", "--verbose-kit"]
    max_wall = args.max_wall_warm if args.cache_restored else args.max_wall_cold
    print(f"LAUNCH PROBE: starting (cache_restored={args.cache_restored}, max_wall={max_wall:.0f}s)", flush=True)
    result = run_probe(cmd, env=env, max_wall_s=max_wall, cache_restored=args.cache_restored)
    verdict = result["verdict"]
    note = (f"elapsed={result['elapsed_s']:.1f}s rc={result['rc']} "
            f"log_bytes={result['samples'][-1]['log_size'] if result['samples'] else 0} "
            f"markers={sorted(result['timeline'].get('markers', {}))}")
    print(f"LAUNCH PROBE: {verdict} driver={info.driver} gpu={info.gpu} host={info.host} {note}")
    print(f"LAUNCH PROBE: timeline -> {result['timeline'].get('kit_log')}")
    append_probe_row(CSV_PATH, info, stage="probe", verdict=verdict, note=note,
                     cache=str(args.cache_restored).lower())

    if verdict in ("OK", "SLOW_OK"):
        return 0
    if verdict in ("SEGFAULT", "HANG"):
        pod_id = os.environ.get("RUNPOD_POD_ID", "")
        api_key = os.environ.get("RUNPOD_API_KEY", "")
        if pod_id and api_key:
            from pod_driver_gate import terminate_pod
            ok, detail = terminate_pod(pod_id, api_key)
            print(f"LAUNCH PROBE: terminating {pod_id} after {verdict}: {'ok' if ok else 'failed'} ({detail})")
    hosts = non_ok_hosts(CSV_PATH)
    if len(hosts) >= 2:
        print(f"LAUNCH PROBE: two hosts have now failed hard ({', '.join(sorted(hosts))}); stopping everything")
    return EXIT_NON_OK


if __name__ == "__main__":
    raise SystemExit(main())
