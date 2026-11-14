#!/usr/bin/env python3
"""Laptop-side job monitor: follow the R2 heartbeat, stop on staleness/STOP_FAILED/deadline, pull outputs.

The container job writes ``jobs/<job_id>/status.json`` every 30 s and self-stops when it finishes. This monitor is the
independent second layer: no heartbeat for 5 minutes, a ``STOP_FAILED`` detail, or the 90-minute wall stops the
instance from the laptop and pulls whatever exists. The detached guard (session_guard.py) remains the third layer for
the USD cap.

    python3 scripts/watch_job.py --job-id vast-... --instance-id 123 --rate 0.18 --deadline-minutes 85
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STALE_S = 300
TICK_S = 30


def decide(status: dict | None, *, now: float, last_ok: float | None, started: float,
           wall_s: float = 90 * 60, stale_s: float = STALE_S, startup_grace_s: float = 900) -> str:
    """``"done" | "stop" | "wait"`` from the latest heartbeat.

    Before the first heartbeat the instance may still be pulling the image, so the staleness rule applies only after
    ``startup_grace_s`` (15 min); afterwards a missing heartbeat for ``stale_s`` (5 min) is a stop.
    """
    if status and str(status.get("detail", "")).startswith("STOP_FAILED"):
        return "stop"
    if last_ok is None:
        return "wait" if now - started < startup_grace_s else "stop"
    if now - last_ok > stale_s:
        return "stop"
    if now - started >= wall_s:
        return "stop"
    if status and status.get("stage") == "done":
        return "done"
    return "wait"


def manifest_diff(remote: list[str], local: list[str]) -> dict:
    return {"missing": sorted(set(remote) - set(local)), "extra": sorted(set(local) - set(remote))}


def _rclone(env: dict, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], env=env, capture_output=True, text=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--rate", type=float, default=0.0)
    parser.add_argument("--deadline-minutes", type=float, default=90)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tick", type=float, default=TICK_S)
    parser.add_argument("--startup-grace", type=float, default=900.0)
    args = parser.parse_args(argv)

    from scripts.sync_to_r2 import REMOTE, _rclone_env, _require_env

    env = _rclone_env(_require_env())
    bucket = os.environ["R2_BUCKET"]
    base = f"{REMOTE}:{bucket}/jobs/{args.job_id}"
    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / "vast_jobs" / args.job_id
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    last_ok = None
    status = None
    while True:
        now = time.time()
        got = _rclone(env, ["copyto", f"{base}/status.json", str(out / "status.json"), "--no-check-dest"])
        if got.returncode == 0 and (out / "status.json").is_file():
            last_ok = now
            try:
                status = json.loads((out / "status.json").read_text())
            except ValueError:
                status = None
        action = decide(status, now=now, last_ok=last_ok, started=started,
                        wall_s=args.deadline_minutes * 60, startup_grace_s=args.startup_grace)
        elapsed = now - started
        spend = elapsed / 3600 * args.rate
        print(f"[watch] t={elapsed:5.0f}s spend=${spend:.2f} stage={(status or {}).get('stage')} "
              f"launches={(status or {}).get('launches_ok')}/{(status or {}).get('launches_total')} -> {action}",
              flush=True)
        if action == "wait":
            time.sleep(args.tick)
            continue
        if action == "stop":
            from scripts.vast_provider import stop_instance
            st, text = stop_instance(os.environ["VAST_API_KEY"], args.instance_id)
            print(f"[watch] stop -> HTTP {st} {text[:120]}")
        _rclone(env, ["copy", base, str(out), "--no-check-dest"])
        listing = _rclone(env, ["lsjson", "--recursive", base])
        remote = [e["Path"] for e in json.loads(listing.stdout or "[]") if not e.get("IsDir")]
        local = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()]
        diff = manifest_diff(remote, local)
        print(f"[watch] manifest remote={len(remote)} local={len(local)} missing={diff['missing']} "
              f"extra={diff['extra']}", flush=True)
        (out / "manifest.json").write_text(json.dumps({"remote": remote, **diff}, indent=2) + "\n")
        return 0 if action == "done" and not diff["missing"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
