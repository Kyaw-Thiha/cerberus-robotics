#!/usr/bin/env python3
"""Run three Vast job variants at once: a hook probe, the args-hook real job, and the ssh+onstart real job.

Creation picks three distinct hosts among offers that pass the driver/VRAM/disk/reliability/download-speed/price rules.
The first real variant to heartbeat wins; the others are destroyed. Timing is the owner's: 40 minutes for the
container to reach running, 4 minutes after that to the first heartbeat, 5 minutes of staleness once live, and the
90-minute job clock starts at the first heartbeat (absolute 3-hour ceiling). All logs are written masked.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONTAINER_GRACE_S = 40 * 60
HEARTBEAT_GRACE_S = 4 * 60
STALE_S = 5 * 60
JOB_WALL_S = 90 * 60
CEILING_S = 3 * 3600
TICK_S = 30
PROBE_IMAGE = "ubuntu:22.04"


def now() -> float:
    return time.time()


def instance_running(iid: str) -> bool:
    import vast_provider as vp

    _, info = vp.instance_info(os.environ["VAST_API_KEY"], iid)
    return (info.get("actual_status") == "running") or (info.get("cur_state") == "running")


def probe_seen(iid: str) -> bool:
    out = subprocess.run(["vastai", "logs", iid], capture_output=True, text=True, timeout=120)
    return "JOB_OK" in (out.stdout or "")


def fetch_status(job_id: str) -> dict | None:
    from scripts.sync_to_r2 import _rclone_env, _require_env, REMOTE

    env = _rclone_env(_require_env())
    local = REPO / "outputs" / "vast_jobs" / job_id / "status.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    got = subprocess.run(["rclone", "copyto", f"{REMOTE}:{os.environ['R2_BUCKET']}/jobs/{job_id}/status.json",
                          str(local), "--no-check-dest"], env=env, capture_output=True, text=True)
    if got.returncode != 0 or not local.is_file():
        return None
    try:
        return json.loads(local.read_text())
    except ValueError:
        return None


def mask_logs(iid: str, out_dir: Path) -> Path:
    from redact import mask_text

    out = subprocess.run(["vastai", "logs", iid], capture_output=True, text=True, timeout=120)
    path = out_dir / f"{iid}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mask_text((out.stdout or "") + (out.stderr or "")))
    return path


def create(spec_path: Path, *, code_bundle: str, deadline_minutes: int, max_price: float,
           job_prefix: str) -> int:
    import vast_provider as vp
    from create_vast_instance import build_job_payload, job_bootstrap, job_env
    from vast_provider import DEFAULT_IMAGE

    key = os.environ["VAST_API_KEY"]
    status, offers, note = vp.search_offers(key)
    if status != 200:
        print(f"vast_session: offer search failed {status}: {note}", file=sys.stderr)
        return 1
    picks = vp.select_distinct_hosts(offers, n=3, max_price=max_price)
    if not picks:
        print(json.dumps({"matched": 0, "offers": len(offers)}))
        return 2

    made = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "targets": {}}
    probe = picks[0]
    probe_payload = {"image": PROBE_IMAGE, "disk": 8.0, "runtype": "args", "label": f"{job_prefix}-probe",
                     "onstart": "bash", "args": ["-c", "echo JOB_OK; sleep 60"], "env": {},
                     "price": float(probe.get("dph_total", 0))}
    st, body = vp.create_instance(key, probe["id"], payload=probe_payload)
    made["targets"]["probe"] = {"instance": str(body.get("new_contract") or body.get("id")),
                                "rate": float(probe.get("dph_total", 0)), "role": "probe"}
    for role, real in (("args", picks[1] if len(picks) > 1 else probe), ("ssh", picks[2] if len(picks) > 2 else probe)):
        job_id = f"{job_prefix}-{role}"
        env = job_env(job_id, code_bundle, deadline_minutes, DEFAULT_IMAGE)
        if role == "args":
            payload = build_job_payload(real, image=DEFAULT_IMAGE, disk_gb=80, label=job_id,
                                        env=env, deadline_minutes=deadline_minutes, hook="args-doc")
        else:
            payload = {"image": DEFAULT_IMAGE, "disk": 80.0, "runtype": "ssh", "label": job_id,
                       "onstart": job_bootstrap(), "env": env, "price": float(real.get("dph_total", 0))}
        st, body = vp.create_instance(key, real["id"], payload=payload)
        made["targets"][role] = {"instance": str(body.get("new_contract") or body.get("id")),
                                 "rate": float(real.get("dph_total", 0)), "job_id": job_id, "role": role}
    spec_path.write_text(json.dumps(made, indent=2) + "\n")
    print(json.dumps({k: {"instance": v["instance"], "rate": v["rate"], "job": v.get("job_id")}
                      for k, v in made["targets"].items()}, indent=2))
    return 0


def watch(spec_path: Path, out_dir: Path) -> int:
    import vast_provider as vp

    key = os.environ["VAST_API_KEY"]
    spec = json.loads(spec_path.read_text())
    targets = spec["targets"]
    created = time.time()
    running_at: dict[str, float] = {}
    last_hb: dict[str, float] = {}
    winner: str | None = None
    first_hb: float | None = None

    def destroy(role: str) -> None:
        info = targets[role]
        if not info.get("destroyed"):
            vp.destroy_instance(key, info["instance"])
            info["destroyed"] = True

    def fail(reason: str) -> int:
        print(f"[session] {reason}; destroying all", flush=True)
        for role, info in targets.items():
            if not info.get("destroyed"):
                mask_logs(info["instance"], out_dir)
                destroy(role)
        return 3

    while True:
        t = time.time()
        elapsed = t - created
        for role, info in list(targets.items()):
            if info.get("destroyed"):
                continue
            iid = info["instance"]
            if iid not in running_at and instance_running(iid):
                running_at[iid] = t
                print(f"[session] {role} {iid} container running at t={elapsed:.0f}s", flush=True)
            if role == "probe":
                if probe_seen(iid):
                    print(f"[session] probe JOB_OK on {iid}", flush=True)
                    destroy("probe")
                elif elapsed > CONTAINER_GRACE_S:
                    print("[session] probe produced no JOB_OK within its grace", flush=True)
                    mask_logs(iid, out_dir)
                    destroy("probe")
                continue
            if winner is None:
                status = fetch_status(info["job_id"])
                if status is not None:
                    winner, first_hb = role, t
                    last_hb[role] = t
                    print(f"[session] WINNER {role} {iid} stage={status.get('stage')} "
                          f"launches={status.get('launches_ok')}/{status.get('launches_total')}", flush=True)
                    for other in ("probe", "args", "ssh"):
                        if other != role:
                            destroy(other)
            if winner == role:
                status = fetch_status(info["job_id"])
                if status is not None:
                    last_hb[role] = t
                    if str(status.get("detail", "")).startswith("STOP_FAILED"):
                        pull(info["job_id"], out_dir)
                        destroy(role)
                        return 3
                    if status.get("stage") == "done":
                        rc = pull(info["job_id"], out_dir)
                        destroy(role)
                        print(f"[session] job wall used={(t - (first_hb or t)):.0f}s", flush=True)
                        return rc
                if t - last_hb.get(role, t) > STALE_S:
                    print("[session] heartbeat stale; stopping", flush=True)
                    rc = pull(info["job_id"], out_dir)
                    destroy(role)
                    return rc if rc else 3
        if winner is None:
            real = [info for role, info in targets.items() if role != "probe" and not info.get("destroyed")]
            if not real or all(i["instance"] in running_at for i in real) and all(
                    t - running_at[i["instance"]] > HEARTBEAT_GRACE_S for i in real):
                return fail("no real variant heartbeated within the grace")
            if elapsed > CONTAINER_GRACE_S + HEARTBEAT_GRACE_S:
                return fail("containers did not reach running within the grace")
        if elapsed > CEILING_S:
            return fail("absolute 3-hour ceiling")
        time.sleep(TICK_S)


def pull(job_id: str, out_dir: Path) -> int:
    from scripts.sync_to_r2 import _rclone_env, _require_env, REMOTE

    env = _rclone_env(_require_env())
    local = out_dir / job_id
    local.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rclone", "copy", f"{REMOTE}:{os.environ['R2_BUCKET']}/jobs/{job_id}", str(local),
                    "--no-check-dest"], env=env, capture_output=True, text=True)
    listing = subprocess.run(["rclone", "lsjson", "--recursive",
                              f"{REMOTE}:{os.environ['R2_BUCKET']}/jobs/{job_id}"], env=env,
                             capture_output=True, text=True)
    remote = [e["Path"] for e in json.loads(listing.stdout or "[]") if not e.get("IsDir")]
    have = [str(p.relative_to(local)) for p in local.rglob("*") if p.is_file()]
    missing = sorted(set(remote) - set(have))
    print(f"[session] pulled {len(have)} files; missing={missing}", flush=True)
    return 0 if not missing else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--spec", default="/tmp/vast_session.json")
    c.add_argument("--code-bundle", required=True)
    c.add_argument("--deadline-minutes", type=int, default=90)
    c.add_argument("--max-price", type=float, default=1.00)
    c.add_argument("--job-prefix", default=f"h4job-{int(time.time())}")
    w = sub.add_parser("watch")
    w.add_argument("--spec", default="/tmp/vast_session.json")
    w.add_argument("--out", default=str(REPO / "outputs" / "vast_jobs" / "session"))
    args = parser.parse_args(argv)
    if args.cmd == "create":
        return create(Path(args.spec), code_bundle=args.code_bundle, deadline_minutes=args.deadline_minutes,
                      max_price=args.max_price, job_prefix=args.job_prefix)
    return watch(Path(args.spec), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
