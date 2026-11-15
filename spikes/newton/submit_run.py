#!/usr/bin/env python3
"""Laptop side of the Newton spike: bundle the branch, upload it, create the job pod, watch it.

Follows the existing job-mode pattern (``scripts/job_entrypoint.sh`` + ``scripts/create_runpod_job.py``)
without editing either: the branch travels as a git bundle on R2, the pod's start command decodes this
spike's own bootstrap from an env variable, and the watcher/guard are the existing ones.

    python3 spikes/newton/submit_run.py --dry-run
    python3 spikes/newton/submit_run.py --start-watch

Budget rules are enforced here: at most 2 pod creations, one live pod, 2.5 USD / 2 h handed to the
guard. Only the restricted R2 variables, the job identity and (for self-stop) RUNPOD_API_KEY are passed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

DEFAULT_IMAGE = "nvidia/cuda:13.0.0-base-ubuntu24.04"
DEFAULT_GPUS = ["NVIDIA RTX A5000", "NVIDIA A40", "NVIDIA GeForce RTX 4090", "NVIDIA L40S",
                "NVIDIA RTX A6000"]
MAX_CREATIONS = 2
CREATIONS = REPO / "outputs" / "newton_pod_creations.json"
CHECKPOINT_REL = "policy/locomotion/checkpoints/unitree_go2_rough/2026-11-07_07-29-40/model_1499.pt"


def find_repo_file(rel: str) -> Path:
    """A gitignored repo file (``.env``, the checkpoint) is absent from a worktree; fall back to the main tree."""
    candidate = REPO / rel
    if candidate.is_file():
        return candidate
    listing = subprocess.run(["git", "-C", str(REPO), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True).stdout
    for line in listing.splitlines():
        if line.startswith("worktree "):
            alternative = Path(line.split(" ", 1)[1]) / rel
            if alternative.is_file():
                return alternative
    return candidate


def find_checkpoint() -> Path:
    return find_repo_file(CHECKPOINT_REL)


ENV_KEYS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "RUNPOD_API_KEY")


def load_env() -> dict:
    """Load the repo ``.env`` into the process without echoing any value (the repo's rule)."""
    path = find_repo_file(".env")
    if not path.is_file():
        return {}
    loaded = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key in ENV_KEYS and value and key not in os.environ:
            os.environ[key] = value
            loaded[key] = True
    return loaded


def rclone(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["rclone", *args], capture_output=True, text=True)


def rclone_env() -> None:
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_TYPE", "s3")
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_PROVIDER", "Cloudflare")
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID", os.environ["R2_ACCESS_KEY_ID"])
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY", os.environ["R2_SECRET_ACCESS_KEY"])
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_ENDPOINT",
                          f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com")
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_ACL", "private")
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET", "true")


def build_bundle(branch: str, target: Path) -> Path:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                          capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True)
    if head.stdout.strip() != branch:
        raise SystemExit(f"submit_run: HEAD is {head.stdout.strip()!r}, expected {branch!r}")
    if dirty.stdout.strip():
        raise SystemExit("submit_run: the worktree is dirty; commit before bundling (the pod gets HEAD only)")
    subprocess.run(["git", "-C", str(REPO), "bundle", "create", str(target), branch], check=True)
    return target


def build_inputs(target_dir: Path) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = find_checkpoint()
    if not checkpoint.is_file():
        raise SystemExit(f"submit_run: checkpoint missing: {checkpoint}")
    tar_path = target_dir / "checkpoint.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(checkpoint, arcname="checkpoint/model_1499.pt")
        meta = checkpoint.parent / "go2_locomotion_ga_v1_candidate.meta.json"
        if meta.is_file():
            tar.add(meta, arcname="checkpoint/go2_locomotion_ga_v1_candidate.meta.json")
    return {"checkpoint_tar": tar_path}


def upload(remote: str, job_id: str, bundle: Path, inputs: dict) -> dict:
    base = f"{remote}/jobs/{job_id}"
    paths = {"bundle": f"spike/{job_id}/code/cerberus-newton.bundle"}
    steps = [(str(bundle), f"{remote}/{paths['bundle']}")]
    for name, local in inputs.items():
        paths[name] = f"spike/{job_id}/inputs/{local.name}"
        steps.append((str(local), f"{remote}/{paths[name]}"))
    for local, dest in steps:
        result = rclone(["copyto", local, dest, "--no-check-dest"])
        print(f"[upload] {local} -> {dest} ({result.returncode})")
        if result.returncode != 0:
            raise SystemExit(f"submit_run: upload failed: {result.stderr[:300]}")
    # the admission manifest is the source of the spawn pose; it is small and already on R2
    src = f"{remote}/{os.environ['R2_BUCKET']}/worlds/grscenes/admission.json"
    dst = f"{remote}/jobs/{job_id}/inputs/admission.json"
    result = rclone(["copyto", src, dst, "--no-check-dest"])
    if result.returncode != 0:
        print(f"[upload] admission.json not copied ({result.returncode}); the scene stage falls back to the origin")
    return paths


def creations() -> int:
    if not CREATIONS.is_file():
        return 0
    return len(json.loads(CREATIONS.read_text()))


def record(pod_id: str, image: str) -> None:
    rows = json.loads(CREATIONS.read_text()) if CREATIONS.is_file() else []
    rows.append({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pod_id": pod_id, "image": image})
    CREATIONS.parent.mkdir(parents=True, exist_ok=True)
    CREATIONS.write_text(json.dumps(rows, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branch", default="newton-spike")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--deadline-minutes", type=int, default=105)
    parser.add_argument("--disk-gb", type=int, default=120)
    parser.add_argument("--gpu-type", action="append", default=None)
    parser.add_argument("--rate", type=float, default=0.79, help="pod USD/hour, for the guard and watcher")
    parser.add_argument("--start-watch", action="store_true", help="start the watcher and the detached guard")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    loaded = load_env()
    print(f"[env] loaded {sorted(loaded)} from .env (values not shown)")
    missing = [k for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
               if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"submit_run: missing R2 variables {missing}")
    if creations() >= MAX_CREATIONS:
        raise SystemExit(f"submit_run: {creations()} pods already created this session (cap {MAX_CREATIONS})")
    rclone_env()

    remote = "cerberus_r2"
    job_id = f"newton-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    stage_dir = REPO / "outputs" / "newton_spike" / job_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle(args.branch, stage_dir / "cerberus-newton.bundle")
    inputs = build_inputs(stage_dir / "inputs")
    print(f"[bundle] {bundle} ({bundle.stat().st_size} bytes)")

    if args.dry_run:
        print(json.dumps({"job_id": job_id, "image": args.image, "gpus": args.gpu_type or DEFAULT_GPUS,
                          "env_keys": sorted(set(ENV_KEYS)), "bundle": str(bundle),
                          "inputs": {k: str(v) for k, v in inputs.items()}}, indent=2))
        return 0

    paths = upload(remote, job_id, bundle, inputs)
    import runpod_provider as rp

    bootstrap = base64.b64encode((REPO / "spikes/newton/newton_job.sh").read_bytes()).decode()
    env = rp.job_env(paths["bundle"], job_id, args.deadline_minutes, args.image)
    env["CERBERUS_CREATED_EPOCH"] = str(int(time.time()))
    env["CERBERUS_LAB_REF"] = "v3.0.0"
    status, body = rp.create_job_pod(
        os.environ["RUNPOD_API_KEY"], image=args.image, gpu_types=args.gpu_type or DEFAULT_GPUS,
        env=env, bootstrap_b64=bootstrap, name=job_id, disk_gb=args.disk_gb)
    if status >= 400:
        print(f"[create] failed HTTP {status}: {json.dumps(body)[:600]}", file=sys.stderr)
        return 1
    pod_id = body.get("id") or body.get("podId") or "unknown"
    record(pod_id, args.image)
    rate = float(args.rate)
    print(json.dumps({"job_id": job_id, "pod_id": pod_id, "image": args.image,
                      "cost_per_hr": body.get("costPerHr"), "created": creations()}, indent=2))

    if args.start_watch:
        guard = subprocess.run(
            [sys.executable, str(SCRIPTS / "session_guard.py"), "--provider", "runpod", "--detach",
             "--id", pod_id, "--rate", str(rate), "--started", str(int(time.time())),
             "--wall", "7200", "--hard-wall", "9000", "--usd", "2.5"],
            capture_output=True, text=True)
        print(f"[guard] {guard.stdout.strip() or guard.stderr.strip()}")
        watch_log = stage_dir / "watch.log"
        with watch_log.open("a") as log:
            subprocess.Popen([sys.executable, str(SCRIPTS / "watch_job.py"), "--job-id", job_id,
                              "--pod-id", pod_id, "--provider", "runpod", "--rate", str(rate),
                              "--deadline-minutes", str(args.deadline_minutes),
                              "--out", str(stage_dir / "artifacts")],
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"[watch] started; tail -f {watch_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
