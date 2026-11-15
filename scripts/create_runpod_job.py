#!/usr/bin/env python3
"""Create a RunPod job pod: no SSH, the image's entrypoint overridden to run the job bootstrap.

The long bootstrap (``scripts/job_entrypoint.sh``) travels base64-encoded in ``CERBERUS_BOOTSTRAP_B64``; the pod's
start command only decodes and runs it. Env is the restricted set from ``runpod_provider.job_env``. The first attempt
sends ``allowedCudaVersions: ["13.0"]`` (fails fast when there is no such stock), the second is filter-free and lets
the driver gate decide. Nothing is stopped here; the job stops itself and the laptop watcher/guard are the backstops.

    python3 scripts/create_runpod_job.py --job-id runpod-... --code-bundle migration/.../code/cerberus-main.bundle
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "kevinkyawthiha/cerberus-remote-v6:v6-ga"
DEFAULT_GPUS = ["NVIDIA A40", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA L40S", "NVIDIA RTX A6000"]
MAX_CREATIONS = 3


def main(argv: list[str] | None = None) -> int:
    import runpod_provider as rp

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--code-bundle", required=True, help="path inside the R2 bucket")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--deadline-minutes", type=int, default=90)
    parser.add_argument("--disk-gb", type=int, default=100)
    parser.add_argument("--gpu-type", action="append", default=None)
    parser.add_argument("--no-cuda-filter", action="store_true", help="skip the one filtered attempt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key and not args.dry_run:
        print("create_runpod_job: RUNPOD_API_KEY is not set", file=sys.stderr)
        return 2
    if len(rp.creations()) >= MAX_CREATIONS:
        print(f"create_runpod_job: {len(rp.creations())} pods already created this session (cap {MAX_CREATIONS})",
              file=sys.stderr)
        return 3

    bootstrap = base64.b64encode((REPO / "scripts" / "job_entrypoint.sh").read_bytes()).decode()
    env = rp.job_env(args.code_bundle, args.job_id, args.deadline_minutes, args.image)
    if args.dry_run:
        preview = {"image": args.image, "gpus": args.gpu_type or DEFAULT_GPUS,
                   "env_keys": sorted(env) + ["CERBERUS_BOOTSTRAP_B64"], "bootstrap_bytes": len(bootstrap)}
        print(json.dumps(preview, indent=2))
        return 0

    status, body = rp.create_job_pod(
        api_key, image=args.image, gpu_types=args.gpu_type or DEFAULT_GPUS, env=env,
        bootstrap_b64=bootstrap, name=args.job_id, disk_gb=args.disk_gb,
        cuda_first=None if args.no_cuda_filter else "13.0")
    if status >= 400:
        print(f"create_runpod_job: failed HTTP {status}: {json.dumps(body)[:500]}", file=sys.stderr)
        return 1
    pod_id = body.get("id") or body.get("podId") or "unknown"
    rp.record_creation(pod_id, args.image, [] if args.no_cuda_filter else ["13.0"])
    print(json.dumps({"pod_id": pod_id, "name": body.get("name"), "cost_per_hr": body.get("costPerHr"),
                      "image": args.image, "created": len(rp.creations())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
