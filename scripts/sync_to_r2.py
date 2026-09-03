#!/usr/bin/env python3
"""Sync training artifacts (checkpoints, videos, tensorboard events, params,
git diffs, eval sweep results) up to Cloudflare R2 for permanent storage.
Plain Python + `rclone` -- Cloudflare's own docs recommend rclone specifically
for "bulk object operations, migrations, and syncing directories" (our exact
use case) over the AWS CLI or Wrangler. No Isaac Sim dependency, runs fine
from a pod's system Python or locally. rclone is configured entirely via env
vars (RCLONE_CONFIG_R2_*, set from R2_* below) -- no `rclone config` needed.

R2 layout mirrors the local layout under a per-module prefix, so this script
is reusable for future phases beyond locomotion, not locomotion-specific
(see docs/project_structure.md's "one directory per stack component" rule --
--module is how a future e.g. manipulation run reuses this same script):

    r2://<bucket>/<module>/<experiment_name>/<run_timestamp>/...
    r2://<bucket>/<module>/<experiment_name>/latest.json   -- {"run": "<run_timestamp>"}

e.g. a Go2 flat run at policy/locomotion/checkpoints/unitree_go2_flat/2026-09-05_14-32-10/
uploads to r2://<bucket>/locomotion/unitree_go2_flat/2026-09-05_14-32-10/, and
updates r2://<bucket>/locomotion/unitree_go2_flat/latest.json to point at it.

Two subcommands, because eval doesn't always run alongside training (often
later, sometimes on a different pod, against a checkpoint that isn't sitting
next to a full local run tree anymore):

    sync-run   -- uploads a full local run directory (checkpoints, videos,
                  tensorboard events, params, git diffs). Also updates
                  latest.json for that experiment.
    sync-eval  -- uploads just eval output file(s) to an *existing* run's
                  .../eval/ folder on R2, derived from the --checkpoint path
                  used to run push_recovery_eval.py (must be the canonical
                  local path: .../checkpoints/<experiment_name>/<run_timestamp>/model_X.pt)
                  -- no full run tree needs to be present locally.

Requires `rclone` (v1.59+) and these vars set (e.g. in .env, sourced with
`set -a; source .env; set +a` before running -- same pattern as WANDB_API_KEY):
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET

Usage:
    python3 scripts/sync_to_r2.py sync-run \
        --module locomotion \
        --run-dir policy/locomotion/checkpoints/unitree_go2_flat/2026-09-05_14-32-10

    python3 scripts/sync_to_r2.py sync-eval \
        --module locomotion \
        --checkpoint policy/locomotion/checkpoints/unitree_go2_flat/2026-09-05_14-32-10/model_250.pt \
        --eval-file policy/locomotion/eval/results/push_recovery_2026-09-06_09-00-00.csv

Standalone for now, invoked manually -- not yet wired into an automated
training-completion wrapper (see policy_locomotion_status.md's automation
design discussion).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

REQUIRED_ENV_VARS = ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"]
REMOTE = "cerberus_r2"  # rclone remote name, only referenced via env vars below -- never written to disk


def _require_env() -> dict[str, str]:
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            f"Missing required env vars: {', '.join(missing)}. Set them in .env and "
            "`set -a; source .env; set +a` before running this script."
        )
    return {v: os.environ[v] for v in REQUIRED_ENV_VARS}


def _rclone_env(env: dict[str, str]) -> dict[str, str]:
    # rclone reads an entirely env-var-defined remote via RCLONE_CONFIG_<REMOTE>_<KEY> --
    # no `rclone config` / config file needed, matches sourcing everything from .env.
    rclone_env = os.environ.copy()
    prefix = f"RCLONE_CONFIG_{REMOTE.upper()}_"
    rclone_env[f"{prefix}TYPE"] = "s3"
    rclone_env[f"{prefix}PROVIDER"] = "Cloudflare"
    rclone_env[f"{prefix}ACCESS_KEY_ID"] = env["R2_ACCESS_KEY_ID"]
    rclone_env[f"{prefix}SECRET_ACCESS_KEY"] = env["R2_SECRET_ACCESS_KEY"]
    rclone_env[f"{prefix}ENDPOINT"] = f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    rclone_env[f"{prefix}ACL"] = "private"
    # avoids a bucket-existence check that can fail for an Object-scoped (not Account-scoped) API token
    rclone_env[f"{prefix}NO_CHECK_BUCKET"] = "true"
    return rclone_env


def _run(cmd: list[str], env: dict[str, str], **kwargs) -> None:
    print(f"[INFO] {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True, **kwargs)


def _write_latest_pointer(bucket: str, module: str, experiment_name: str, run_timestamp: str, rclone_env: dict) -> None:
    latest_key = f"{REMOTE}:{bucket}/{module}/{experiment_name}/latest.json"
    with tempfile.TemporaryDirectory() as tmp:
        latest_path = Path(tmp) / "latest.json"
        latest_path.write_text(json.dumps({"run": run_timestamp}))
        _run(["rclone", "copyto", str(latest_path), latest_key], env=rclone_env)
    print(f"[INFO] Updated {latest_key} -> {run_timestamp}")


def sync_run(module: str, run_dir: Path) -> None:
    env = _require_env()
    rclone_env = _rclone_env(env)
    bucket = env["R2_BUCKET"]

    experiment_name = run_dir.resolve().parent.name  # e.g. "unitree_go2_flat"
    run_timestamp = run_dir.resolve().name  # e.g. "2026-09-05_14-32-10"
    remote_prefix = f"{REMOTE}:{bucket}/{module}/{experiment_name}/{run_timestamp}/"

    _run(["rclone", "sync", str(run_dir), remote_prefix, "--progress"], env=rclone_env)
    _write_latest_pointer(bucket, module, experiment_name, run_timestamp, rclone_env)


def sync_eval(module: str, checkpoint: Path, eval_files: list[Path]) -> None:
    env = _require_env()
    rclone_env = _rclone_env(env)
    bucket = env["R2_BUCKET"]

    if checkpoint.suffix != ".pt":
        raise SystemExit(
            f"--checkpoint doesn't look like a canonical checkpoint path "
            f"(.../checkpoints/<experiment_name>/<run_timestamp>/model_X.pt): {checkpoint}"
        )
    run_dir = checkpoint.resolve().parent  # .../<experiment_name>/<run_timestamp>/
    experiment_name = run_dir.parent.name
    run_timestamp = run_dir.name

    for eval_file in eval_files:
        remote_target = f"{REMOTE}:{bucket}/{module}/{experiment_name}/{run_timestamp}/eval/{eval_file.name}"
        _run(["rclone", "copyto", str(eval_file), remote_target], env=rclone_env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync training artifacts to Cloudflare R2.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("sync-run", help="Upload a full local run directory.")
    run_parser.add_argument("--module", type=str, required=True, help="Top-level R2 prefix, e.g. 'locomotion'.")
    run_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the run's log_dir, e.g. policy/locomotion/checkpoints/unitree_go2_flat/<timestamp>/",
    )

    eval_parser = subparsers.add_parser("sync-eval", help="Upload eval output to an existing run's eval/ folder.")
    eval_parser.add_argument("--module", type=str, required=True, help="Top-level R2 prefix, e.g. 'locomotion'.")
    eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="The --checkpoint path used to run push_recovery_eval.py -- used to locate the run on R2.",
    )
    eval_parser.add_argument(
        "--eval-file",
        type=Path,
        required=True,
        action="append",
        dest="eval_files",
        help="An eval output file to upload -- repeat for multiple (e.g. the summary and per-trial CSVs).",
    )

    args = parser.parse_args()

    if args.command == "sync-run":
        if not args.run_dir.is_dir():
            raise SystemExit(f"--run-dir does not exist or is not a directory: {args.run_dir}")
        sync_run(args.module, args.run_dir)
    elif args.command == "sync-eval":
        missing = [f for f in args.eval_files if not f.is_file()]
        if missing:
            raise SystemExit(f"--eval-file(s) do not exist: {missing}")
        sync_eval(args.module, args.checkpoint, args.eval_files)


if __name__ == "__main__":
    main()
