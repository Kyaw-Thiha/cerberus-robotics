#!/usr/bin/env python3
"""Pull one job's artifacts from R2 and print the numbers the report needs.

    python3 spikes/newton/collect_results.py --job-id newton-...
    python3 spikes/newton/collect_results.py --job-id newton-... --dest /tmp/newton

Reads only: the job prefix ``jobs/<job_id>/`` on the configured R2 remote. Never prints credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def rclone_env() -> None:
    env_file = REPO / ".env"
    if not env_file.is_file():
        listing = subprocess.run(["git", "-C", str(REPO), "worktree", "list", "--porcelain"],
                                 capture_output=True, text=True).stdout
        for line in listing.splitlines():
            if line.startswith("worktree "):
                candidate = Path(line.split(" ", 1)[1]) / ".env"
                if candidate.is_file():
                    env_file = candidate
                    break
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_TYPE", "s3")
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_PROVIDER", "Cloudflare")
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID", os.environ.get("R2_ACCESS_KEY_ID", ""))
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY", os.environ.get("R2_SECRET_ACCESS_KEY", ""))
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_ENDPOINT",
                          f"https://{os.environ.get('R2_ACCOUNT_ID', '')}.r2.cloudflarestorage.com")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--dest", default=None)
    args = parser.parse_args(argv)
    rclone_env()
    bucket = os.environ["R2_BUCKET"]
    dest = Path(args.dest) if args.dest else REPO / "outputs" / "newton_spike" / args.job_id / "r2"
    dest.mkdir(parents=True, exist_ok=True)
    base = f"cerberus_r2:{bucket}/jobs/{args.job_id}"
    result = subprocess.run(["rclone", "copy", base, str(dest), "--no-check-dest"], capture_output=True, text=True)
    print(f"[pull] rc={result.returncode} -> {dest}")
    if result.returncode != 0:
        print(result.stderr[-500:])

    def show(name: str, rel: str, keys: list[str] | None = None) -> None:
        path = dest / rel
        if not path.is_file():
            print(f"[{name}] missing ({rel})")
            return
        try:
            data = json.loads(path.read_text())
        except ValueError:
            print(f"[{name}] not json")
            return
        print(f"[{name}] {rel}")
        if keys:
            for key in keys:
                value = data
                for part in key.split("."):
                    value = value.get(part, {}) if isinstance(value, dict) else None
                print(f"    {key} = {value}")
        else:
            print(json.dumps(data, indent=2)[:4000])

    show("host", "outputs/host.json")
    show("smoke", "outputs/stage1/smoke.json")
    launches = dest / "outputs/stage1/launches.csv"
    if launches.is_file():
        rows = launches.read_text().strip().splitlines()[1:]
        ok = sum(1 for row in rows if row.endswith(",1"))
        print(f"[launches] {ok}/{len(rows)} started: {rows}")
    show("gl", "outputs/stage2/gl_probe.json")
    show("video", "outputs/stage2/video.json")
    show("parity", "outputs/stage3/parity.json", ["obs_shape", "contract.all_passed", "contract.checks"])
    for order in ("none", "contract", "physx", "mjwarp"):
        show(f"transfer[{order}]", f"outputs/stage4_{order}/transfer.json",
             ["policy.source", "rollout.fall_rate", "rollout.mean_abs_vx_error_mps",
              "rollout.median_steps_to_first_fall", "per_level"])
    show("scene-parse", "outputs/stage5/usd_probe.json")
    show("scene-load", "outputs/stage5/scene_load.json", ["run", "spawn", "spawn_source"])
    rc = dest / "outputs/stage5/load_rc.txt"
    if rc.is_file():
        print(f"[scene-load] rc={rc.read_text().strip()}")
    log = dest / "job.log"
    if log.is_file():
        text = log.read_text(errors="replace")
        print(f"[log] {len(text)} bytes; last lines:")
        print("\n".join(text.strip().splitlines()[-25:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
