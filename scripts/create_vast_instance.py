#!/usr/bin/env python3
"""Create the cheapest matching Vast.ai instance, with a live self-test of the stop path.

Offer rules (client-side after a raw fetch): verified, rentable, reliability > 0.95, one GPU with >= 24 GB VRAM,
enough disk, an open port, and driver_version 580.x (>= 580.95.05, < 590) or exactly 595.58.03. Price cap 1.00 USD/h
unless the owner pre-approved otherwise. If nothing matches, the search summary is printed and the script exits 2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import vast_provider as vp

JOB_ENV_KEYS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
                "CERBERUS_CODE_BUNDLE", "CERBERUS_JOB_ID", "CERBERUS_DEADLINE_MINUTES", "CERBERUS_IMAGE_TAG")


def job_env(job_id: str, code_bundle: str, deadline_minutes: int, image: str) -> dict[str, str]:
    """Only the restricted R2 credentials and the four job variables; never VAST/WANDB/RUNPOD or the whole .env."""
    env = {key: os.environ.get(key, "") for key in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")}
    missing = [k for k, v in env.items() if not v]
    if missing:
        raise SystemExit(f"missing R2 variables: {', '.join(missing)}")
    env.update({"CERBERUS_CODE_BUNDLE": code_bundle, "CERBERUS_JOB_ID": job_id,
                "CERBERUS_DEADLINE_MINUTES": str(deadline_minutes), "CERBERUS_IMAGE_TAG": image})
    return env


def job_bootstrap() -> str:
    """Tiny in-container bootstrap: rclone config, fetch the bundle, run scripts/job_entrypoint.sh from it."""
    return """set -e
export RCLONE_CONFIG_CERBERUS_R2_TYPE=s3
export RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
export RCLONE_CONFIG_CERBERUS_R2_ACL=private
export RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true
echo "[bootstrap] env: $(env | grep -cE '^CERBERUS_|^R2_') vars"
rclone copyto "cerberus_r2:$R2_BUCKET/$CERBERUS_CODE_BUNDLE" /root/code.bundle
GIT_LFS_SKIP_SMUDGE=1 git clone /root/code.bundle /workspace/cerberus
cd /workspace/cerberus
exec bash scripts/job_entrypoint.sh
"""


def build_job_payload(chosen: dict, *, image: str, disk_gb: int, label: str, env: dict[str, str],
                      deadline_minutes: int) -> dict:
    """Vast args/entrypoint launch mode: the image is untouched, our bootstrap runs from `args`."""
    return {"image": image, "disk": float(disk_gb), "runtype": "args", "label": label,
            "args": ["bash", "-lc", job_bootstrap()],
            "env": env, "onstart": "",
            "price": float(chosen.get("dph_total", 0.0))}


def summarise(offers: list[dict]) -> dict:
    by_gpu: dict[str, int] = {}
    for offer in offers:
        name = str(offer.get("gpu_name", "?"))
        by_gpu[name] = by_gpu.get(name, 0) + 1
    prices = sorted(vp.price(o) for o in offers)
    return {"count": len(offers), "by_gpu": by_gpu,
            "min_price": prices[0] if prices else None, "max_price": prices[-1] if prices else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-price", type=float, default=vp.MAX_PRICE_USD_PER_H)
    parser.add_argument("--disk-gb", type=int, default=vp.DEFAULT_DISK_GB)
    parser.add_argument("--image", default=vp.DEFAULT_IMAGE)
    parser.add_argument("--label", default="cerberus-h4")
    parser.add_argument("--wait", type=float, default=600.0, help="seconds to wait for ssh details")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--code-bundle", default=None, help="R2 path of the code bundle (migration/.../code/....bundle)")
    parser.add_argument("--deadline-minutes", type=int, default=85)
    parser.add_argument("--dry-run", action="store_true", help="search and print, create nothing")
    args = parser.parse_args(argv)
    key = os.environ.get("VAST_API_KEY", "")
    if not key:
        print("create_vast: VAST_API_KEY is not set", file=sys.stderr)
        return 2

    status, offers, note = vp.search_offers(key)
    if status != 200:
        print(f"create_vast: offer search failed HTTP {status}: {note}", file=sys.stderr)
        return 1
    selected = vp.select_offers(offers, args.max_price)
    if not selected:
        print(json.dumps({"matched": 0, "raw": summarise(offers),
                          "rejections": [{"id": o.get("id"), "price": vp.price(o),
                                          "why": vp.offer_ok(o)[1]} for o in offers[:20]]}, indent=2))
        return 2
    chosen = selected[0]
    print(f"create_vast: chosen ask={chosen.get('id')} gpu={chosen.get('gpu_name')} "
          f"driver={chosen.get('driver_version')} price={vp.price(chosen):.2f}/h "
          f"matches={len(selected)}", flush=True)
    if args.dry_run:
        print(json.dumps({"action": "dry-run", "chosen": chosen}, indent=2)[:3000])
        return 0

    job_id = args.job_id or f"vast-{int(time.time())}"
    bundle = args.code_bundle or os.environ.get("CERBERUS_CODE_BUNDLE", "")
    if not bundle:
        print("create_vast: --code-bundle (or CERBERUS_CODE_BUNDLE) is required in job mode", file=sys.stderr)
        return 2
    env = job_env(job_id, bundle, args.deadline_minutes, args.image)
    payload = build_job_payload(chosen, image=args.image, disk_gb=args.disk_gb, label=args.label,
                                env=env, deadline_minutes=args.deadline_minutes)
    status, body = vp.create_instance(key, chosen["id"], payload=payload)
    if status >= 400 or not body.get("success", True):
        print(f"create_vast: create failed HTTP {status}: {json.dumps(body)[:400]}", file=sys.stderr)
        return 1
    instance_id = str(body.get("new_contract") or body.get("id") or "")
    print(f"create_vast: created instance {instance_id}", flush=True)

    # live self-test of the stop path's authentication/headers before anything depends on it
    st_status, _ = vp.instance_info(key, instance_id)
    if st_status != 200:
        print(f"create_vast: STOP-PATH SELF-TEST FAILED (HTTP {st_status}); stopping instance", file=sys.stderr)
        vp.stop_instance(key, instance_id)
        return 4
    print(f"create_vast: stop-path self-test OK (GET instance/{instance_id} -> 200)", flush=True)

    deadline = time.time() + args.wait
    info: dict = {}
    while time.time() < deadline:
        _, info = vp.instance_info(key, instance_id)
        if info.get("ssh_host") and info.get("ssh_port"):
            break
        time.sleep(10)
    result = {"instance_id": instance_id, "ssh_host": info.get("ssh_host"), "ssh_port": info.get("ssh_port"),
              "status": info.get("actual_status") or info.get("status_msg"), "dph_total": vp.price(chosen),
              "gpu": chosen.get("gpu_name"), "driver": chosen.get("driver_version")}
    print(json.dumps(result, indent=2))
    if not (info.get("ssh_host") and info.get("ssh_port")):
        print("create_vast: no ssh details within the wait; instance left running", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
