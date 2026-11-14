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

    status, body = vp.create_instance(key, chosen["id"], image=args.image, disk_gb=args.disk_gb,
                                      label=args.label, env={"CERBERUS_ON_POD": "1",
                                                             "CERBERUS_IMAGE_TAG": args.image})
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
