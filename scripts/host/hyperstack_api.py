#!/usr/bin/env python3
"""A small Hyperstack REST client, used by the ``hyperstack_*.sh`` wrappers.

Only the calls this repository needs are implemented, each as one subcommand so the shell wrappers stay readable and
the request/response handling is testable without an account.  The API base is
``https://infrahub-api.nexgencloud.com/v1`` (``HYPERSTACK_API_BASE`` overrides it, which is how the tests point at a
fake server) and authentication is the ``api_key`` header.  The key is read from ``HYPERSTACK_API_KEY`` only; it is
never printed, and every JSON body is passed through ``scripts/redact.py`` before it reaches stdout.

Commands (all print JSON):
  get PATH | post PATH JSON | delete PATH          raw calls
  list-vms                                         every VM in the organisation
  find-vm --name N | --id I                        one VM, resolved through the list
  vm --id I                                        one VM by id
  wait-vm --id I [--timeout S]                     poll until ACTIVE and floating_ip is set
  delete-vm --id I [--timeout S]                   DELETE, then poll until the VM is gone
  price --flavor NAME [--volume-gb N]              hourly price from the pricebook, all-in
  pick-image --region R                            Ubuntu image with the highest driver build
  ensure-environment --region R [--name N]         the default-<region> environment, created if absent
  ensure-keypair --region R --name N --public-key-file P
  create-vm --payload FILE                         POST /core/virtual-machines, print instances[0]
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from redact import mask_text, masked_json

DEFAULT_BASE = "https://infrahub-api.nexgencloud.com/v1"
UBUNTU = re.compile(r"^Ubuntu", re.IGNORECASE)
DRIVER = re.compile(r"\bR(\d{3})(?:\.(\d{2}))?\b")


class ApiError(RuntimeError):
    """A non-2xx answer, or an answer that does not have the shape the caller needs."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {mask_text(message)[:800]}")
        self.status = status


def _base() -> str:
    return os.environ.get("HYPERSTACK_API_BASE", DEFAULT_BASE).rstrip("/")


def _key() -> str:
    key = os.environ.get("HYPERSTACK_API_KEY", "")
    if not key.strip():
        raise SystemExit(
            "HYPERSTACK_API_KEY is empty or unset; source .env in a subshell before calling the API"
        )
    return key


def request(method: str, path: str, payload: Any | None = None, *, timeout: float = 60.0) -> Any:
    """One API call.  Raises ApiError on a non-2xx answer; returns parsed JSON (or text)."""
    url = f"{_base()}/{path.lstrip('/')}"
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"api_key": _key(), "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise ApiError(exc.code, body) from None
    except urllib.error.URLError as exc:
        raise ApiError(0, f"cannot reach {url}: {exc.reason}") from None
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def get(path: str) -> Any:
    return request("GET", path)


def post(path: str, payload: Any) -> Any:
    return request("POST", path, payload)


def delete(path: str) -> Any:
    return request("DELETE", path)


def _rows(data: Any, *keys: str) -> list[dict]:
    """Find the first list under one of ``keys`` (the API nests lists under a name, not always the same one)."""
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def list_vms() -> list[dict]:
    rows: list[dict] = []
    page = 1
    while True:
        data = get(f"/core/virtual-machines?page={page}&page_size=100")
        batch = _rows(data, "instances")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
        page += 1
        if page > 50:
            return rows


def find_vm(*, vm_id: int | None = None, name: str | None = None) -> dict | None:
    if vm_id is not None:
        try:
            data = get(f"/core/virtual-machines/{int(vm_id)}")
        except ApiError as exc:
            if exc.status == 404:
                return None
            raise
        instance = data.get("instance") if isinstance(data, dict) else None
        return instance if isinstance(instance, dict) else None
    if not name:
        raise SystemExit("find-vm needs --id or --name")
    for vm in list_vms():
        if vm.get("name") == name:
            return vm
    return None


def wait_vm(vm_id: int, timeout: float, interval: float = 5.0) -> dict:
    """Wait for ACTIVE plus a floating IP.  A terminal bad state fails immediately."""
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        vm = find_vm(vm_id=vm_id)
        if vm is None:
            raise ApiError(404, f"VM {vm_id} disappeared while waiting for ACTIVE")
        last = vm
        status = str(vm.get("status", "")).upper()
        if status in {"ERROR", "DELETED", "DELETING"}:
            raise ApiError(0, f"VM {vm_id} entered {status}")
        if status == "ACTIVE" and vm.get("floating_ip"):
            return vm
        time.sleep(interval)
    detail = {"status": last.get("status"), "floating_ip": last.get("floating_ip")} if last else {}
    raise ApiError(0, f"VM {vm_id} was not ACTIVE with a floating IP within {timeout:.0f}s: {detail}")


def wait_gone(vm_id: int, timeout: float, interval: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        vm = find_vm(vm_id=vm_id)
        if vm is None:
            return
        if str(vm.get("status", "")).upper() == "DELETED":
            return
        time.sleep(interval)
    raise ApiError(0, f"VM {vm_id} still exists after {timeout:.0f}s")


def pricebook() -> dict[str, float]:
    data = get("/pricebook")
    rows = data if isinstance(data, list) else _rows(data, "pricebook", "prices")
    out: dict[str, float] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("name"):
            try:
                out[str(row["name"])] = float(row["value"])
            except (TypeError, ValueError):
                pass
    return out


def flavors() -> list[dict]:
    data = get("/core/flavors")
    flat: list[dict] = []
    for group in _rows(data, "data", "flavors"):
        for flavor in group.get("flavors", []) if isinstance(group.get("flavors"), list) else []:
            if isinstance(flavor, dict):
                flat.append(flavor)
    return flat


def find_flavor(name: str) -> dict:
    for flavor in flavors():
        if flavor.get("name") == name:
            return flavor
    if not name:
        raise SystemExit("flavor name is empty")
    raise ApiError(404, f"flavor {name} not found")


def price_for(flavor_name: str, volume_gb: int = 0, *, public_ip: bool = True) -> dict:
    flavor = find_flavor(flavor_name)
    prices = pricebook()
    gpu = str(flavor.get("gpu") or "")
    gpu_price = prices.get(gpu)
    if gpu_price is None:
        raise ApiError(0, f"no pricebook entry for GPU {gpu!r} (flavor {flavor_name})")
    disk_gb = int(flavor.get("disk") or 0) + int(volume_gb or 0)
    disk_price = prices.get("Cloud-SSD", 0.0) * disk_gb
    ip_price = prices.get("PublicIP", 0.0) if public_ip else 0.0
    total = gpu_price + disk_price + ip_price
    return {
        "flavor": flavor_name,
        "gpu": gpu,
        "gpu_count": int(flavor.get("gpu_count") or 1),
        "disk_gb": disk_gb,
        "disk_price_per_hour": round(disk_price, 6),
        "public_ip_price_per_hour": round(ip_price, 6),
        "total_per_hour": round(total, 6),
    }


def driver_rank(name: str) -> tuple[int, int]:
    match = DRIVER.search(name)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2) or 0))


def pick_image(region: str) -> dict:
    data = get("/core/images")
    candidates: list[dict] = []
    for group in _rows(data, "images"):
        if str(group.get("region_name", "")).upper() != region.upper():
            continue
        for image in group.get("images", []) if isinstance(group.get("images"), list) else []:
            if not isinstance(image, dict) or not UBUNTU.search(str(image.get("name", ""))):
                continue
            image = dict(image)
            image["driver_rank"] = list(driver_rank(str(image.get("name", ""))))
            candidates.append(image)
    if not candidates:
        raise ApiError(404, f"no Ubuntu image in {region}")
    candidates.sort(key=lambda i: (tuple(i["driver_rank"]), int(i.get("id") or 0)), reverse=True)
    best = candidates[0]
    return {
        "id": best.get("id"),
        "name": best.get("name"),
        "version": best.get("version"),
        "driver_rank": best["driver_rank"],
    }


def ensure_environment(region: str, name: str | None = None) -> dict:
    wanted = name or f"default-{region.upper()}"
    data = get("/core/environments")
    for env in _rows(data, "environments"):
        if env.get("name") == wanted:
            return env
    created = post("/core/environments", {"name": wanted, "region": region.upper()})
    env = created.get("environment") if isinstance(created, dict) else None
    return env if isinstance(env, dict) else {"name": wanted, "region": region.upper()}


def ensure_keypair(region: str, name: str, public_key_file: str) -> dict:
    public_key = Path(os.path.expanduser(public_key_file)).read_text(encoding="utf-8").strip()
    if not public_key.startswith(("ssh-", "ecdsa-", "sk-")):
        raise SystemExit(f"{public_key_file} does not look like a public key")
    data = get("/core/keypairs")
    for pair in _rows(data, "keypairs"):
        if pair.get("name") == name:
            return {"name": name, "created": False, "id": pair.get("id")}
    payload = {
        "name": name,
        "environment_name": f"default-{region.upper()}",
        "public_key": public_key,
    }
    created = post("/core/keypairs", payload)
    pair = created.get("keypair") if isinstance(created, dict) else None
    return {"name": name, "created": True, "id": (pair or {}).get("id")}


def valid_cidr(text: str) -> str:
    try:
        network = ipaddress.ip_network(text, strict=False)
    except ValueError as exc:
        raise SystemExit(f"invalid CIDR {text!r}: {exc}") from None
    if network.version != 4:
        raise SystemExit(f"the SSH rule must be IPv4, got {text!r}")
    return str(network)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("get", "delete"):
        p = sub.add_parser(name)
        p.add_argument("path")

    p = sub.add_parser("post")
    p.add_argument("path")
    p.add_argument("payload", help="JSON text or @file")

    sub.add_parser("list-vms")

    p = sub.add_parser("find-vm")
    p.add_argument("--id", type=int)
    p.add_argument("--name")

    p = sub.add_parser("vm")
    p.add_argument("--id", type=int, required=True)

    p = sub.add_parser("wait-vm")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--timeout", type=float, default=900.0)

    p = sub.add_parser("delete-vm")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--timeout", type=float, default=300.0)

    p = sub.add_parser("price")
    p.add_argument("--flavor", required=True)
    p.add_argument("--volume-gb", type=int, default=0)
    p.add_argument("--no-public-ip", action="store_true")

    p = sub.add_parser("pick-image")
    p.add_argument("--region", required=True)

    p = sub.add_parser("ensure-environment")
    p.add_argument("--region", required=True)
    p.add_argument("--name")

    p = sub.add_parser("ensure-keypair")
    p.add_argument("--region", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--public-key-file", required=True)

    p = sub.add_parser("create-vm")
    p.add_argument("--payload", required=True, help="JSON text or @file")

    p = sub.add_parser("validate-cidr")
    p.add_argument("cidr")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "get":
            out = get(args.path)
        elif args.cmd == "delete":
            out = delete(args.path)
        elif args.cmd == "post":
            payload = Path(args.payload[1:]).read_text() if args.payload.startswith("@") else json.loads(args.payload)
            out = post(args.path, payload)
        elif args.cmd == "list-vms":
            out = {"instances": list_vms()}
        elif args.cmd == "find-vm":
            out = find_vm(vm_id=args.id, name=args.name)
        elif args.cmd == "vm":
            out = find_vm(vm_id=args.id)
        elif args.cmd == "wait-vm":
            out = wait_vm(args.id, args.timeout)
        elif args.cmd == "delete-vm":
            delete(f"/core/virtual-machines/{args.id}")
            wait_gone(args.id, args.timeout)
            out = {"deleted": True, "id": args.id}
        elif args.cmd == "price":
            out = price_for(args.flavor, args.volume_gb, public_ip=not args.no_public_ip)
        elif args.cmd == "pick-image":
            out = pick_image(args.region)
        elif args.cmd == "ensure-environment":
            out = ensure_environment(args.region, args.name)
        elif args.cmd == "ensure-keypair":
            out = ensure_keypair(args.region, args.name, args.public_key_file)
        elif args.cmd == "create-vm":
            payload = Path(args.payload[1:]).read_text() if args.payload.startswith("@") else json.loads(args.payload)
            created = post("/core/virtual-machines", payload)
            instances = _rows(created, "instances")
            if not instances:
                raise ApiError(0, f"create returned no instance: {created}")
            out = instances[0]
        else:  # validate-cidr
            print(json.dumps({"cidr": valid_cidr(args.cidr)}))
            return 0
    except ApiError as exc:
        print(f"hyperstack_api: {exc}", file=sys.stderr)
        return 2
    print(masked_json(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
