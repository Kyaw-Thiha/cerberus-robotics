"""Vast.ai provider glue: offer filtering, instance creation, stop/destroy and the session guard decisions.

Everything that touches HTTP goes through an injectable transport so the logic is tested without a network call; the
CLIs (``create_vast_instance.py``, ``vast_stop.py``, ``session_guard.py``) are thin wrappers.

The API key is read from ``VAST_API_KEY`` and never printed. All requests send the same ``User-Agent`` as the stop
path, because a missing UA is exactly what broke the RunPod dead-man (Cloudflare 1010).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

API = "https://console.vast.ai/api/v0"
USER_AGENT = "cerberus-vast-tools/1.0"
DEFAULT_IMAGE = "kevinkyawthiha/cerberus-remote-v6:v6-ga"
DEFAULT_DISK_GB = 80
MIN_VRAM_MB = 24000
MIN_RELIABILITY = 0.95
MIN_INET_DOWN_MBPS = 700.0
MAX_PRICE_USD_PER_H = 1.00
SESSION_WALL_S = 90 * 60
SESSION_HARD_WALL_S = 3 * 3600
USD_CAP = 3.00


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", str(text or ""))
    return tuple(int(p) for p in parts[:3]) if parts else ()


def driver_ok(driver: str) -> bool:
    """Accept 580.x from 580.95.05 up to (excluding) 590, or exactly 595.58.03."""
    v = _version_tuple(driver)
    if len(v) < 2:
        return False
    if v[0] == 580:
        return v >= _version_tuple("580.95.05")
    return str(driver).strip() == "595.58.03"


def _get(offer: dict, *names, default=None):
    for name in names:
        if name in offer and offer[name] is not None:
            return offer[name]
    return default


def open_port(offer: dict) -> bool:
    """A rentable offer must expose a TCP port for SSH; fields differ across API versions."""
    ports = _get(offer, "ports", "port_forwards")
    if isinstance(ports, (list, tuple)):
        return len(ports) > 0
    if isinstance(ports, str) and ports.strip():
        return True
    count = _get(offer, "direct_port_count", "port_count")
    if count is not None:
        try:
            return int(count) > 0
        except (TypeError, ValueError):
            return False
    return True  # field absent: do not exclude on a guess


def offer_ok(offer: dict) -> tuple[bool, str]:
    if not bool(_get(offer, "verified", "verification") in (True, "verified")):
        return False, "not verified"
    if not bool(_get(offer, "rentable", default=False)):
        return False, "not rentable"
    reliability = _get(offer, "reliability2", "reliability")
    try:
        if reliability is None or float(reliability) <= MIN_RELIABILITY:
            return False, f"reliability {reliability}"
    except (TypeError, ValueError):
        return False, "unparseable reliability"
    try:
        if int(_get(offer, "num_gpus", default=0)) != 1:
            return False, "not one GPU"
    except (TypeError, ValueError):
        return False, "unparseable gpu count"
    try:
        if int(_get(offer, "gpu_ram", "gpu_ram_mb", default=0)) < MIN_VRAM_MB:
            return False, "VRAM below 24 GB"
    except (TypeError, ValueError):
        return False, "unparseable VRAM"
    try:
        if float(_get(offer, "disk_space", "disk_space_gb", default=0)) < DEFAULT_DISK_GB:
            return False, "disk too small"
    except (TypeError, ValueError):
        return False, "unparseable disk"
    try:
        if float(_get(offer, "inet_down", default=0)) < MIN_INET_DOWN_MBPS:
            return False, "download speed below 700 Mbps"
    except (TypeError, ValueError):
        return False, "unparseable inet_down"
    if not open_port(offer):
        return False, "no open port"
    driver = _get(offer, "driver_version", "driver_ver", default="")
    if not driver_ok(driver):
        return False, f"driver {driver!r} not allowed"
    return True, "ok"


def price(offer: dict) -> float:
    try:
        return float(_get(offer, "dph_total", "dph_total_usd", default=1e9))
    except (TypeError, ValueError):
        return 1e9


def select_offers(offers: list[dict], max_price: float = MAX_PRICE_USD_PER_H) -> list[dict]:
    """Client-side filter (driver version included) then cheapest first."""
    kept = [o for o in offers if offer_ok(o)[0] and price(o) <= max_price]
    return sorted(kept, key=price)


def select_distinct_hosts(offers: list[dict], n: int = 3, max_price: float = MAX_PRICE_USD_PER_H) -> list[dict]:
    """Cheapest matching offer per distinct host, up to ``n`` hosts, cheapest first."""
    best: dict[str, dict] = {}
    for offer in select_offers(offers, max_price):
        host = str(_get(offer, "host_id", "machine_id", default=offer.get("id")))
        if host not in best:
            best[host] = offer
    return list(best.values())[:n]


@dataclass
class Instance:
    id: str
    ssh_host: str = ""
    ssh_port: int = 0
    dph_total: float = 0.0
    raw: dict = field(default_factory=dict)


Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, str]]


def http_request(url: str, method: str, headers: dict[str, str], body: bytes | None) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}"


def headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "Accept": "application/json", "User-Agent": USER_AGENT}


def search_offers(api_key: str, *, transport: Transport | None = None) -> tuple[int, list[dict], str]:
    """Fetch raw offers; filtering happens client-side. Returns ``(status, offers, note)``."""
    query = json.dumps({"rentable": {"eq": True}, "num_gpus": {"eq": 1}})
    url = f"{API}/bundles/?q={urllib.parse.quote(query)}"
    status, text = (transport or http_request)(url, "GET", headers(api_key), None)
    if status != 200:
        return status, [], text[:300]
    try:
        payload = json.loads(text)
    except ValueError:
        return status, [], "unparseable offers payload"
    offers = payload.get("offers", payload) if isinstance(payload, dict) else payload
    return status, list(offers or []), ""


def create_instance(api_key: str, ask_id: str | int, *, image: str = DEFAULT_IMAGE,
                    disk_gb: int = DEFAULT_DISK_GB, label: str = "cerberus",
                    env: dict[str, str] | None = None, payload: dict | None = None,
                    transport: Transport | None = None) -> tuple[int, dict]:
    body = dict(payload) if payload is not None else {
        "image": image, "disk": float(disk_gb), "runtype": "ssh", "label": label,
        "onstart": "", "env": dict(env or {})}
    url = f"{API}/asks/{ask_id}/"
    status, text = (transport or http_request)(url, "PUT", headers(api_key), json.dumps(body).encode())
    try:
        return status, json.loads(text or "{}")
    except ValueError:
        return status, {"raw": text[:500]}


def instance_info(api_key: str, instance_id: str, *, transport: Transport | None = None) -> tuple[int, dict]:
    status, text = (transport or http_request)(f"{API}/instances/{instance_id}/", "GET", headers(api_key), None)
    try:
        payload = json.loads(text or "{}")
    except ValueError:
        return status, {}
    inst = payload.get("instances", payload) if isinstance(payload, dict) else payload
    if isinstance(inst, list):
        inst = inst[0] if inst else {}
    return status, inst if isinstance(inst, dict) else {}


def start_instance(api_key: str, instance_id: str, *, transport: Transport | None = None) -> tuple[int, str]:
    return (transport or http_request)(f"{API}/instances/{instance_id}/", "PUT", headers(api_key),
                                       json.dumps({"state": "running"}).encode())


def stop_instance(api_key: str, instance_id: str, *, transport: Transport | None = None) -> tuple[int, str]:
    """Stop (not destroy): ``PUT /instances/{id}/ {"state": "stopped"}`` with the shared headers."""
    return (transport or http_request)(f"{API}/instances/{instance_id}/", "PUT", headers(api_key),
                                       json.dumps({"state": "stopped"}).encode())


def destroy_instance(api_key: str, instance_id: str, *, transport: Transport | None = None) -> tuple[int, str]:
    return (transport or http_request)(f"{API}/instances/{instance_id}/", "DELETE", headers(api_key), None)


@dataclass
class GuardDecision:
    action: str  # "wait" | "stop"
    reason: str
    remaining_s: float
    spend_usd: float


def guard_decision(*, started_epoch: float, now_epoch: float, rate_per_hour: float,
                   wall_s: float = SESSION_WALL_S, usd_cap: float = USD_CAP) -> GuardDecision:
    """Stop at the wall clock or the USD cap, whichever comes first."""
    elapsed = max(0.0, now_epoch - started_epoch)
    spend = elapsed / 3600.0 * float(rate_per_hour)
    wall_remaining = wall_s - elapsed
    if spend >= usd_cap:
        return GuardDecision("stop", f"USD cap {usd_cap:.2f} reached ({spend:.2f})", wall_remaining, spend)
    if wall_remaining <= 0:
        return GuardDecision("stop", f"wall clock {wall_s / 60:.0f} min reached", wall_remaining, spend)
    return GuardDecision("wait", "", wall_remaining, spend)
