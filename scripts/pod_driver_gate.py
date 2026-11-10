#!/usr/bin/env python3
"""Host-driver gate: refuse to run on a driver build that segfaults Isaac Sim 6.x.

Why this exists: the H4 pod session lost its physics runs to a startup SIGSEGV in ``libcarb.tasking`` (exit 139) on
driver 595.91.07, on three different hosts. That matches NVIDIA's 595.71.05/595.79/595.84 failures; 595.58.03 is the
validated 595 build and the 580 branch from 580.95.05 is validated by the Isaac Lab 3.0 docs. The host driver is the
variable, so the check runs before anything is downloaded or launched.

Rules live in ``scripts/pod_driver_allowlist.json`` (not hardcoded) with the evidence beside them.

On reject: one-line verdict with the exact driver string, a row in ``outputs/launch_probe.csv``, and exit 2. A pod that
fails this gate holds no data (nothing has been downloaded yet), so it may be terminated immediately through the
provider API -- the one owner-approved exception to "never terminate a pod".
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = Path(__file__).resolve().parent / "pod_driver_allowlist.json"
CSV_PATH = REPO_ROOT / "outputs" / "launch_probe.csv"
CSV_COLUMNS = ("ts", "stage", "driver", "cuda", "gpu", "host", "datacenter", "image", "verdict", "note")
RUNPOD_API = "https://rest.runpod.io/v1/pods"
EXIT_REJECT = 2


@dataclass(frozen=True)
class HostInfo:
    driver: str
    cuda: str
    gpu: str
    host: str
    datacenter: str
    image: str


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", text)
    return tuple(int(p) for p in parts[:3]) if parts else ()


def driver_allowed(driver: str, allowlist: dict) -> tuple[bool, str]:
    """``(allowed, reason)`` for an exact driver version string such as ``595.58.03``."""
    version = _version_tuple(driver)
    if len(version) < 2:
        return False, "unparseable driver version"
    branch = str(version[0])
    for rule in allowlist.get("rules", []):
        if str(rule.get("branch")) != branch:
            continue
        if "exact_version" in rule:
            if driver.strip() == rule["exact_version"]:
                return True, rule.get("why", "exact match")
            return False, f"595 branch only allows {rule['exact_version']}"
        if "min_version" in rule:
            if version >= _version_tuple(rule["min_version"]):
                return True, rule.get("why", "minimum match")
            return False, f"580 branch needs >= {rule['min_version']}"
    return False, f"branch {branch}.x is not allowlisted"


def parse_cuda_version(nvidia_smi_text: str) -> str:
    """The ``CUDA Version: X.Y`` shown in the nvidia-smi header, or ``unknown``."""
    found = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_text)
    return found.group(1) if found else "unknown"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_host_info(image: str | None = None) -> HostInfo:
    """Everything the verdict and the CSV need, read from the host."""
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()
    gpu = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()
    header = _run(["nvidia-smi"])
    return HostInfo(
        driver=(driver[0].strip() if driver else "unknown"),
        cuda=parse_cuda_version(header),
        gpu=(gpu[0].strip() if gpu else "unknown"),
        host=platform.node() or "unknown",
        datacenter=(os.environ.get("RUNPOD_DATACENTER_ID") or os.environ.get("RUNPOD_DC_ID") or "unknown"),
        image=(image or os.environ.get("CERBERUS_IMAGE_TAG") or "unknown"),
    )


def append_probe_row(path: Path, info: HostInfo, *, stage: str, verdict: str, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stage": stage,
            "driver": info.driver,
            "cuda": info.cuda,
            "gpu": info.gpu,
            "host": info.host,
            "datacenter": info.datacenter,
            "image": info.image,
            "verdict": verdict,
            "note": note,
        })


def terminate_pod(pod_id: str, api_key: str) -> tuple[bool, str]:
    """DELETE the pod through the provider API. Only called when the gate rejects (no data on the pod yet)."""
    request = urllib.request.Request(
        f"{RUNPOD_API}/{pod_id}", method="DELETE",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status < 300, f"HTTP {response.status}"
    except (urllib.error.URLError, OSError) as e:
        return False, f"{type(e).__name__}: {e}"


def main(argv: list[str] | None = None) -> int:
    from pod_guard import require_pod

    require_pod("pod_driver_gate.sh")
    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    info = collect_host_info()
    allowed, reason = driver_allowed(info.driver, allowlist)
    verdict = "ALLOW" if allowed else "REJECT"
    line = (f"DRIVER GATE: {verdict} driver={info.driver} cuda={info.cuda} gpu={info.gpu} "
            f"datacenter={info.datacenter} image={info.image} host={info.host} ({reason})")
    print(line)
    append_probe_row(CSV_PATH, info, stage="gate", verdict=verdict, note=reason)
    if allowed:
        return 0

    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if pod_id and api_key:
        ok, detail = terminate_pod(pod_id, api_key)
        print(f"DRIVER GATE: terminating {pod_id} (no data on it yet): {'ok' if ok else 'failed'} ({detail})")
        append_probe_row(CSV_PATH, info, stage="gate", verdict="TERMINATED" if ok else "TERMINATE_FAILED", note=detail)
    else:
        print("DRIVER GATE: no RUNPOD_POD_ID/RUNPOD_API_KEY; terminate the pod manually")
    return EXIT_REJECT


if __name__ == "__main__":
    sys.exit(main())
