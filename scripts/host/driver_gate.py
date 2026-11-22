#!/usr/bin/env python3
"""Host-driver gate: refuse to run on a driver build that segfaults Isaac Sim 6.x.

Why this exists: the H4 cloud session lost its physics runs to a startup SIGSEGV in ``libcarb.tasking`` (exit 139) on
driver 595.91.07, on three different hosts. That matches NVIDIA's 595.71.05/595.79/595.84 failures; 595.58.03 is the
validated 595 build and the 580 branch from 580.95.05 is validated by the Isaac Lab 3.0 docs. The host driver is the
variable, so the check runs before anything is downloaded or launched.

Rules live in ``scripts/host/driver_allowlist.json`` (not hardcoded) with the evidence beside them.

On reject: one-line verdict with the exact driver string, a row in ``outputs/launch_probe.csv``, and exit 2. The gate
runs before anything is downloaded, so a host that fails it holds no data. What to do with such a host is the caller's
decision and is provider-specific, so the gate only offers a hook: ``--on-reject-cmd '<shell command>'`` runs after the
verdict is recorded (for example ``brev delete <name>``) and its exit status is printed, never used as the gate result.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = Path(__file__).resolve().parent / "driver_allowlist.json"
CSV_PATH = REPO_ROOT / "outputs" / "launch_probe.csv"
CSV_COLUMNS = ("ts", "stage", "driver", "cuda", "gpu", "host", "datacenter", "image", "verdict", "tier",
               "cache", "note")
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


TIER_A = "A"
TIER_B = "B"
TIER_REJECT = "REJECT"


def classify_driver(driver: str, allowlist: dict) -> tuple[str, str]:
    """``(tier, reason)``: A accepted, B accepted only if the launch probe passes, REJECT terminate."""
    version = _version_tuple(driver)
    if len(version) < 2:
        return TIER_REJECT, "unparseable driver version"
    branch = str(version[0])
    for rule in allowlist.get("tier_a", []):
        if rule.get("exact_version") and driver.strip() == rule["exact_version"]:
            return TIER_A, rule.get("why", "exact match")
        if str(rule.get("branch")) == branch and "min_version" in rule:
            if version >= _version_tuple(rule["min_version"]):
                return TIER_A, rule.get("why", "minimum match")
            return TIER_REJECT, f"580 branch below {rule['min_version']}"
    for rule in allowlist.get("tier_b", []):
        if branch in [str(b) for b in rule.get("branches", [])]:
            return TIER_B, rule.get("why", "probe required")
    below = allowlist.get("reject_below")
    if below and version < _version_tuple(below):
        return TIER_REJECT, f"driver below {below}"
    return TIER_REJECT, f"branch {branch}.x has no evidence of a working run"


def parse_cuda_version(nvidia_smi_text: str) -> str:
    """The ``CUDA Version: X.Y`` shown in the nvidia-smi header, or ``unknown``."""
    found = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_text)
    return found.group(1) if found else "unknown"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
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
        datacenter=(os.environ.get("CERBERUS_HOST_LOCATION") or "unknown"),
        image=(image or os.environ.get("CERBERUS_IMAGE_TAG") or "unknown"),
    )


def append_probe_row(path: Path, info: HostInfo, *, stage: str, verdict: str, note: str = "",
                     tier: str = "", cache: str = "") -> None:
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
            "tier": tier,
            "cache": cache,
            "note": note,
        })


def run_reject_hook(command: str) -> tuple[bool, str]:
    """Run the caller's ``--on-reject-cmd`` through the shell; report success and a short detail."""
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"{type(e).__name__}: {e}"
    return proc.returncode == 0, f"exit {proc.returncode}"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--on-reject-cmd", default="",
                        help="shell command to run after a REJECT verdict (for example a provider's delete command)")
    args = parser.parse_args(argv)
    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    info = collect_host_info()
    tier, reason = classify_driver(info.driver, allowlist)
    verdict = "ALLOW" if tier == TIER_A else ("PROBE_REQUIRED" if tier == TIER_B else "REJECT")
    line = (f"DRIVER GATE: TIER {tier} ({verdict}) driver={info.driver} cuda={info.cuda} gpu={info.gpu} "
            f"location={info.datacenter} image={info.image} host={info.host} ({reason})")
    print(line)
    append_probe_row(CSV_PATH, info, stage="gate", verdict=verdict, note=f"tier {tier}; {reason}", tier=tier)
    if tier in (TIER_A, TIER_B):
        return 0

    if args.on_reject_cmd:
        ok, detail = run_reject_hook(args.on_reject_cmd)
        print(f"DRIVER GATE: on-reject command {'ok' if ok else 'failed'} ({detail})")
        append_probe_row(CSV_PATH, info, stage="gate", verdict="HOOK_OK" if ok else "HOOK_FAILED", note=detail,
                         tier=tier)
    else:
        print("DRIVER GATE: no --on-reject-cmd given; stop or release this host yourself")
    return EXIT_REJECT


if __name__ == "__main__":
    sys.exit(main())
