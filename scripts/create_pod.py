#!/usr/bin/env python3
"""Create a Cerberus pod through the Runpod REST API, steering the host driver branch.

The pod's host driver is the thing that decides whether Isaac Sim 6.x starts at all (see
``scripts/pod_driver_allowlist.json``). Runpod selects the driver branch from the CUDA list it is given, so this
script sends ``allowedCudaVersions: ["13.0"]`` once; if the provider has no such host it retries filter-free and the
driver gate enforces the tier rule before anything runs. The API has no ``13.2`` value (13.0, 12.9 ... 11.8 only).

The API key is read from ``RUNPOD_API_KEY`` and never printed. The script prints the pod id and stops there: nothing
is terminated, and the dead-man switch (``scripts/pod_deadman.sh``) is what stops the pod later.

    RUNPOD_API_KEY=... python3 scripts/create_pod.py --dry-run          # show the request body
    RUNPOD_API_KEY=... python3 scripts/create_pod.py                     # create it
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

API_URL = "https://rest.runpod.io/v1/pods"
USER_AGENT = "cerberus-pod-tools/1.0"
DEFAULT_IMAGE = "kevinkyawthiha/cerberus-remote-v6:v6-ga"
DEFAULT_GPU_TYPES = ("NVIDIA A40", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000",
                     "NVIDIA L40S", "NVIDIA RTX A6000")
CUDA_FIRST = "13.0"
#: The API's own enum. There is no "13.2": after one filtered attempt this script creates filter-free and lets the
#: driver gate enforce the tier rule, which is the only legal way to express "fall back to the other branch".
VALID_CUDA_VERSIONS = ("13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8")
MAX_CREATIONS_PER_SESSION = 4
CREATIONS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "pod_creations.json"
DEFAULT_CONTAINER_DISK_GB = 100

Transport = Callable[[str, dict[str, str], bytes], tuple[int, str]]
"""``(url, headers, body) -> (status, text)``; injectable so the request can be tested without network."""


@dataclass(frozen=True)
class PodSpec:
    name: str = "cerberus-pod"
    image: str = DEFAULT_IMAGE
    gpu_types: tuple[str, ...] = DEFAULT_GPU_TYPES
    cuda_versions: tuple[str, ...] = (CUDA_FIRST,)
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB
    volume_gb: int = 0
    volume_mount: str = "/workspace"
    ports: tuple[str, ...] = ("22/tcp",)
    cloud_type: str = "SECURE"
    gpu_count: int = 1
    extra_env: dict[str, str] = field(default_factory=dict)

    def env(self) -> dict[str, str]:
        env = {"CERBERUS_ON_POD": "1", "CERBERUS_IMAGE_TAG": self.image}
        # the pod needs the key for the gate's terminate, the probe's terminate and the dead-man switch
        if os.environ.get("RUNPOD_API_KEY"):
            env["RUNPOD_API_KEY"] = os.environ["RUNPOD_API_KEY"]
        env.update(self.extra_env)
        return env


def build_payload(spec: PodSpec) -> dict:
    """The JSON body for ``POST /pods``. Ordering matters: gpuTypes and CUDA versions are priority lists."""
    payload = {
        "name": spec.name,
        "imageName": spec.image,
        "gpuTypeIds": list(spec.gpu_types),
        "gpuCount": int(spec.gpu_count),
        **({"allowedCudaVersions": list(spec.cuda_versions)} if spec.cuda_versions else {}),
        "containerDiskInGb": int(spec.container_disk_gb),
        "volumeInGb": int(spec.volume_gb),
        "volumeMountPath": spec.volume_mount,
        "ports": list(spec.ports),
        "cloudType": spec.cloud_type,
        "env": spec.env(),
    }
    if spec.volume_gb == 0 and "volumeMountPath" in payload:
        payload.pop("volumeMountPath")
    return payload


def http_post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}"


def create_pod(spec: PodSpec, api_key: str, transport: Transport | None = None) -> tuple[int, dict]:
    """POST the pod; returns ``(status, parsed_body)``."""
    payload = build_payload(spec)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "User-Agent": USER_AGENT}
    status, text = (transport or http_post)(API_URL, headers, json.dumps(payload).encode("utf-8"))
    try:
        parsed = json.loads(text) if text else {}
    except ValueError:
        parsed = {"raw": text[:2000]}
    return status, parsed


def validate_cuda_versions(versions) -> None:
    for version in versions or ():
        if version not in VALID_CUDA_VERSIONS:
            raise SystemExit(
                f"allowedCudaVersions {version!r} is not accepted by the API; valid values: "
                + ", ".join(VALID_CUDA_VERSIONS) + " (omit the filter and let the driver gate decide instead)"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="cerberus-pod")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--gpu-type", action="append", default=None,
                        help="ordered priority list; repeatable (default: A40, then RTX 4090)")
    parser.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    parser.add_argument("--volume-gb", type=int, default=0)
    parser.add_argument("--volume-mount", default="/workspace")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="extra env var for the pod; CERBERUS_ON_POD=1 is always set")
    parser.add_argument("--creations-file", default=str(CREATIONS_PATH),
                        help="where the per-session creation count is kept (cap 4)")
    parser.add_argument("--dry-run", action="store_true", help="print the request body and exit; needs no API key")
    return parser


def spec_from_args(args) -> PodSpec:
    extra_env = {}
    for item in args.env:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--env needs KEY=VALUE, got {item!r}")
        extra_env[key] = value
    return PodSpec(
        name=args.name, image=args.image,
        gpu_types=tuple(args.gpu_type) if args.gpu_type else DEFAULT_GPU_TYPES,
        container_disk_gb=args.container_disk_gb, volume_gb=args.volume_gb,
        volume_mount=args.volume_mount, extra_env=extra_env,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = spec_from_args(args)

    if args.dry_run:
        print(json.dumps(build_payload(spec), indent=2))
        return 0

    creations_file = Path(args.creations_file)
    creations = json.loads(creations_file.read_text()) if creations_file.is_file() else []
    if len(creations) >= MAX_CREATIONS_PER_SESSION:
        print(f"create_pod: refusing to create: {len(creations)} pods already created this session "
              f"(cap {MAX_CREATIONS_PER_SESSION})", flush=True)
        return 3

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("create_pod: RUNPOD_API_KEY is not set; export it and retry", flush=True)
        return 2

    validate_cuda_versions(spec.cuda_versions)
    status, body = create_pod(spec, api_key)
    if status >= 400 and spec.cuda_versions == (CUDA_FIRST,):
        print(f"create_pod: CUDA {CUDA_FIRST} attempt failed (HTTP {status}); creating filter-free "
              "(the gate enforces the tier rule)", flush=True)
        spec = PodSpec(**{**spec.__dict__, "cuda_versions": ()})
        status, body = create_pod(spec, api_key)

    if status >= 400:
        print(f"create_pod: failed HTTP {status}: {json.dumps(body)[:600]}", flush=True)
        return 1
    pod_id = body.get("id") or body.get("podId") or "unknown"
    creations.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pod_id": pod_id,
                      "cuda": list(spec.cuda_versions), "gpu_types": list(spec.gpu_types)})
    creations_file.parent.mkdir(parents=True, exist_ok=True)
    creations_file.write_text(json.dumps(creations, indent=1) + "\n")
    print(f"create_pod: created pod {pod_id} cuda={list(spec.cuda_versions) or 'filter-free'} "
          f"image={spec.image} ({len(creations)}/{MAX_CREATIONS_PER_SESSION} this session)", flush=True)
    print("create_pod: nothing was terminated; the dead-man switch stops it after CERBERUS_POD_MAX_HOURS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
