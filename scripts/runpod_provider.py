#!/usr/bin/env python3
"""RunPod REST helpers shared by the job-mode tooling (create, stop, read).

Job mode runs the image with no SSH: ``dockerEntrypoint`` is ``["/bin/bash", "-c"]`` and ``dockerStartCmd`` is a short
command that base64-decodes ``CERBERUS_BOOTSTRAP_B64`` and runs it. The long bootstrap therefore travels in the env,
never in the request body. ``ports`` is empty.

    from runpod_provider import create_job_pod, stop_pod, get_pod
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://rest.runpod.io/v1/pods"
USER_AGENT = "cerberus-pod-tools/1.0"
SHORT_BOOTSTRAP = 'echo "$CERBERUS_BOOTSTRAP_B64" | base64 -d | bash'


def _request(method: str, url: str, api_key: str, body: bytes | None = None) -> tuple[int, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "Accept": "application/json", "User-Agent": USER_AGENT}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}"


def create_job_pod(api_key: str, *, image: str, gpu_types: list[str], env: dict[str, str],
                   bootstrap_b64: str, name: str = "cerberus-job", disk_gb: int = 100,
                   cuda_first: str | None = "13.0", transport=None) -> tuple[int, dict]:
    """POST a job pod; retries filter-free once when the CUDA-filtered attempt fails."""
    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": list(gpu_types),
        "gpuCount": 1,
        "containerDiskInGb": int(disk_gb),
        "ports": [],
        "cloudType": "SECURE",
        "dockerEntrypoint": ["/bin/bash", "-c"],
        "dockerStartCmd": [SHORT_BOOTSTRAP],
        "env": {**env, "CERBERUS_BOOTSTRAP_B64": bootstrap_b64},
    }
    attempts = ([payload | {"allowedCudaVersions": [cuda_first]}] if cuda_first else []) + [payload]
    status, text = 0, ""
    for attempt in attempts:
        status, text = (transport or _request)("POST", API_URL, api_key, json.dumps(attempt).encode())
        if status < 400:
            break
    try:
        parsed = json.loads(text) if text else {}
    except ValueError:
        parsed = {"raw": text[:2000]}
    return status, parsed


def stop_pod(api_key: str, pod_id: str, transport=None) -> tuple[int, str]:
    return (transport or _request)("POST", f"{API_URL}/{pod_id}/stop", api_key, b"{}")


def get_pod(api_key: str, pod_id: str, transport=None) -> tuple[int, dict]:
    status, text = (transport or _request)("GET", f"{API_URL}/{pod_id}", api_key)
    try:
        return status, (json.loads(text) if text else {})
    except ValueError:
        return status, {"raw": text[:1000]}


def creations_file() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "pod_creations.json"


def creations() -> list[dict]:
    path = creations_file()
    return json.loads(path.read_text()) if path.is_file() else []


def record_creation(pod_id: str, image: str, cuda: list[str] | None) -> None:
    import time

    path = creations_file()
    rows = creations()
    rows.append({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pod_id": pod_id,
                 "image": image, "cuda": cuda or []})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1) + "\n")


def job_env(code_bundle: str, job_id: str, deadline_minutes: int, image: str) -> dict[str, str]:
    """The restricted pod env for a job pod: R2 access plus the job's identity. Nothing else."""
    env = {k: os.environ[k] for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                                      "R2_BUCKET") if k in os.environ}
    env.update({"CERBERUS_CODE_BUNDLE": code_bundle, "CERBERUS_JOB_ID": job_id,
                "CERBERUS_DEADLINE_MINUTES": str(deadline_minutes), "CERBERUS_IMAGE_TAG": image})
    if os.environ.get("RUNPOD_API_KEY"):
        env["RUNPOD_API_KEY"] = os.environ["RUNPOD_API_KEY"]
    return env
