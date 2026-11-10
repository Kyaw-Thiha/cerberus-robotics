#!/usr/bin/env python3
"""Stop (not delete) a Runpod pod through the REST API -- the action the dead-man switch calls.

Kept separate from :mod:`create_pod` so the stop path has no creation defaults to get wrong. The API key comes from
``RUNPOD_API_KEY`` and is never printed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

STOP_URL = "https://rest.runpod.io/v1/pods/{pod_id}/stop"

Transport = Callable[[str, dict[str, str], bytes], tuple[int, str]]


def stop_url(pod_id: str) -> str:
    return STOP_URL.format(pod_id=pod_id)


def deadline_from_now(*, now: int, hours: float) -> int:
    """Epoch second at which the pod must be stopped."""
    return int(now + float(hours) * 3600)


def http_request(url: str, headers: dict[str, str], body: bytes) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}"


def stop_pod(pod_id: str, api_key: str, transport: Transport | None = None) -> tuple[int, dict]:
    status, text = (transport or http_request)(stop_url(pod_id), {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, b"")
    try:
        return status, (json.loads(text) if text else {})
    except ValueError:
        return status, {"raw": text[:2000]}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.pod_id:
        print("pod_stop: no pod id (--pod-id or RUNPOD_POD_ID)", flush=True)
        return 2
    if args.dry_run:
        print(f"pod_stop: would POST {stop_url(args.pod_id)}", flush=True)
        return 0
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("pod_stop: RUNPOD_API_KEY is not set", flush=True)
        return 2
    status, body = stop_pod(args.pod_id, api_key)
    print(f"pod_stop: {args.pod_id} -> HTTP {status} {json.dumps(body)[:300]}", flush=True)
    return 0 if status < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
