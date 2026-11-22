#!/usr/bin/env python3
"""Download public candidate checkpoints -- no credentials needed, unlike
sync_to_r2.py (upload-only, maintainer-only, needs private R2 keys). This is
the download-only counterpart for anyone trying the project out: plain HTTPS
GETs against the public `cerberus-public` bucket.

Bucket layout (mirrors policy/<module>/checkpoints/ locally):
    <PUBLIC_BASE_URL>/manifest.json      -- {"locomotion": [...], "manipulation": [...]}
    <PUBLIC_BASE_URL>/<module>/<filename>

R2 doesn't expose directory listing to public/anonymous requests, so
manifest.json is the one file every module's file list is read from --
update it by hand alongside any future upload to this bucket.

Usage:
    python3 scripts/fetch_checkpoint.py --list                        # every module + file
    python3 scripts/fetch_checkpoint.py locomotion                    # all locomotion checkpoints
    python3 scripts/fetch_checkpoint.py locomotion go2_locomotion_v1_candidate.onnx  # just one
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

PUBLIC_BASE_URL = "https://cerberus-public.kthiha.com"

# Cloudflare's bot-protection rejects Python's default urllib User-Agent
# outright (HTTP 403, "error code: 1010") even though the bucket/domain
# itself is public and working -- confirmed by re-running the same request
# with a browser-style User-Agent, which succeeds. Every request needs this
# header, not just a one-off workaround.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; cerberus-fetch-checkpoint/1.0)"}

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fetch_manifest() -> dict[str, list[str]]:
    url = f"{PUBLIC_BASE_URL}/manifest.json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS)) as resp:
            return json.load(resp)
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Couldn't reach {url} ({e}). If you're seeing this from the Cerberus repo itself, "
            "the checkpoint bucket may not have public access enabled yet -- ask a maintainer."
        ) from None


def _download(module: str, filename: str) -> None:
    dest_dir = REPO_ROOT / "policy" / module / "checkpoints"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    url = f"{PUBLIC_BASE_URL}/{module}/{filename}"
    print(f"[INFO] Downloading {url} -> {dest_path}")
    with urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS)) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public Cerberus candidate checkpoints.")
    parser.add_argument("module", nargs="?", help="e.g. 'locomotion'. Omit with --list to see all modules.")
    parser.add_argument("filename", nargs="?", default=None, help="Specific file. Omit to fetch every file for the module.")
    parser.add_argument("--list", action="store_true", help="Print available modules/files, don't download anything.")
    args = parser.parse_args()

    manifest = _fetch_manifest()

    if args.list:
        for module, files in manifest.items():
            print(f"{module}:")
            for f in files:
                print(f"  {f}")
        return

    if args.module is None:
        parser.error("module is required unless --list is given")
    if args.module not in manifest:
        parser.error(f"unknown module {args.module!r}, available: {sorted(manifest)}")

    if args.filename is not None:
        _download(args.module, args.filename)
    else:
        for filename in manifest[args.module]:
            _download(args.module, filename)


if __name__ == "__main__":
    main()
