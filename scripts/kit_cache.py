#!/usr/bin/env python3
"""Kit startup-cache bundle: save after a good launch, restore before the next one on the same driver and GPU.

Cold Kit starts compile ray-tracing shaders and fill the caches below; that is the leading explanation for the
"300 s of silence" the last session saw. After the first OK/SLOW_OK launch on a host this tars the caches and uploads
them to ``cache/kit/<driver>_<gpu-model>/``; at the next pod start (after the driver gate) a matching bundle is
restored, which is what makes the probe's 5-minute warm wall cap meaningful.

Never restore across a different driver or GPU model: the bundle's manifest carries both, the remote key encodes
both, and the local state file records what was restored.

    python3 scripts/kit_cache.py discover
    python3 scripts/kit_cache.py save --driver 595.91.07 --gpu "NVIDIA GeForce RTX 4090"
    python3 scripts/kit_cache.py restore --driver 595.91.07 --gpu "NVIDIA GeForce RTX 4090"
    python3 scripts/kit_cache.py status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

#: Isaac Lab's docker docs cache list plus the Omniverse/Warp locations this image is known to use. Only existing,
#: non-empty directories are bundled; the discovered list is recorded so the next run can confirm it.
CACHE_CANDIDATES = (
    "~/.cache/ov",
    "~/.cache/nvidia/GLCache",
    "~/.cache/nvidia/ComputeCache",
    "~/.nv/GLCache",
    "~/.cache/warp",
    "~/.cache/kit",
    "~/.cache/omni.kit",
    "~/.nvidia-omniverse/cache",
    "~/.local/share/ov",
)

STATE_NAME = ".cerberus_kit_cache.json"
REMOTE_PREFIX = "cache/kit"


def _home(home: Path | None = None) -> Path:
    return Path(home) if home else Path.home()


def discover(home: Path | None = None) -> list[Path]:
    """Existing, non-empty cache directories in a stable order."""
    root = _home(home)
    found: list[Path] = []
    for candidate in CACHE_CANDIDATES:
        path = (root / candidate.replace("~/", "")) if home else Path(os.path.expanduser(candidate))
        if path.is_dir() and any(path.iterdir()):
            found.append(path)
    return found


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_").lower()


def cache_key(driver: str, gpu: str) -> str:
    return f"{slug(driver)}_{slug(gpu)}"


def state_path(home: Path | None = None) -> Path:
    return _home(home) / STATE_NAME


def cache_restored(driver: str | None = None, gpu: str | None = None, home: Path | None = None) -> bool:
    """True when the state file says a bundle was restored for this exact driver/GPU."""
    path = state_path(home)
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return False
    if driver and gpu:
        return data.get("driver") == driver and data.get("gpu") == gpu
    return bool(data.get("restored_at"))


def _archive(paths: list[Path], out: Path, home: Path) -> Path:
    """tar.gz of ``paths`` with arcnames relative to ``home`` (zstd is not in the stdlib; gzip is universal)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=str(path.relative_to(home)))
    return out


def _extract(archive: Path, home: Path) -> list[str]:
    with tarfile.open(archive, "r:*") as tar:
        names = tar.getnames()
        try:
            tar.extractall(home, filter="data")  # our own bundle; data filter avoids path escapes
        except TypeError:  # Python < 3.12 has no filter argument
            tar.extractall(home)  # noqa: S202
    return names


def _default_upload(local: Path, remote_dir: str) -> bool:
    try:
        from scripts.sync_to_r2 import _rclone_env, _require_env, REMOTE  # type: ignore[attr-defined]
    except ImportError:  # running as `python3 scripts/kit_cache.py` on the pod
        from sync_to_r2 import _rclone_env, _require_env, REMOTE  # type: ignore[no-redef]

    env = _rclone_env(_require_env())
    bucket = os.environ["R2_BUCKET"]
    cmd = ["rclone", "copyto", str(local), f"{REMOTE}:{bucket}/{remote_dir}/{local.name}"]
    return subprocess.run(cmd, env=env, capture_output=True, text=True).returncode == 0


def _default_download(remote_dir: str, local_dir: Path) -> bool:
    try:
        from scripts.sync_to_r2 import _rclone_env, _require_env, REMOTE  # type: ignore[attr-defined]
    except ImportError:  # running as `python3 scripts/kit_cache.py` on the pod
        from sync_to_r2 import _rclone_env, _require_env, REMOTE  # type: ignore[no-redef]

    env = _rclone_env(_require_env())
    bucket = os.environ["R2_BUCKET"]
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["rclone", "copy", f"{REMOTE}:{bucket}/{remote_dir}", str(local_dir)]
    return subprocess.run(cmd, env=env, capture_output=True, text=True).returncode == 0


def save(driver: str, gpu: str, *, home: Path | None = None, upload=None, out_dir: Path | None = None) -> dict:
    """Archive the discovered caches and upload them under the driver/GPU key. Returns a manifest."""
    home = _home(home)
    dirs = discover(home)
    manifest = {"driver": driver, "gpu": gpu, "key": cache_key(driver, gpu),
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dirs": [str(d.relative_to(home)) for d in dirs], "files": 0, "bytes": 0}
    if not dirs:
        return {**manifest, "saved": False, "reason": "no cache directories found"}
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="kit_cache_"))
    archive = _archive(dirs, work / "kit_cache.tar.gz", home)
    manifest["bytes"] = archive.stat().st_size
    upload = upload or _default_upload
    ok = upload(archive, f"{REMOTE_PREFIX}/{manifest['key']}")
    manifest["saved"] = bool(ok)
    return manifest


def restore(driver: str, gpu: str, *, home: Path | None = None, download=None, work_dir: Path | None = None) -> dict:
    """Fetch and extract the bundle for this exact driver/GPU; writes the state file on success."""
    home = _home(home)
    key = cache_key(driver, gpu)
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="kit_cache_restore_"))
    download = download or _default_download
    ok = download(f"{REMOTE_PREFIX}/{key}", work)
    if not ok:
        return {"restored": False, "reason": "no bundle for this driver/GPU", "key": key}
    archives = sorted(work.glob("kit_cache.tar.*"))
    if not archives:
        return {"restored": False, "reason": "bundle missing from the download", "key": key}
    names = _extract(archives[0], home)
    state = {"driver": driver, "gpu": gpu, "key": key, "dirs": sorted({n.split('/')[0] for n in names}),
             "restored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    state_path(home).write_text(json.dumps(state, indent=2) + "\n")
    return {"restored": True, **state}


def _host_driver_gpu() -> tuple[str, str]:
    def query(field: str) -> str:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15)
            return out.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            return "unknown"

    return query("driver_version"), query("name")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["discover", "save", "restore", "status"])
    parser.add_argument("--driver", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--home", default=None, help="override the home whose caches are read (tests)")
    args = parser.parse_args(argv)
    if args.action in ("discover", "status"):
        print(json.dumps({"dirs": [str(p) for p in discover(Path(args.home) if args.home else None)],
                          "restored": cache_restored(home=Path(args.home) if args.home else None)}, indent=2))
        return 0
    driver, gpu = args.driver, args.gpu
    if not driver or not gpu:
        host_driver, host_gpu = _host_driver_gpu()
        driver, gpu = driver or host_driver, gpu or host_gpu
    result = (save(driver, gpu) if args.action == "save"
              else restore(driver, gpu))
    print(json.dumps(result, indent=2))
    return 0 if result.get("saved", result.get("restored", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
