"""Scene payload sizes, measured on the R2 side before anything is downloaded, plus a geometry-only fetch.

Why: the first session's fetch of one GRScenes home filled a 120 GB container disk, and the existing fetch
script (``scripts/fetch_grscenes_from_r2.py``) takes ``/Materials/**`` and ``/models/**`` unconditionally --
shared assets for every mirrored home. This module lists what is actually there per component and per file
class (geometry vs textures vs material definitions), then can copy **only the USD geometry** so a
collision-only scene can be parsed without the multi-gigabyte texture set.

    python -m spikes.newton.scene_assets --list --out <dir>
    python -m spikes.newton.scene_assets --fetch-lite --scene-id MV7J... --dest <dir> --out <dir>

Nothing here writes outside ``spikes/newton``'s own artifacts and the given ``--dest``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

GEOMETRY = {".usd", ".usda", ".usdc", ".usdz"}
TEXTURE = {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga", ".tif", ".tiff", ".bmp"}
MATERIAL = {".mdl"}
PREFIX = "worlds/grscenes/home"


def rclone_env() -> None:
    for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines() \
            if (Path(__file__).resolve().parents[2] / ".env").is_file() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_TYPE", "s3")
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_PROVIDER", "Cloudflare")
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID", os.environ.get("R2_ACCESS_KEY_ID", ""))
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY", os.environ.get("R2_SECRET_ACCESS_KEY", ""))
    os.environ.setdefault("RCLONE_CONFIG_CERBERUS_R2_ENDPOINT",
                          f"https://{os.environ.get('R2_ACCOUNT_ID', '')}.r2.cloudflarestorage.com")


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _component(path: str) -> str:
    parts = path.split("/")
    if parts[0] == "scenes" and len(parts) > 1:
        return f"scenes/{parts[1]}"
    return parts[0]


def _classify(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in GEOMETRY:
        return "geometry"
    if ext in TEXTURE:
        return "texture"
    if ext in MATERIAL:
        return "material"
    return "other"


def list_sizes(prefix: str = PREFIX) -> dict:
    bucket = os.environ["R2_BUCKET"]
    result = subprocess.run(["rclone", "lsf", "--files-only", "--format", "sp", "-R",
                             f"cerberus_r2:{bucket}/{prefix}"], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"rclone lsf failed: {result.stderr[-400:]}")
    per_component: dict[str, dict[str, int]] = {}
    per_class: dict[str, int] = {}
    total = 0
    files = 0
    for line in result.stdout.splitlines():
        size_text, _, path = line.partition(";")
        if not path:
            continue
        try:
            size = int(size_text)
        except ValueError:
            continue
        component = _component(path)
        kind = _classify(path)
        slot = per_component.setdefault(component, {})
        slot[kind] = slot.get(kind, 0) + size
        slot["total"] = slot.get("total", 0) + size
        per_class[kind] = per_class.get(kind, 0) + size
        total += size
        files += 1
    return {"prefix": prefix, "files": files, "total_bytes": total,
            "per_component": per_component, "per_class": per_class}


def print_sizes(report: dict) -> None:
    print(f"[disk] {report['prefix']}: {report['files']} objects, {_human(report['total_bytes'])} total", flush=True)
    print(f"[disk] {'component':<40} {'geometry':>12} {'texture':>12} {'material':>10} {'other':>10} {'total':>12}",
          flush=True)
    for component, sizes in sorted(report["per_component"].items(), key=lambda kv: -kv[1]["total"])[:20]:
        print(f"[disk] {component:<40} {_human(sizes.get('geometry', 0)):>12} {_human(sizes.get('texture', 0)):>12} "
              f"{_human(sizes.get('material', 0)):>10} {_human(sizes.get('other', 0)):>10} "
              f"{_human(sizes['total']):>12}", flush=True)
    print("[disk] by class: " + ", ".join(f"{k}={_human(v)}" for k, v in sorted(report["per_class"].items())),
          flush=True)


def fetch_lite(scene_id: str, dest: Path) -> dict:
    """Copy only the USD geometry (and JSON metadata) of one scene and its shared model library."""
    bucket = os.environ["R2_BUCKET"]
    dest.mkdir(parents=True, exist_ok=True)
    sid = scene_id if scene_id.endswith("_usd") else f"{scene_id}_usd"
    includes = ["/scenes/" + sid + "/**", "/models/**", "/manifest.json"]
    cmd = ["rclone", "copy", f"cerberus_r2:{bucket}/{PREFIX}", str(dest), "--transfers", "16",
           "--stats", "30s", "-v", "--exclude", "*.mdl"]
    for inc in includes:
        cmd += ["--include", inc]
    for ext in (".usd", ".usda", ".usdc", ".usdz", ".json"):
        cmd += ["--include", f"*{ext}"]
    print(f"[fetch-lite] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[fetch-lite] rc={result.returncode}", flush=True)
    if result.returncode != 0:
        print(result.stderr[-1500:], flush=True)
    return {"rc": result.returncode, "dest": str(dest), "du": du(dest)}


def du(root: Path) -> dict:
    """Bytes per top-level component and per class in a local tree (the same axes as the R2 listing)."""
    per_component: dict[str, dict[str, int]] = {}
    per_class: dict[str, int] = {}
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        component = _component(str(rel))
        kind = _classify(str(rel))
        size = path.stat().st_size
        slot = per_component.setdefault(component, {})
        slot[kind] = slot.get(kind, 0) + size
        slot["total"] = slot.get("total", 0) + size
        per_class[kind] = per_class.get(kind, 0) + size
        total += size
    return {"root": str(root), "total_bytes": total, "per_component": per_component, "per_class": per_class}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--list", action="store_true", help="list R2 sizes only; downloads nothing")
    parser.add_argument("--fetch-lite", action="store_true", help="copy USD geometry only")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--dest", default="/workspace/scene_lite")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rclone_env()

    payload: dict = {}
    if args.list or not args.fetch_lite:
        payload["r2"] = list_sizes(args.prefix)
        print_sizes(payload["r2"])
    if args.fetch_lite:
        if not args.scene_id:
            raise SystemExit("--fetch-lite needs --scene-id")
        payload["fetch_lite"] = fetch_lite(args.scene_id, Path(args.dest))
        print(f"[fetch-lite] local du: {_human(payload['fetch_lite']['du']['total_bytes'])}", flush=True)
        for component, sizes in sorted(payload["fetch_lite"]["du"]["per_component"].items(),
                                       key=lambda kv: -kv[1]["total"])[:15]:
            print(f"[fetch-lite] {component:<40} {_human(sizes['total']):>12}", flush=True)
    (out / "scene_assets.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[out] wrote {out / 'scene_assets.json'}", flush=True)
    return 0


if __name__ == "__main__":
    from spikes.newton.common import run_stage

    raise SystemExit(run_stage(main))
