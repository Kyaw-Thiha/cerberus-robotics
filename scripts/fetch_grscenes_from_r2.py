#!/usr/bin/env python3
"""Pull the private GRScenes home mirror from R2 into the repo's data dir.

Counterpart of `fetch_grscenes_to_r2.py`. Layout under --dest (default
data/scenes/grscenes/home, gitignored data -- never commit, CC BY-NC-SA 4.0):

    <dest>/Materials/          mdl + textures
    <dest>/models/             layout + object usd
    <dest>/scenes/<id>_usd/    start_result_*.usd, metadata.json
    <dest>/manifest.json
    <dest>/env.sh              export MDL_SYSTEM_PATH=<abs dest>/Materials

R2 cannot store symlinks, but the scene USDs reference assets relative to their
own folder through the `Materials` / `models` symlinks of the original release
(`scenes/<id>_usd/Materials -> ../../Materials`, same for models). So after the
copy this script recreates those relative links in every scene folder that lacks
them. Source it before launching Isaac: `source <dest>/env.sh`.

Needs rclone and R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.

Usage:
    python3 scripts/fetch_grscenes_from_r2.py
    python3 scripts/fetch_grscenes_from_r2.py --scene-id <id> --dest /tmp/grscenes
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

try:
    from sync_to_r2 import REMOTE, _rclone_env, _require_env
except ImportError:  # duplicated from scripts/sync_to_r2.py so this runs standalone
    REMOTE = "cerberus_r2"

    def _require_env() -> dict[str, str]:
        names = [
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
        ]
        missing = [v for v in names if not os.environ.get(v)]
        if missing:
            raise SystemExit(f"Missing required env vars: {', '.join(missing)}.")
        return {v: os.environ[v] for v in names}

    def _rclone_env(env: dict[str, str]) -> dict[str, str]:
        rclone_env = os.environ.copy()
        prefix = f"RCLONE_CONFIG_{REMOTE.upper()}_"
        rclone_env[f"{prefix}TYPE"] = "s3"
        rclone_env[f"{prefix}PROVIDER"] = "Cloudflare"
        rclone_env[f"{prefix}ACCESS_KEY_ID"] = env["R2_ACCESS_KEY_ID"]
        rclone_env[f"{prefix}SECRET_ACCESS_KEY"] = env["R2_SECRET_ACCESS_KEY"]
        rclone_env[f"{prefix}ENDPOINT"] = (
            f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
        )
        rclone_env[f"{prefix}ACL"] = "private"
        rclone_env[f"{prefix}NO_CHECK_BUCKET"] = "true"
        return rclone_env


DEFAULT_PREFIX = "worlds/grscenes/home"
DEFAULT_DEST = "data/scenes/grscenes/home"
LINKS = {"Materials": "../../Materials", "models": "../../models"}


def normalize_scene_id(sid: str) -> str:
    sid = sid.strip().strip("/")
    return sid if sid.endswith("_usd") else sid + "_usd"


def rclone_args(remote: str, dest: Path, scene_id: str | None) -> list[str]:
    cmd = [
        "rclone",
        "copy",
        remote,
        str(dest),
        "--transfers",
        "8",
        "--checksum",
        "--stats",
        "30s",
        "-v",
    ]
    if scene_id:
        sid = normalize_scene_id(scene_id)
        for inc in (
            "/Materials/**",
            "/models/**",
            f"/scenes/{sid}/**",
            "/manifest.json",
        ):
            cmd += ["--include", inc]
    return cmd


def find_scene_ids(dest: Path) -> list[str]:
    scenes = dest / "scenes"
    if not scenes.is_dir():
        return []
    return sorted(
        p.name for p in scenes.iterdir() if p.is_dir() and p.name.endswith("_usd")
    )


def ensure_symlinks(dest: Path) -> int:
    """Recreate scene-level and model-level links omitted by the R2 file mirror."""
    made = 0
    for sid in find_scene_ids(dest):
        for name, target in LINKS.items():
            link = dest / "scenes" / sid / name
            if os.path.lexists(link):
                continue
            link.symlink_to(target)
            made += 1
    models = dest / "models"
    if models.is_dir():
        for instance in models.rglob("instance.usd"):
            link = instance.parent / "Materials"
            if os.path.lexists(link):
                continue
            target = os.path.relpath(dest / "Materials", link.parent)
            link.symlink_to(target)
            made += 1
    return made


def write_env_sh(dest: Path) -> str:
    line = f"export MDL_SYSTEM_PATH={(dest.resolve() / 'Materials')}"
    (dest / "env.sh").write_text(line + "\n")
    return line


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--prefix", default=DEFAULT_PREFIX, help="R2 prefix (default %(default)s)"
    )
    p.add_argument(
        "--dest", default=DEFAULT_DEST, help="local destination (default %(default)s)"
    )
    p.add_argument(
        "--scene-id",
        default=None,
        help="only fetch this scene (plus Materials/models/manifest)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    env = _require_env()
    rclone_env = _rclone_env(env)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE}:{env['R2_BUCKET']}/{args.prefix.strip('/')}/"
    cmd = rclone_args(remote, dest, args.scene_id)
    print(f"[INFO] {' '.join(cmd)}")
    subprocess.run(cmd, env=rclone_env, check=True)
    made = ensure_symlinks(dest)
    line = write_env_sh(dest)
    ids = find_scene_ids(dest)
    print(f"[INFO] created {made} relative symlink(s)")
    print(f"[INFO] scene ids ({len(ids)}): {', '.join(ids) if ids else '(none)'}")
    print(f"[INFO] wrote {dest / 'env.sh'}; run `source {dest / 'env.sh'}` or:")
    print(line)


if __name__ == "__main__":
    main()
