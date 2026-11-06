#!/usr/bin/env python3
"""Vendors SLAM backend source locally (gitignored under third_party/slam/)."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_SLAM = REPO_ROOT / "third_party" / "slam"

BACKENDS = {
    "fast_lio2": ("https://github.com/Kyaw-Thiha/cerberus-spark-fast-lio.git", None),
    "bievr_lio": ("https://github.com/Kyaw-Thiha/cerberus-BIEVR-LIO.git", None),
    "fast_livo2": ("https://github.com/Kyaw-Thiha/cerberus-FAST-LIVO2.git", "humble"),
    "orb_slam3": ("https://github.com/Kyaw-Thiha/cerberus-ORB_SLAM3.git", None),
    "multicam_slam": ("https://github.com/Kyaw-Thiha/cerberus-Multi_ORB_SLAM.git", None),
    # Replaces lvi_q (2026-09-21) -- see slam/slam/core/algorithms.py's
    # gs_livo registration comment for why.
    "gs_livo": ("https://github.com/Kyaw-Thiha/cerberus-GS-LIVO.git", None),
}

# Sub-vendored dependencies. Some backends pull in third-party C++ libraries
# (rpg_vikit is the only one today -- GS-LIVO's LIVMapper.cpp dereferences
# vk::AbstractCamera via rpg_vikit's headers, which upstream GS-LIVO does
# NOT vendor). The convention: "parent:child" -> the parent backend's
# source tree must already be present at third_party/slam/<parent> when
# the child gets fetched. The child lands at third_party/slam/<parent>/src/<child>
# (or wherever the parent's CMakeLists.txt's add_subdirectory call expects it).
SUB_VENDORS = {
    "gs_livo:rpg_vikit": ("https://github.com/Robotic-Developer-Road/rpg_vikit.git", None),
}


def ensure_colcon_ignore() -> None:
    """Vendored backends (fast_lio2, orb_slam3, etc.) ship their own
    CMakeLists.txt/package.xml -- without this marker, `colcon build` walks
    into third_party/ and tries to build them locally (confirmed: this
    actually happened, ORB_SLAM2/ORB_SLAM3 failed on a CMake version
    mismatch, other vendored packages aborted). These are built ONLY inside
    their own per-backend Docker images, never here.
    third_party/ is gitignored and gets wiped/re-fetched, so this marker is
    (re-)created here rather than relying on a one-off file surviving --
    idempotent, safe to call every run."""
    marker = THIRD_PARTY_SLAM.parent / "COLCON_IGNORE"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def fetch_one(name: str) -> None:
    url, ref = BACKENDS[name]
    dest = THIRD_PARTY_SLAM / name
    if dest.is_dir() and any(dest.iterdir()):
        print(f"[fetch_slam_backends] {name}: already present at {dest}, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, str(dest)]
    print(f"[fetch_slam_backends] {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def fetch_sub_vendor(key: str) -> None:
    """Fetch a sub-vendor (a dependency nested inside a backend's tree).
    The key format is "parent:child" -- parent is the backend under
    third_party/slam/, child is the directory under that backend's
    source tree (typically <parent>/src/<child>)."""
    parent, child = key.split(":", 1)
    url, ref = SUB_VENDORS[key]

    # Cache the clone at the top level so subsequent runs are fast (same
    # skip-if-present policy as the regular BACKENDS dict). The actual
    # copy into the parent is idempotent -- the destination might already
    # be populated from a previous run, in which case we skip the rsync.
    cache_dest = THIRD_PARTY_SLAM / f".__subvendor_cache__{parent}__{child}"
    dest = THIRD_PARTY_SLAM / parent / "src" / child

    if dest.is_dir() and any(dest.iterdir()):
        print(f"[fetch_slam_backends] {key}: already present at {dest}, skipping")
        return

    if not (cache_dest.is_dir() and any(cache_dest.iterdir())):
        cache_dest.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [url, str(cache_dest)]
        print(f"[fetch_slam_backends] {key}: {' '.join(cmd)} (cache)")
        subprocess.run(cmd, check=True)
    else:
        print(f"[fetch_slam_backends] {key}: cached at {cache_dest}, reusing")

    # Copy (not symlink -- the destination is gitignored anyway, and
    # a hard copy means the parent backend's tree is self-contained
    # once fetched, which is friendlier to Docker COPY semantics that
    # don't follow symlinks through bind mounts consistently).
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_slam_backends] {key}: copying {cache_dest} -> {dest}")
    shutil.copytree(cache_dest, dest, dirs_exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.names and not args.all:
        parser.print_usage(sys.stderr)
        print("error: pass one or more backend names, or --all", file=sys.stderr)
        return 1

    targets = list(BACKENDS) if args.all else args.names
    unknown = [name for name in targets if name not in BACKENDS]
    if unknown:
        print(
            f"error: unknown backend(s) {unknown!r}, valid choices: {sorted(BACKENDS)}",
            file=sys.stderr,
        )
        return 1

    ensure_colcon_ignore()
    for name in targets:
        fetch_one(name)
        # Sub-vendors that belong to this backend. Always run them after
        # the parent so a fresh parent fetch is followed by the dep fetch.
        for sub_key in SUB_VENDORS:
            sub_parent = sub_key.split(":", 1)[0]
            if sub_parent == name:
                fetch_sub_vendor(sub_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
