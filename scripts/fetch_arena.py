#!/usr/bin/env python3
"""Fetches Isaac Lab Arena source (gitignored under third_party/IsaacLab-Arena/).

Arena is Apache-2.0 and "alpha": we read it (and install it editable on the pod) but never vendor or patch
it. The default ref is ``main`` (Python >=3.12, Isaac Sim 6.2.0 container, Lab 3.0.0 GA; see
docs/arena_integration_log.md "Plan revision").

What is fetched
- A shallow (``--depth 1``) clone WITHOUT submodules by default. Git LFS smudging is disabled
  (``GIT_LFS_SKIP_SMUDGE=1``): Arena tracks ~170 doc images/gifs/mp4s in LFS and a plain clone pulls ~600 MB
  of them for nothing.
- ``--submodules`` additionally fetches ``submodules/IsaacLab`` (the Isaac Lab commit Arena pins), shallow,
  over https (``.gitmodules`` uses ``git@github.com:`` ssh URLs, which fail on a pod with no ssh key, so an
  ``insteadOf`` rewrite is passed per command; no user/global git config is touched). The
  ``submodules/Isaac-GR00T`` submodule is deliberately skipped (large, unrelated to Cerberus).

Recorded in ``.cerberus_fetch.json``: url, ref, ``commit`` (Arena HEAD SHA) and ``isaaclab_commit`` (the
Isaac Lab SHA Arena pins, read from the git tree with ``git ls-tree HEAD submodules/IsaacLab`` -- available
even when the submodule itself is not checked out).

``ref`` must be a branch or tag name (``git clone --branch``); to pin an exact commit, check out the SHA that
was recorded, e.g. ``git -C third_party/IsaacLab-Arena fetch --depth 1 origin <sha> && git ... checkout <sha>``.

Usage:
    python scripts/fetch_arena.py
    python scripts/fetch_arena.py --ref release/0.3.0 --force
    python scripts/fetch_arena.py --submodules
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ARENA = REPO_ROOT / "third_party" / "IsaacLab-Arena"

ARENA_URL = "https://github.com/isaac-sim/IsaacLab-Arena"
DEFAULT_REF = "main"
LICENSE_NOTE = "Apache-2.0"
METADATA_FILENAME = ".cerberus_fetch.json"
PACKAGE_DIR = "isaaclab_arena"
ISAACLAB_SUBMODULE = "submodules/IsaacLab"
# Submodules fetched by --submodules (GR00T is intentionally absent, see module docstring).
FETCHED_SUBMODULES = (ISAACLAB_SUBMODULE,)
# .gitmodules uses ssh URLs; rewrite to https for the one command only.
SSH_TO_HTTPS = ["-c", "url.https://github.com/.insteadOf=git@github.com:"]


def arena_path() -> Path:
    """Return the Arena checkout root, raising FileNotFoundError if it has not been fetched."""
    if not is_fetched():
        raise FileNotFoundError(
            f"Arena not found at {THIRD_PARTY_ARENA}; run `python scripts/fetch_arena.py` to fetch it"
        )
    return THIRD_PARTY_ARENA


def is_fetched() -> bool:
    return (THIRD_PARTY_ARENA / METADATA_FILENAME).is_file() and (
        THIRD_PARTY_ARENA / PACKAGE_DIR
    ).is_dir()


def read_metadata() -> dict:
    """Return the recorded fetch metadata ({} if none)."""
    path = THIRD_PARTY_ARENA / METADATA_FILENAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _run(cmd: list[str]) -> None:
    print(f"[fetch_arena] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=_git_env())


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return env


def _capture(cmd: list[str]) -> str:
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=_git_env()
    ).stdout.strip()


def parse_ls_tree_sha(output: str, path: str = ISAACLAB_SUBMODULE) -> str | None:
    """Extract the gitlink SHA from `git ls-tree HEAD <path>` output, or None if it is not a submodule entry.

    Line format: ``160000 commit <sha>\\t<path>``.
    """
    for line in output.splitlines():
        meta, _, entry_path = line.partition("\t")
        fields = meta.split()
        if (
            entry_path.strip() == path
            and len(fields) == 3
            and fields[0] == "160000"
            and fields[1] == "commit"
        ):
            return fields[2]
    return None


def _init_submodules(dest: Path) -> None:
    _run(
        [
            "git",
            *SSH_TO_HTTPS,
            "-C",
            str(dest),
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "--",
            *FETCHED_SUBMODULES,
        ]
    )


def fetch(
    ref: str = DEFAULT_REF, force: bool = False, submodules: bool = False
) -> bool:
    """Shallow clone of Arena at `ref`. Returns True if git ran, False if skipped.

    If Arena is already present (and not `force`), nothing is re-cloned; the only action is fetching the
    submodules when `submodules` is requested and they were not fetched before.
    """
    dest = THIRD_PARTY_ARENA
    if is_fetched() and not force:
        meta = read_metadata()
        if submodules and not meta.get("submodules"):
            print(f"[fetch_arena] present at {dest} without submodules; fetching them")
            _init_submodules(dest)
            meta["submodules"] = True
            (dest / METADATA_FILENAME).write_text(json.dumps(meta, indent=2) + "\n")
            return True
        print(
            f"[fetch_arena] already present at {dest}, skipping (use --force to re-fetch)"
        )
        return False
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    clone = ["git", "clone", "--depth", "1"]
    if ref:
        clone += ["--branch", ref]
    clone += [ARENA_URL, str(dest)]
    _run(clone)
    if submodules:
        _init_submodules(dest)

    sha = _capture(["git", "-C", str(dest), "rev-parse", "HEAD"])
    tree = _capture(["git", "-C", str(dest), "ls-tree", "HEAD", ISAACLAB_SUBMODULE])
    metadata = {
        "url": ARENA_URL,
        "ref": ref,
        "commit": sha,
        "isaaclab_commit": parse_ls_tree_sha(tree),
        "submodules": bool(submodules),
        "license": LICENSE_NOTE,
    }
    (dest / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ref",
        default=DEFAULT_REF,
        help="branch or tag of IsaacLab-Arena to fetch (default: main)",
    )
    parser.add_argument(
        "--submodules",
        action="store_true",
        help="also fetch submodules/IsaacLab (shallow, https)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete and re-fetch even if already present",
    )
    args = parser.parse_args()

    fetch(ref=args.ref, force=args.force, submodules=args.submodules)
    meta = read_metadata()
    print(
        f"[fetch_arena] commit={meta.get('commit')} isaaclab_commit={meta.get('isaaclab_commit')}"
    )
    print(THIRD_PARTY_ARENA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
