#!/usr/bin/env python3
"""Fetches the ARX5 arm URDF + meshes into gitignored third_party/.

- X5A (default, used for the fused Go2 + ARX5 robot): ARX's official model, github.com/ARXroboticsX/ARX_Model
  (branch master, folder X5/X5A) -> third_party/arx-model/. That repo has NO license file: the fetched files stay
  local; never copy them into tracked paths and never upload them to R2.
- X5, L5, X5_no_gripper, X5_umi: the arx5-sdk models (MIT) -> third_party/arx5-sdk/. Kept as a reference; the SDK
  X5.urdf has no gripper joints.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_ARX5 = REPO_ROOT / "third_party" / "arx5-sdk"
THIRD_PARTY_ARX_MODEL = REPO_ROOT / "third_party" / "arx-model"

ARX5_SDK_URL = "https://github.com/real-stanford/arx5-sdk"
DEFAULT_REF = "main"
LICENSE_NOTE = "MIT, Copyright (c) 2024 Yihuai Gao"
MODELS_DIR = "models"

ARX_MODEL_URL = "https://github.com/ARXroboticsX/ARX_Model"
ARX_MODEL_REF = "master"
ARX_MODEL_SUBDIR = "X5/X5A"
ARX_MODEL_LICENSE_NOTE = "NONE (no LICENSE file upstream): local use only, never commit or upload these files"

METADATA_FILENAME = ".cerberus_fetch.json"
DEFAULT_VARIANT = "X5A"
ARX_MODEL_VARIANTS = ("X5A",)
SDK_VARIANTS = ("X5", "L5", "X5_no_gripper", "X5_umi")
VARIANTS = (*ARX_MODEL_VARIANTS, *SDK_VARIANTS)


def variant_source(variant: str) -> str:
    """'arx-model' for the official ARX model variants, 'arx5-sdk' for the SDK ones."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown ARX5 variant {variant!r}, valid choices: {list(VARIANTS)}")
    return "arx-model" if variant in ARX_MODEL_VARIANTS else "arx5-sdk"


def arm_urdf_path(variant: str = DEFAULT_VARIANT) -> Path:
    """Path of the variant's URDF, raising FileNotFoundError if it has not been fetched.

    X5A -> <third_party/arx-model>/X5/X5A/urdf/X5A.urdf; SDK variants -> <third_party/arx5-sdk>/models/<variant>.urdf.
    """
    if variant_source(variant) == "arx-model":
        path = THIRD_PARTY_ARX_MODEL / ARX_MODEL_SUBDIR / "urdf" / f"{variant}.urdf"
    else:
        path = THIRD_PARTY_ARX5 / MODELS_DIR / f"{variant}.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"ARX5 URDF not found at {path}; run `python scripts/fetch_arx5_assets.py` to fetch it")
    return path


def is_fetched(source: str = "arx5-sdk") -> bool:
    if source == "arx-model":
        return (THIRD_PARTY_ARX_MODEL / METADATA_FILENAME).is_file() and (
            THIRD_PARTY_ARX_MODEL / ARX_MODEL_SUBDIR
        ).is_dir()
    return (THIRD_PARTY_ARX5 / METADATA_FILENAME).is_file() and (THIRD_PARTY_ARX5 / MODELS_DIR).is_dir()


def _run(cmd: list[str]) -> None:
    print(f"[fetch_arx5_assets] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _sparse_fetch(dest: Path, url: str, ref: str, subdir: str, license_note: str, fetched: bool, force: bool) -> bool:
    if fetched and not force:
        print(f"[fetch_arx5_assets] already present at {dest}, skipping (use --force to re-fetch)")
        return False
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    clone = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse"]
    if ref:
        clone += ["--branch", ref]
    clone += [url, str(dest)]
    _run(clone)
    _run(["git", "-C", str(dest), "sparse-checkout", "set", subdir])

    rev_cmd = ["git", "-C", str(dest), "rev-parse", "HEAD"]
    sha = subprocess.run(rev_cmd, check=True, capture_output=True, text=True).stdout.strip()
    metadata = {"url": url, "ref": ref, "commit": sha, "license": license_note}
    (dest / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return True


def fetch(ref: str = DEFAULT_REF, force: bool = False) -> bool:
    """arx5-sdk: sparse, shallow, blob-filtered clone of only models/. Returns True if git ran, False if skipped."""
    return _sparse_fetch(
        THIRD_PARTY_ARX5, ARX5_SDK_URL, ref, MODELS_DIR, LICENSE_NOTE, is_fetched("arx5-sdk"), force
    )


def fetch_arx_model(ref: str = ARX_MODEL_REF, force: bool = False) -> bool:
    """ARX_Model: sparse, shallow, blob-filtered clone of only X5/X5A (records the commit SHA in the metadata)."""
    return _sparse_fetch(
        THIRD_PARTY_ARX_MODEL,
        ARX_MODEL_URL,
        ref,
        ARX_MODEL_SUBDIR,
        ARX_MODEL_LICENSE_NOTE,
        is_fetched("arx-model"),
        force,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default=None, help="branch or tag to fetch (default: master for X5A, main for SDK)")
    parser.add_argument(
        "--variant", default=DEFAULT_VARIANT, help=f"URDF variant to fetch and print, one of {list(VARIANTS)}"
    )
    parser.add_argument("--all", action="store_true", help="fetch both the ARX_Model (X5A) and the arx5-sdk models")
    parser.add_argument("--force", action="store_true", help="delete and re-fetch even if already present")
    args = parser.parse_args()

    if args.variant not in VARIANTS:
        print(f"error: unknown variant {args.variant!r}, valid choices: {list(VARIANTS)}", file=sys.stderr)
        return 1

    sources = {variant_source(args.variant)}
    if args.all:
        sources = {"arx-model", "arx5-sdk"}
    if "arx-model" in sources:
        fetch_arx_model(ref=args.ref or ARX_MODEL_REF, force=args.force)
    if "arx5-sdk" in sources:
        fetch(ref=args.ref or DEFAULT_REF, force=args.force)
    print(arm_urdf_path(args.variant))
    return 0


if __name__ == "__main__":
    sys.exit(main())
