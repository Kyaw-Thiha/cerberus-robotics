#!/usr/bin/env python3
"""Download the OmniGibson/BEHAVIOR-1K dataset and prune to Cerberus's chosen scenes.

Run as a single `docker build` RUN step (see docker/Dockerfile.remote) so the
prune happens in the same image layer as the download — a later layer's
deletion wouldn't shrink the image. See docs/omnigibson_handoff.md sections 2
and 4 for the download API and the chosen scene list.
"""

import shutil
import sys
import time
from pathlib import Path

# Household (primary mission) + grocery/restaurant/office (secondary
# missions) scenes actually selected for Cerberus, per
# docs/omnigibson_handoff.md section 4. Everything else the dataset ships
# gets pruned post-download to keep the image smaller.
ALLOWED_SCENES = {
    "Beechwood_0_int", "Beechwood_1_int", "Beechwood_0_garden",
    "Benevolence_0_int", "Benevolence_1_int", "Benevolence_2_int",
    "Ihlen_0_int", "Ihlen_1_int",
    "Merom_0_int", "Merom_1_int", "Merom_0_garden",
    "Pomaria_0_int", "Pomaria_1_int", "Pomaria_2_int", "Pomaria_0_garden",
    "Rs_int", "Rs_garden",
    "Wainscott_0_int", "Wainscott_1_int", "Wainscott_0_garden",
    "house_single_floor", "house_double_floor_lower", "house_double_floor_upper",
    "grocery_store_cafe",
    "restaurant_brunch",
    "office_large",
}

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 30


def timed(label, fn):
    start = time.time()
    print(f"[fetch_omnigibson_assets] starting: {label}", flush=True)
    result = fn()
    print(f"[fetch_omnigibson_assets] done: {label} ({time.time() - start:.1f}s)", flush=True)
    return result


def with_retries(label, fn):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return timed(f"{label} (attempt {attempt}/{MAX_RETRIES})", fn)
        except Exception as exc:  # noqa: BLE001 - retry anything transient (network flake etc.)
            last_exc = exc
            print(f"[fetch_omnigibson_assets] {label} failed on attempt {attempt}: {exc}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC)
    print(f"[fetch_omnigibson_assets] {label} failed after {MAX_RETRIES} attempts", flush=True)
    raise last_exc


def download():
    from omnigibson.utils.asset_utils import download_assets, download_behavior_1k_assets

    with_retries("download_assets", download_assets)
    # accept_license=True: see plans/omnigibson_handoff.md section 9 for the
    # considered license decision behind this, not an oversight.
    with_retries("download_behavior_1k_assets", lambda: download_behavior_1k_assets(accept_license=True))


def prune_scenes():
    from omnigibson.macros import gm

    dataset_path = Path(gm.DATASET_PATH)
    scenes_dir = dataset_path / "scenes"
    print(f"[fetch_omnigibson_assets] dataset path: {dataset_path}", flush=True)

    if not scenes_dir.is_dir():
        # Open verification item per docs/omnigibson_handoff.md section 10 -
        # log what's actually there instead of guessing/silently no-op'ing.
        print(
            f"[fetch_omnigibson_assets] WARNING: expected scenes dir not found at {scenes_dir} "
            "- dataset layout differs from assumption, skipping prune",
            flush=True,
        )
        print(f"[fetch_omnigibson_assets] dataset_path contents: {list(dataset_path.iterdir())}", flush=True)
        return

    found = sorted(p.name for p in scenes_dir.iterdir() if p.is_dir())
    print(f"[fetch_omnigibson_assets] found {len(found)} scene dirs: {found}", flush=True)

    missing = ALLOWED_SCENES - set(found)
    if missing:
        print(f"[fetch_omnigibson_assets] WARNING: expected scenes not present: {sorted(missing)}", flush=True)

    removed = []
    for scene_dir in scenes_dir.iterdir():
        if scene_dir.is_dir() and scene_dir.name not in ALLOWED_SCENES:
            shutil.rmtree(scene_dir)
            removed.append(scene_dir.name)

    print(f"[fetch_omnigibson_assets] pruned {len(removed)} unused scene dirs: {sorted(removed)}", flush=True)


def main():
    start = time.time()
    download()
    prune_scenes()
    print(f"[fetch_omnigibson_assets] total time: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
