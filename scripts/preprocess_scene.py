#!/usr/bin/env python3
"""Resolve (and therefore preprocess) one GRScenes scene into its metric USD wrapper, on the pod only.

Thin wrapper around the plain_usd backend: ``load_scene`` is what builds ``scene_metric.usd``. Kept as a script so
Kit retry and the shell drivers can call it without a heredoc (a heredoc cannot be replayed on a retry).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    from pod_guard import require_pod

    require_pod("preprocess_scene.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="grscenes:<folder>")
    args = parser.parse_args(argv)

    from common.cerberus_worlds.registry import get_backend

    info = get_backend("plain_usd").load_scene(args.scene)
    print(json.dumps({"scene": args.scene, "usd_path": info.get("usd_path"),
                      "variant": info.get("variant")}, indent=1))
    return 0 if info.get("usd_path") else 1


if __name__ == "__main__":
    raise SystemExit(main())
