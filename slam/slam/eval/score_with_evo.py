"""Command-building logic for evo ATE and RPE scoring, kept separate from
actually invoking evo. evo, computing ATE/RPE per
candidate against /ground_truth/odom -- the same tool the KIT benchmark
paper itself used. `bag2 <bagfile> <ref_topic> <est_topic> [options]` syntax
confirmed real and current against evo's own repo/wiki (2026-09-20) -- not a
guess.
"""

from __future__ import annotations

import subprocess


def build_evo_ape_command(bag_path: str, backend_name: str) -> list[str]:
    return [
        "evo_ape", "bag2", bag_path,
        "/ground_truth/odom", "/slam/odom",
        "--align",
        "--save_results", f"slam/slam/eval/results/{backend_name}_ape.zip",
    ]


def build_evo_rpe_command(bag_path: str, backend_name: str) -> list[str]:
    return [
        "evo_rpe", "bag2", bag_path,
        "/ground_truth/odom", "/slam/odom",
        "--align",
        "--save_results", f"slam/slam/eval/results/{backend_name}_rpe.zip",
    ]


def run_evo_scoring(bag_path: str, backend_name: str) -> tuple[subprocess.CompletedProcess, subprocess.CompletedProcess]:
    ape_result = subprocess.run(build_evo_ape_command(bag_path, backend_name), check=True)
    rpe_result = subprocess.run(build_evo_rpe_command(bag_path, backend_name), check=True)
    return ape_result, rpe_result
