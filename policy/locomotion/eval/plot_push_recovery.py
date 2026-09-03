"""Plots the push-recovery curve(s) from push_recovery_eval.py's output CSV(s).

Runs locally (pixi run python policy/locomotion/eval/plot_push_recovery.py ...)
-- no Isaac Sim needed, just matplotlib/pandas, per docs/isaac_lab_workflow.md's
local/remote split: pull the CSV down from the remote box, plot it here.

Accepts one CSV (the flat-checkpoint case, a single curve) or multiple
--csv label=path pairs plotted together on one x-axis. Each CSV's
`terrain_level` column (added when push_recovery_eval.py was run with
--terrain_levels) is "flat" when there's no terrain axis, else an int row
index -- when a CSV has more than one distinct level, one line per level is
drawn (labeled "<label> (level N)") instead of collapsing them together.

Each point gets a Wilson score confidence interval as error bars, computed
from the CSV's own `trials`/`successes` counts (real per-cell trial counts
from push_recovery_eval.py, not a heuristic) -- matters most at small
trials_per_cell, where a bare success_rate can hide how noisy the estimate
actually is (e.g. trials_per_cell=4 near p=0.5 has a ~25-point margin).
Wilson over a naive normal approximation because it stays inside [0, 1] and
holds up better at small n and p near 0 or 1.
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion. Returns
    (lower, upper) bounds in [0, 1]."""
    if trials == 0:
        return (0.0, 0.0)
    p_hat = successes / trials
    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _plot_curve(ax, df: pd.DataFrame, label: str) -> None:
    df = df.sort_values("magnitude_n")
    intervals = [wilson_interval(int(s), int(t)) for s, t in zip(df["successes"], df["trials"])]
    rate = df["success_rate"] * 100
    lower_err = [rate.iloc[i] - lo * 100 for i, (lo, _) in enumerate(intervals)]
    upper_err = [hi * 100 - rate.iloc[i] for i, (_, hi) in enumerate(intervals)]
    ax.errorbar(
        df["magnitude_n"],
        rate,
        yerr=[lower_err, upper_err],
        marker="o",
        capsize=3,
        label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot push-recovery success-rate curve(s).")
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="One or more label=path.csv pairs, e.g. --csv flat=results/push_recovery_2026-09-02.csv. "
        "Repeat for multiple curves (e.g. one per terrain difficulty level). Use the summary CSV "
        "(not the _trials.csv) -- it already has the trials/successes counts needed for error bars.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output image path (default: shows interactively).")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for entry in args.csv:
        label, path = entry.split("=", 1)
        df = pd.read_csv(path)
        levels = df["terrain_level"].unique() if "terrain_level" in df.columns else ["flat"]
        if len(levels) > 1:
            for lvl in sorted(levels, key=str):
                _plot_curve(ax, df[df["terrain_level"] == lvl], f"{label} (level {lvl})")
        else:
            _plot_curve(ax, df, label)

    ax.set_xlabel("Push magnitude (N)")
    ax.set_ylabel("Recovery rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Push-recovery success rate vs. push magnitude (95% Wilson CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"[INFO] Saved plot to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
