#!/usr/bin/env python3
"""The results card: success rate, falls and path efficiency for PPO vs JEPLO vs the kinematic bound.

Input JSON: {"rows": [{"policy": "PPO", "success_rate": 0.8, "falls": 1, "path_efficiency": 0.7,
                       "n": 10, "seeds": "0-9"}, ...], "out": "99_results.png"}
One clean grouped bar chart, N and seeds printed in the subtitle. No chartjunk, colour-blind-safe colours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = {"PPO": "#4C72B0", "JEPLO": "#DD8452", "Kinematic bound": "#8C8C8C"}


def render(rows: list[dict], out: Path, title: str = "Navigation in scene 1 (physics, N seeds)") -> Path:
    metrics = [("success_rate", "Success rate", 1.0), ("falls", "Falls", None),
               ("path_efficiency", "Path efficiency (SPL)", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=200)
    for ax, (key, label, cap) in zip(axes, metrics):
        names = [row["policy"] for row in rows]
        values = [float(row.get(key, 0.0)) for row in rows]
        bars = ax.bar(names, values, color=[COLOURS.get(n, "#55A868") for n in names], width=0.6)
        ax.set_title(label, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        if cap is not None:
            ax.set_ylim(0, cap * 1.08)
        for bar, value in zip(bars, values):
            text = f"{value:.0%}" if cap is not None and key != "falls" else f"{value:g}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text, ha="center", va="bottom", fontsize=9)
    n = rows[0].get("n", "?") if rows else "?"
    seeds = rows[0].get("seeds", "?") if rows else "?"
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.02, f"N = {n} per policy; seeds {seeds}; ground-truth map and pose, same planner and controller",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", required=True, help="metrics JSON ({rows: [...], out: path})")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.json).read_text())
    out = Path(args.out or data.get("out") or "99_results.png")
    render(data["rows"], out, title=data.get("title", "Navigation in scene 1 (physics, N seeds)"))
    print(f"[results] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
