#!/usr/bin/env python3
"""Build the September-vs-GA parity table, plot and verdict from two push-recovery sweeps.

A checkpoint stores weights, not a physics engine. Moving the same checkpoint to a new stack (Isaac Lab
2.3.2 -> 3.0) can change behaviour through the quaternion order, the actuator model, the asset, or the
checkpoint loader -- so "the numbers moved" is only interpretable next to the contract check
(``contract_check.py``) that says whether the checkpoint even sees the same problem.

Parity criteria (from the H2 handoff, restated here so the report cannot drift from them):
  * per-cell success within 10 percentage points of the September CSV
  * mean across cells within 5 points
Either violation -> PARITY BROKEN. Both satisfied -> PARITY HOLDS.

Pure python apart from matplotlib for ``--out-plot`` (imported lazily), so this runs and is tested
without Isaac.

Usage:
    python3 policy/locomotion/eval/make_parity_report.py \
        --reference policy/locomotion/runs/2026-09-08_19-28-32_analysis/eval/eval_rough_full_sweep.csv \
        --candidate policy/locomotion/runs/<ga run>/eval/eval_rough_full_sweep.csv \
        --out-md docs/reports/phase0_ga_parity.md --out-plot docs/reports/phase0_ga_parity.png

Exit codes: 0 report written, 1 criteria violated (report still written), 2 bad input.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PER_CELL_TOLERANCE = 0.10  # 10 percentage points
MEAN_TOLERANCE = 0.05  # 5 percentage points


def load_summary(path: str | Path) -> dict[tuple[float, str], dict]:
    """Read a push-recovery summary CSV -> {(magnitude_n, terrain_level): row}."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"sweep CSV not found: {path}")
    cells: dict[tuple[float, str], dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["magnitude_n"]), str(row["terrain_level"]))
            if key in cells:
                raise ValueError(f"{path}: duplicate cell {key}")
            cells[key] = {
                "magnitude_n": key[0],
                "terrain_level": key[1],
                "trials": int(row["trials"]),
                "successes": int(row["successes"]),
                "success_rate": float(row["success_rate"]),
            }
    if not cells:
        raise ValueError(f"{path}: no data rows")
    return cells


def _cell_sort_key(key: tuple[float, str]) -> tuple[float, float]:
    mag, level = key
    try:
        level_value = float(level)
    except ValueError:
        level_value = -1.0
    return (mag, level_value)


def join_cells(reference: dict, candidate: dict, label: str = "candidate") -> tuple[list[dict], list[tuple]]:
    """Pair the two sweeps cell by cell -> (rows, candidate-only cell keys).

    Missing cells are reported, never silently skipped: a sweep that quietly dropped the hardest
    level would otherwise look like parity.
    """
    rows = []
    for key in sorted(reference, key=_cell_sort_key):
        ref = reference[key]
        cand = candidate.get(key)
        row = {
            "magnitude_n": ref["magnitude_n"],
            "terrain_level": ref["terrain_level"],
            "reference_rate": ref["success_rate"],
            "reference_trials": ref["trials"],
            f"{label}_rate": cand["success_rate"] if cand else None,
            f"{label}_trials": cand["trials"] if cand else None,
        }
        row["delta_pp"] = (
            (cand["success_rate"] - ref["success_rate"]) * 100.0 if cand is not None else None
        )
        rows.append(row)
    extra = [k for k in candidate if k not in reference]
    return rows, extra


def evaluate_parity(
    rows: list[dict],
    extra_cells: list[tuple],
    per_cell_tolerance: float = PER_CELL_TOLERANCE,
    mean_tolerance: float = MEAN_TOLERANCE,
) -> dict:
    """Apply the handoff's two criteria. Returns the numbers either way, plus the verdict."""
    paired = [r for r in rows if r["delta_pp"] is not None]
    if not paired:
        return {
            "verdict": "COULD NOT COMPARE",
            "reason": "no overlapping cells between the two sweeps",
            "cells_compared": 0,
            "mean_delta_pp": None,
            "mean_ok": False,
            "per_cell_tolerance_pp": per_cell_tolerance * 100.0,
            "mean_tolerance_pp": mean_tolerance * 100.0,
            "per_cell_failures": [],
            "missing_cells": [f"{r['magnitude_n']:g} N / level {r['terrain_level']}" for r in rows],
            "extra_cells": [f"{m:g} N / level {lvl}" for (m, lvl) in extra_cells],
            "max_abs_delta_pp": None,
            "worst_cell": None,
        }
    deltas = [r["delta_pp"] for r in paired]
    mean_ref = sum(r["reference_rate"] for r in paired) / len(paired)
    cand_key = next((k for k in paired[0] if k.endswith("_rate") and k != "reference_rate"), None)
    mean_cand = sum(r[cand_key] for r in paired) / len(paired) if cand_key else None
    mean_delta = (mean_cand - mean_ref) * 100.0 if mean_cand is not None else None
    per_cell_failures = [
        f"{r['magnitude_n']:g} N / level {r['terrain_level']}: {r['delta_pp']:+.1f} pp" for r in paired
        if abs(r["delta_pp"]) > per_cell_tolerance * 100.0
    ]
    missing = [f"{r['magnitude_n']:g} N / level {r['terrain_level']}" for r in rows if r["delta_pp"] is None]
    mean_ok = mean_delta is not None and abs(mean_delta) <= mean_tolerance * 100.0
    cells_ok = not per_cell_failures
    verdict = "PARITY HOLDS" if (cells_ok and mean_ok and not missing) else "PARITY BROKEN"
    return {
        "verdict": verdict,
        "cells_compared": len(paired),
        "mean_reference": mean_ref,
        "mean_candidate": mean_cand,
        "mean_delta_pp": mean_delta,
        "mean_tolerance_pp": mean_tolerance * 100.0,
        "mean_ok": mean_ok,
        "per_cell_tolerance_pp": per_cell_tolerance * 100.0,
        "per_cell_failures": per_cell_failures,
        "missing_cells": missing,
        "extra_cells": [f"{m:g} N / level {lvl}" for (m, lvl) in extra_cells],
        "max_abs_delta_pp": max(abs(d) for d in deltas),
        "worst_cell": max(paired, key=lambda r: abs(r["delta_pp"])),
    }


def find_cell(rows: list[dict], magnitude_n: float, terrain_level: str) -> dict | None:
    for r in rows:
        if r["magnitude_n"] == float(magnitude_n) and str(r["terrain_level"]) == str(terrain_level):
            return r
    return None


def render_markdown(
    rows: list[dict],
    result: dict,
    *,
    title: str,
    reference_label: str,
    candidate_label: str,
    contract: dict | None = None,
    plot_path: str | None = None,
    notes: list[str] | None = None,
) -> str:
    cand_rate = next((k for k in rows[0] if k.endswith("_rate") and k != "reference_rate"), "candidate_rate")
    lines = [f"# {title}", ""]
    if notes:
        lines += [*notes, ""]

    lines += [
        "## Parity criteria",
        "",
        f"- per-cell success within {result['per_cell_tolerance_pp']:.0f} percentage points of {reference_label}",
        f"- mean across cells within {result['mean_tolerance_pp']:.0f} points",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
    ]
    if result.get("reason"):
        lines += [f"Reason: {result['reason']}", ""]

    if "mean_delta_pp" in result and result["mean_delta_pp"] is not None:
        lines += [
            (
                f"Mean success: {reference_label} {result['mean_reference']:.1%} -> {candidate_label} "
                f"{result['mean_candidate']:.1%} ({result['mean_delta_pp']:+.1f} pp; tolerance "
                f"±{result['mean_tolerance_pp']:.0f} pp, {'ok' if result['mean_ok'] else 'violated'})"
            ),
            "",
            (
                f"Largest per-cell move: {result['max_abs_delta_pp']:.1f} pp at "
                f"{result['worst_cell']['magnitude_n']:g} N / level {result['worst_cell']['terrain_level']}"
            ),
            "",
        ]

    lines += [
        "## Side by side (100 trials per cell)",
        "",
        f"| push | level | {reference_label} | {candidate_label} | delta |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        cand = r[cand_rate]
        delta = r["delta_pp"]
        lines.append(
            f"| {r['magnitude_n']:g} N | {r['terrain_level']} | {r['reference_rate']:.0%} | "
            f"{'n/a' if cand is None else f'{cand:.0%}'} | "
            f"{'n/a' if delta is None else f'{delta:+.1f} pp'} |"
        )

    if result["per_cell_failures"]:
        lines += ["", "### Cells outside the per-cell tolerance", ""]
        lines += [f"- {line}" for line in result["per_cell_failures"]]
    if result["missing_cells"]:
        lines += ["", "### Reference cells missing from the candidate sweep", ""]
        lines += [f"- {line}" for line in result["missing_cells"]]
    if result["extra_cells"]:
        lines += ["", "### Candidate cells absent from the reference sweep", ""]
        lines += [f"- {line}" for line in result["extra_cells"]]

    if contract:
        lines += ["", "## Contract check", "", f"Reference: `{contract.get('reference')}`", ""]
        lines += [f"Verdict: **{contract.get('verdict')}**", "", "```", contract.get("table", ""), "```"]
        if contract.get("mismatches"):
            lines += ["", "Mismatches (these invalidate a like-for-like reading of the table):", ""]
            lines += [f"- {m}" for m in contract["mismatches"]]

    if plot_path:
        lines += ["", f"![parity]({Path(plot_path).name})", ""]
    return "\n".join(lines) + "\n"


def plot_parity(rows: list[dict], path: str | Path, *, reference_label: str, candidate_label: str, title: str):
    """Success vs push magnitude, one line per terrain level, both stacks. Lazy matplotlib import."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cand_rate = next((k for k in rows[0] if k.endswith("_rate") and k != "reference_rate"), "candidate_rate")
    levels = sorted({r["terrain_level"] for r in rows}, key=lambda s: float(s) if s.replace(".", "", 1).isdigit() else -1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = plt.cm.viridis([i / max(1, len(levels) - 1) for i in range(len(levels))]) if len(levels) > 1 else ["C0"]
    for level, color in zip(levels, colors, strict=False):
        subset = [r for r in rows if r["terrain_level"] == level]
        xs = [r["magnitude_n"] for r in subset]
        ax.plot(xs, [r["reference_rate"] * 100 for r in subset], marker="o", color=color, label=f"{reference_label} level {level}")
        ax.plot(
            xs,
            [None if r[cand_rate] is None else r[cand_rate] * 100 for r in subset],
            marker="x",
            linestyle="--",
            color=color,
            label=f"{candidate_label} level {level}",
        )
    ax.set_xlabel("push magnitude (N)")
    ax.set_ylabel("recovery success (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True, help="Reference (September) summary CSV.")
    parser.add_argument("--candidate", required=True, help="Candidate (GA) summary CSV.")
    parser.add_argument("--reference-label", default="September")
    parser.add_argument("--candidate-label", default="GA")
    parser.add_argument("--title", default="Phase 0 parity: September vs GA")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-plot", default=None)
    parser.add_argument("--contract", default=None, help="Contract-check text file written by push_recovery_eval.py.")
    parser.add_argument("--note", action="append", default=[], help="Extra line for the report header. Repeatable.")
    args = parser.parse_args(argv)

    try:
        reference = load_summary(args.reference)
        candidate = load_summary(args.candidate)
    except (FileNotFoundError, ValueError) as e:
        print(f"[parity] {e}", file=sys.stderr)
        return 2

    rows, extra = join_cells(reference, candidate, label="candidate")
    result = evaluate_parity(rows, extra)

    # The handoff names three reference cells; surface them explicitly so the report answers the
    # question that was actually asked rather than leaving it to be read off the table.
    for magnitude, level in ((0.0, "0"), (120.0, "0"), (120.0, "8")):
        cell = find_cell(rows, magnitude, level)
        if cell is not None and cell["delta_pp"] is not None:
            args.note = [
                *args.note,
                (
                    f"Reference cell {magnitude:g} N / level {level}: {cell['reference_rate']:.0%} -> "
                    f"{cell['candidate_rate']:.0%} ({cell['delta_pp']:+.1f} pp)"
                ),
            ]

    contract = None
    if args.contract and Path(args.contract).is_file():
        text = Path(args.contract).read_text()
        contract = {
            "reference": next((line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("reference:")), ""),
            "verdict": next((line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("verdict:")), ""),
            "table": text,
            "mismatches": [
                line for line in text.splitlines() if line.startswith("- ") and "reference=" in line
            ],
        }

    plot_path = None
    if args.out_plot:
        plot_path = plot_parity(
            rows,
            args.out_plot,
            reference_label=args.reference_label,
            candidate_label=args.candidate_label,
            title=args.title,
        )

    md = render_markdown(
        rows,
        result,
        title=args.title,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
        contract=contract,
        plot_path=plot_path,
        notes=args.note,
    )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    print(f"[parity] {result['verdict']}: {result['cells_compared']} cells compared")
    if result.get("mean_delta_pp") is not None:
        print(
            f"[parity] mean {args.reference_label} {result['mean_reference']:.1%} -> "
            f"{args.candidate_label} {result['mean_candidate']:.1%} ({result['mean_delta_pp']:+.1f} pp)"
        )
    for line in result["per_cell_failures"]:
        print(f"[parity] outside tolerance: {line}")
    print(f"[parity] wrote {out_md}" + (f" and {plot_path}" if plot_path else ""))
    return 0 if result["verdict"] == "PARITY HOLDS" else 1


if __name__ == "__main__":
    sys.exit(main())
