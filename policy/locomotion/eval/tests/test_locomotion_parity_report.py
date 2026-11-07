"""Tests for the September-vs-GA parity report generator (pure python, no Isaac).

The point of these tests is the *verdict*, not the formatting: the criteria are what decide whether a
checkpoint may be called parity-equivalent, so each way of failing one is exercised, including the case
where every cell is individually within tolerance but the mean is not.
"""

import importlib.util
import csv
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "make_parity_report.py"
_spec = importlib.util.spec_from_file_location("_parity_report_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

REPO_ROOT = Path(__file__).resolve().parents[4]
SEPTEMBER_SWEEP = REPO_ROOT / "policy" / "locomotion" / "runs" / "2026-09-08_19-28-32_analysis" / "eval" / "eval_rough_full_sweep.csv"

needs_september_sweep = pytest.mark.skipif(
    not SEPTEMBER_SWEEP.is_file(), reason="September sweep CSV is gitignored (present on the original workstation only)"
)

# The full September sweep: 6 magnitudes x 3 terrain levels = 18 cells.
GRID = [(m, level) for m in (0.0, 40.0, 80.0, 120.0, 160.0, 200.0) for level in ("0", "4", "8")]


def write_sweep(path: Path, rates: dict[tuple[float, str], float], trials: int = 100) -> Path:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["magnitude_n", "terrain_level", "trials", "successes", "success_rate"])
        writer.writeheader()
        for (magnitude, level), rate in rates.items():
            writer.writerow(
                {
                    "magnitude_n": magnitude,
                    "terrain_level": level,
                    "trials": trials,
                    "successes": round(rate * trials),
                    "success_rate": rate,
                }
            )
    return path


def baseline_rates(offset: float = 0.0, overrides: dict | None = None) -> dict:
    """{(magnitude, level): success_rate} for the full grid."""
    rates = {(m, level): 0.9 + offset for m, level in GRID}
    rates.update(overrides or {})
    return rates


def run_report(tmp_path: Path, reference: dict, candidate: dict, label: str = "candidate"):
    ref_csv = write_sweep(tmp_path / "ref.csv", reference)
    cand_csv = write_sweep(tmp_path / "cand.csv", candidate)
    out_md = tmp_path / f"report_{label}.md"
    code = _mod.main(["--reference", str(ref_csv), "--candidate", str(cand_csv), "--out-md", str(out_md)])
    return code, out_md.read_text()


@needs_september_sweep
class TestSeptemberSweepParsing:
    def test_full_grid_is_18_cells_at_100_trials(self):
        cells = _mod.load_summary(SEPTEMBER_SWEEP)
        assert len(cells) == 18
        assert {(f"{m:g}", lvl) for m, lvl in cells} == {(f"{m:g}", lvl) for m, lvl in GRID}
        assert all(c["trials"] == 100 for c in cells.values())

    def test_named_reference_cells(self):
        cells = _mod.load_summary(SEPTEMBER_SWEEP)
        assert cells[(0.0, "0")]["success_rate"] == pytest.approx(0.94)
        assert cells[(120.0, "0")]["success_rate"] == pytest.approx(0.89)
        assert cells[(120.0, "8")]["success_rate"] == pytest.approx(0.61)

    def test_identical_sweeps_hold(self, tmp_path):
        cells = _mod.load_summary(SEPTEMBER_SWEEP)
        ref_csv = write_sweep(tmp_path / "ref.csv", {k: v["success_rate"] for k, v in cells.items()})
        cand_csv = write_sweep(tmp_path / "cand.csv", {k: v["success_rate"] for k, v in cells.items()})
        out_md = tmp_path / "september.md"
        code = _mod.main(["--reference", str(ref_csv), "--candidate", str(cand_csv), "--out-md", str(out_md)])
        assert code == 0
        assert "PARITY HOLDS" in out_md.read_text()


class TestCriteria:
    def test_identical_sweeps_hold_and_report_zero_delta(self, tmp_path):
        code, md = run_report(tmp_path, baseline_rates(), baseline_rates())
        assert code == 0
        assert "PARITY HOLDS" in md
        assert "+0.0 pp" in md

    def test_single_cell_beyond_ten_points_breaks_parity(self, tmp_path):
        code, md = run_report(
            tmp_path, baseline_rates(), baseline_rates(overrides={(120.0, "8"): 0.9 - 0.11})
        )
        assert code == 1
        assert "PARITY BROKEN" in md
        assert "120 N / level 8" in md

    def test_within_ten_points_per_cell_holds(self, tmp_path):
        code, md = run_report(tmp_path, baseline_rates(), baseline_rates(overrides={(120.0, "8"): 0.9 - 0.09}))
        assert code == 0
        assert "PARITY HOLDS" in md

    def test_every_cell_within_tolerance_but_mean_beyond_five_breaks_parity(self, tmp_path):
        # +6 pp everywhere: each cell is inside the 10 pp cell tolerance, the mean is not 5 pp.
        code, md = run_report(tmp_path, baseline_rates(), baseline_rates(offset=0.06))
        assert code == 1
        assert "PARITY BROKEN" in md
        assert "violated" in md

    def test_missing_cell_breaks_parity(self, tmp_path):
        candidate = baseline_rates()
        candidate.pop((200.0, "8"))
        code, md = run_report(tmp_path, baseline_rates(), candidate)
        assert code == 1
        assert "PARITY BROKEN" in md
        assert "missing" in md.lower()

    def test_extra_candidate_cell_is_reported(self, tmp_path):
        candidate = baseline_rates()
        candidate[(300.0, "8")] = 0.7  # magnitude the reference sweep never ran
        code, md = run_report(tmp_path, baseline_rates(), candidate)
        assert code == 0  # extra cells are reported, not a parity violation
        assert "Candidate cells absent" in md

    def test_no_overlap_cannot_compare(self, tmp_path):
        other = {(999.0, "0"): 0.5}
        code, md = run_report(tmp_path, baseline_rates(), other)
        assert code == 1
        assert "COULD NOT COMPARE" in md


class TestInputValidation:
    def test_duplicate_cell_rejected(self, tmp_path):
        path = tmp_path / "dup.csv"
        path.write_text(
            "magnitude_n,terrain_level,trials,successes,success_rate\n"
            "0.0,0,100,94,0.94\n0.0,0,100,90,0.90\n"
        )
        with pytest.raises(ValueError, match="duplicate cell"):
            _mod.load_summary(path)

    def test_missing_file_is_reported_not_raised_through_main(self, tmp_path, capsys):
        code = _mod.main(
            ["--reference", str(tmp_path / "nope.csv"), "--candidate", str(tmp_path / "nope2.csv"), "--out-md", str(tmp_path / "x.md")]
        )
        assert code == 2
        assert "not found" in capsys.readouterr().err

    def test_report_lists_reference_cells(self, tmp_path):
        code, md = run_report(
            tmp_path,
            baseline_rates(overrides={(0.0, "0"): 0.94, (120.0, "0"): 0.89, (120.0, "8"): 0.61}),
            baseline_rates(overrides={(0.0, "0"): 0.94, (120.0, "0"): 0.89, (120.0, "8"): 0.61}),
        )
        assert code == 0
        assert "Reference cell 120 N / level 8: 61% -> 61% (+0.0 pp)" in md
