"""Diagnostics: per-cell blocked reason and the up-facing height histogram (evidence, no threshold change)."""

import numpy as np
import pytest

from planning.occupancy import (
    REASON_FREE,
    REASON_HORIZONTAL_BAND,
    REASON_NAMES,
    REASON_STEP_RULE,
    REASON_THICKENED,
    REASON_UNKNOWN,
    REASON_VERTICAL_BAND,
    OccupancyConfig,
    build_grid,
)
from planning.viz import save_reason_png, save_up_histogram_png


def quad(x0, x1, y0, y1, z):
    points = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=float)
    return points, np.array([[0, 1, 2], [0, 2, 3]])


def box(x0, x1, y0, y1, z0, z1):
    points = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=float)
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    return points, faces


def combine(*parts):
    points, faces, offset = [], [], 0
    for p, f in parts:
        points.append(p)
        faces.append(f + offset)
        offset += len(p)
    return np.concatenate(points), np.concatenate(faces)


def room_with_platform(*extra):
    """4x4 floor, a wall box, and a 1x1 m platform TOP at 0.2 m (the surface the band rule sees).

    The platform is authored as a quad, not a box: the synthetic ``box`` helper's winding is not consistent, and a
    box top that faces down would exercise the overhang rule instead of the one under test.
    """
    platform = quad(0.5, 1.5, 0.5, 1.5, 0.2)
    return combine(quad(-2, 2, -2, 2, 0.0), box(1.8, 1.9, -1.0, 1.0, 0.0, 2.6), platform, *extra)


def diagnose(*parts, floor_z=0.0):
    return build_grid(*parts, config=OccupancyConfig(), floor_z=floor_z, scene_id="diag", collect_diagnostics=True)


def cell_reason(grid, x, y):
    iy, ix = grid.cell_of(x, y)
    return int(grid.diagnostics["category"][iy, ix])


def test_default_build_has_no_diagnostics():
    grid = build_grid(*room_with_platform(), floor_z=0.0)
    assert grid.diagnostics is None


def test_raised_platform_is_labelled_horizontal_band():
    grid = diagnose(*room_with_platform())
    # the platform top at 0.2 m is inside 0.12..0.45 above the floor underneath it
    assert cell_reason(grid, 1.0, 1.0) == REASON_HORIZONTAL_BAND


def test_wall_face_is_labelled_vertical_band():
    grid = diagnose(*room_with_platform())
    assert cell_reason(grid, 1.85, 0.0) == REASON_VERTICAL_BAND


def test_open_floor_and_outside_are_labelled():
    grid = diagnose(*room_with_platform())
    assert cell_reason(grid, -1.5, -1.5) == REASON_FREE
    # cells with no floor sample at all are UNKNOWN; find one rather than assuming a corner
    no_floor = np.isnan(grid.diagnostics["up_above_floor"])
    assert no_floor.any()
    assert set(np.unique(grid.diagnostics["category"][no_floor])).issubset({REASON_UNKNOWN})


def test_up_above_floor_records_the_platform_height():
    grid = diagnose(*room_with_platform())
    iy, ix = grid.cell_of(1.0, 1.0)
    assert grid.diagnostics["up_above_floor"][iy, ix] == pytest.approx(0.2, abs=0.01)


def test_histogram_covers_the_platform_and_is_counted():
    grid = diagnose(*room_with_platform())
    hist = grid.diagnostics["hist_blocked"]
    assert sum(hist) > 0
    edges = grid.diagnostics["hist_bin_edges_m"]
    assert edges[0] == 0.0 and edges[-1] >= 3.0


def test_every_category_has_a_name():
    grid = diagnose(*room_with_platform())
    for reason in np.unique(grid.diagnostics["category"]):
        assert int(reason) in REASON_NAMES


def test_step_rule_is_labelled_on_a_steep_floor_jump():
    low = quad(-2, 0, -1, 1, 0.0)
    high = quad(0, 2, -1, 1, 0.18)  # inside the 0.20 m floor window, beyond the 0.12 m step rule
    grid = diagnose(*combine(low, high))
    assert cell_reason(grid, -0.025, 0.0) == REASON_STEP_RULE
    assert cell_reason(grid, 0.025, 0.0) == REASON_STEP_RULE


def test_thin_surface_thickening_is_labelled_when_no_other_rule_fires():
    wall = box(0.0, 2.0, 0.0, 0.05, 0.0, 2.6)  # 5 cm wall, one cell thick
    grid = diagnose(*combine(quad(-2, 2, -2, 2, 0.0), wall))
    reasons = {int(r) for r in np.unique(grid.diagnostics["category"])}
    assert REASON_THICKENED in reasons


def test_previews_and_histogram_write_files(tmp_path):
    grid = diagnose(*room_with_platform())
    reasons = save_reason_png(grid, tmp_path / "reasons.png")
    hist = save_up_histogram_png(grid.diagnostics, tmp_path / "hist.png")
    assert reasons.is_file() and reasons.stat().st_size > 1000
    assert hist.is_file() and hist.stat().st_size > 1000
