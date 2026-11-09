"""Tests for the occupancy grid: synthetic geometry, band semantics, inflation, I/O."""

import numpy as np
import pytest

from planning.occupancy import (
    BLOCKED,
    FREE,
    UNKNOWN,
    OccupancyConfig,
    OccupancyGrid,
    build_grid,
    estimate_floor_level,
    from_usd,
    inflate,
    load_npz,
    save_npz,
    save_preview_png,
)


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


def room(*extra):
    """4x4 m floor, a wall box, a table top on legs, plus whatever else is passed."""
    return combine(quad(-2, 2, -2, 2, 0.0), box(1.0, 1.2, -1.5, 1.5, 0.0, 2.6),
                   quad(0, 1, 0, 1, 0.75), box(0.0, 0.05, 0.0, 1.0, 0.0, 0.75), *extra)


def cell(grid, x, y):
    iy, ix = grid.cell_of(x, y)
    return int(grid.grid[iy, ix])


def test_open_floor_is_free_and_wall_is_blocked():
    grid = build_grid(*room(), floor_z=0.0, scene_id="room")
    assert cell(grid, -1.0, -1.0) == FREE
    assert cell(grid, 1.0, 0.0) == BLOCKED  # the wall's vertical face
    assert cell(grid, 0.02, 0.5) == BLOCKED  # a table leg
    assert cell(grid, 0.5, 0.5) == FREE  # under the table top: walk-under for a Go2


def test_outside_the_floor_footprint_is_unknown():
    grid = build_grid(*combine(quad(-1, 1, -1, 1, 0.0), box(0.0, 0.1, 0.0, 0.5, 0.0, 1.0)), floor_z=0.0)
    # the grid spans the geometry plus a margin; a corner cell with no floor sample stays UNKNOWN
    assert int(grid.grid[0, 0]) == UNKNOWN
    assert grid.free_fraction() < 1.0


def test_a_step_inside_the_band_blocks():
    # a 0.2 m platform: its tread is inside the 0.12-0.45 m band above the surrounding floor
    step = box(0.0, 1.0, 0.0, 1.0, 0.0, 0.2)
    grid = build_grid(*combine(quad(-2, 2, -2, 2, 0.0), step), floor_z=0.0)
    assert cell(grid, 0.5, 0.5) == BLOCKED


def test_a_low_rug_does_not_block():
    rug = quad(-1, 1, -1, 1, 0.03)
    grid = build_grid(*combine(quad(-2, 2, -2, 2, 0.0), rug), floor_z=0.0)
    assert cell(grid, 0.0, 0.0) == FREE


def test_floor_height_jump_beyond_max_step_blocks_both_cells():
    # two floors at different heights meeting at x=0 with nothing underneath the upper one: 0.18 m is inside the
    # 0.20 m window, so both sides count as "this level" and the jump rule is what must stop the planner.
    low = quad(-2, 0, -1, 1, 0.0)
    high = quad(0, 2, -1, 1, 0.18)
    grid = build_grid(*combine(low, high), floor_z=0.0)
    assert cell(grid, -1.0, 0.0) == FREE
    assert cell(grid, 1.0, 0.0) == FREE
    assert cell(grid, -0.025, 0.0) == BLOCKED  # the last cell of the low floor
    assert cell(grid, 0.025, 0.0) == BLOCKED   # the first cell of the high floor


def test_a_different_level_reads_unknown_not_free():
    upper_floor = quad(0, 2, -1, 1, 3.0)
    grid = build_grid(*combine(quad(-2, 2, -2, 2, 0.0), upper_floor), floor_z=0.0)
    # the cell is under the upper slab, so its lowest up-facing surface is still the ground floor
    assert cell(grid, 0.5, 0.0) == FREE
    # ... but a cell whose ONLY surface is the upper floor is not part of this level
    outside = build_grid(*combine(quad(-2, 0, -2, 2, 0.0), quad(0.0, 2, -1, 1, 3.0)), floor_z=0.0)
    assert cell(outside, 1.0, 0.0) == UNKNOWN


def test_estimate_floor_level_uses_the_dominant_height():
    heights = np.array([0.0] * 100 + [0.75] * 5 + [2.6] * 2)
    assert estimate_floor_level(heights) == pytest.approx(0.0)
    assert estimate_floor_level(heights, floor_z=0.75) == pytest.approx(0.75)


def test_inflate_grows_a_disk_by_the_robot_radius():
    grid = np.full((21, 21), FREE, dtype=np.int8)
    grid[10, 10] = BLOCKED
    inflated = inflate(grid, radius_m=0.15, resolution_m=0.05)  # 3 cells
    assert inflated[10, 10]
    assert inflated[10, 13] and inflated[13, 10]
    assert not inflated[10, 14]
    assert not inflated[13, 13]  # corner of the 3-cell box is outside the circle
    assert not inflated[10, 6]


def test_inflate_radius_zero_is_the_blocked_mask():
    grid = np.full((5, 5), FREE, dtype=np.int8)
    grid[2, 2] = BLOCKED
    assert np.array_equal(inflate(grid, 0.0, 0.05), grid == BLOCKED)


def test_grid_helpers_round_trip_world_coordinates():
    grid = OccupancyGrid(grid=np.full((10, 10), FREE, dtype=np.int8), origin_xy=(1.0, 2.0), resolution_m=0.5,
                         floor_z=0.0, config=OccupancyConfig())
    assert grid.cell_of(1.1, 2.1) == (0, 0)
    assert grid.centre_of(0, 0) == pytest.approx((1.25, 2.25))
    assert grid.inside(9, 9) and not grid.inside(10, 0)
    assert grid.is_free(1.1, 2.1)


def test_npz_round_trip(tmp_path):
    grid = build_grid(*room(), floor_z=0.0, scene_id="room")
    path = save_npz(grid, tmp_path / "occupancy.npz")
    back = load_npz(path)
    assert back.scene_id == "room"
    assert back.resolution_m == grid.resolution_m
    assert back.floor_z == grid.floor_z
    assert np.array_equal(back.grid, grid.grid)
    assert np.array_equal(back.inflated, grid.inflated)
    assert back.config == grid.config


def test_preview_png_is_written(tmp_path):
    grid = build_grid(*room(), floor_z=0.0, scene_id="room")
    out = save_preview_png(grid, tmp_path / "occupancy.png")
    assert out.is_file() and out.stat().st_size > 1000


def test_config_rejects_a_bad_band():
    with pytest.raises(ValueError):
        OccupancyConfig(band_low_m=0.5, band_high_m=0.2)


def test_config_from_yaml_is_the_checked_in_config():
    config = OccupancyConfig.from_yaml()
    assert config.resolution_m == 0.05
    assert config.robot_radius_m == 0.25
    assert config.floor_window_m > config.max_step_m


def test_from_usd_reads_a_synthetic_stage(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from common.cerberus_worlds.scene_geometry import write_triangle_mesh_usd

    points, faces = room()
    usd = write_triangle_mesh_usd(points, faces, tmp_path / "room.usd", root_prim="/R", mesh_prim="/R/M")
    grid = from_usd(usd, scene_id="room", floor_z=0.0)
    assert cell(grid, -1.0, -1.0) == FREE
    assert cell(grid, 1.0, 0.0) == BLOCKED
    stage = Usd.Stage.Open(str(usd))
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    assert stage.GetPrimAtPath("/R/M").IsA(UsdGeom.Mesh)


def test_a_sub_cell_thick_wall_cannot_be_threaded():
    # a 5 cm wall is exactly one cell thick: without the thickening step the planner finds a 1-cell tunnel
    wall = box(0.0, 2.0, 0.0, 0.05, 0.0, 2.6)
    grid = build_grid(*combine(quad(-2, 2, -2, 2, 0.0), wall), floor_z=0.0)
    for y in (-0.02, 0.0, 0.02, 0.04):
        assert cell(grid, 1.0, y) == BLOCKED
