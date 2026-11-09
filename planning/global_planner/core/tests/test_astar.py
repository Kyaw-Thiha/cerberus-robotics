"""A* tests: optimality, collision-freedom, failure modes, corner cutting, inflation."""

import math

import numpy as np
import pytest

from planning.global_planner.core.astar import (
    astar,
    line_of_sight,
    path_is_collision_free,
    shortcut_cells,
    traversable_mask,
)
from planning.occupancy import BLOCKED, FREE, OccupancyGrid
from planning.occupancy import OccupancyConfig


def make_grid(mask, *, res=0.1, inflate=False, radius_m=0.1):
    """``mask`` True = free. Origin at (0, 0); cell centres at (i+0.5)*res."""
    grid = np.where(mask, FREE, BLOCKED).astype(np.int8)
    out = OccupancyGrid(
        grid=grid,
        origin_xy=(0.0, 0.0),
        resolution_m=res,
        floor_z=0.0,
        config=OccupancyConfig(resolution_m=res, robot_radius_m=radius_m),
        scene_id="toy",
    )
    if inflate:
        from planning.occupancy import inflate as dilate

        out.inflated = dilate(out.grid, radius_m, res)
    else:
        out.inflated = np.zeros_like(grid, dtype=bool)
    return out


def centre(grid, iy, ix):
    return grid.centre_of(iy, ix)


def test_straight_line_is_shortest_and_exact():
    grid = make_grid(np.ones((10, 10), dtype=bool))
    plan = astar(grid, centre(grid, 0, 0), centre(grid, 0, 8))
    assert plan.success
    assert plan.length_m == pytest.approx(0.8, abs=1e-9)
    assert plan.cells[0] == (0, 0) and plan.cells[-1] == (0, 8)


def test_diagonal_is_straight_after_shortcutting():
    grid = make_grid(np.ones((10, 10), dtype=bool))
    plan = astar(grid, centre(grid, 0, 0), centre(grid, 9, 9))
    assert plan.success
    assert plan.length_m == pytest.approx(9 * 0.1 * math.sqrt(2), rel=1e-6)


def test_path_avoids_a_wall_and_uses_the_gap():
    mask = np.ones((11, 11), dtype=bool)
    mask[:, 5] = False
    mask[0, 5] = True  # gap at the bottom row
    grid = make_grid(mask)
    plan = astar(grid, centre(grid, 5, 0), centre(grid, 5, 10))
    assert plan.success
    assert plan.length_m > 1.0  # not the 1.0 m straight line through the wall
    assert any(cell[1] == 5 for cell in plan.cells)
    assert all(mask[iy, ix] for iy, ix in plan.cells)


def test_unreachable_is_a_failure_not_an_exception():
    mask = np.ones((11, 11), dtype=bool)
    mask[:, 5] = False
    grid = make_grid(mask)
    plan = astar(grid, centre(grid, 5, 0), centre(grid, 5, 10))
    assert not plan.success
    assert plan.reason == "unreachable"


def test_blocked_start_and_goal_are_named_failures():
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    grid = make_grid(mask)
    assert astar(grid, centre(grid, 2, 2), centre(grid, 0, 0)).reason == "start_blocked"
    assert astar(grid, centre(grid, 0, 0), centre(grid, 2, 2)).reason == "goal_blocked"


def test_outside_the_grid_is_a_failure():
    grid = make_grid(np.ones((5, 5), dtype=bool))
    plan = astar(grid, (-1.0, -1.0), centre(grid, 0, 0))
    assert not plan.success and plan.reason == "start_outside_grid"


def test_no_corner_cutting_between_two_blocked_diagonals():
    # free cells on the diagonal, blocked cells on both orthogonals: a diagonal move would cut a corner
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = True
    grid = make_grid(mask)
    plan = astar(grid, centre(grid, 0, 0), centre(grid, 2, 2))
    assert not plan.success  # the only route would cut a corner, so there is none


def test_inflation_can_close_a_one_cell_corridor():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, :] = True  # a 1-cell corridor
    open_grid = make_grid(mask)
    assert astar(open_grid, centre(open_grid, 4, 0), centre(open_grid, 4, 8)).success
    inflated = make_grid(mask, inflate=True, radius_m=0.15)  # 1.5 cells: the corridor is swallowed
    assert not astar(inflated, centre(inflated, 4, 0), centre(inflated, 4, 8)).success


def test_unknown_cells_are_blocked_unless_allowed():
    grid = make_grid(np.ones((6, 6), dtype=bool))
    grid.grid[3, 3] = -1  # UNKNOWN
    blocked = astar(grid, centre(grid, 0, 0), centre(grid, 5, 5))
    allowed = astar(grid, centre(grid, 0, 0), centre(grid, 5, 5), allow_unknown=True)
    assert blocked.success and allowed.success
    assert allowed.length_m <= blocked.length_m + 1e-9


def test_shortcut_keeps_line_of_sight_and_removes_detours():
    free = np.ones((10, 10), dtype=bool)
    cells = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    assert shortcut_cells(free, cells) == [(0, 0), (4, 0)]
    assert line_of_sight(free, (0, 0), (4, 0))
    free[2, 2] = False
    assert not line_of_sight(free, (0, 0), (4, 4))


def test_traversable_mask_excludes_blocked_and_inflated():
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    grid = make_grid(mask, inflate=True, radius_m=0.2)
    free = traversable_mask(grid)
    assert not free[2, 2]
    assert not free[2, 3]  # inflated
    assert free[0, 0]


def test_planned_paths_are_collision_free_on_a_cluttered_grid():
    mask = np.ones((40, 40), dtype=bool)
    mask[10:30, 18] = False
    mask[18, 10:28] = False
    mask[10:30, 18] = True  # a gap in the vertical wall
    grid = make_grid(mask)
    plan = astar(grid, centre(grid, 5, 5), centre(grid, 35, 35))
    assert plan.success
    assert path_is_collision_free(grid, plan.path)
    assert all(mask[iy, ix] for iy, ix in plan.cells)


def test_path_check_catches_a_segment_through_a_wall():
    mask = np.ones((20, 20), dtype=bool)
    mask[:, 10] = False
    grid = make_grid(mask)
    straight_through = [centre(grid, 5, 2), centre(grid, 5, 17)]
    assert not path_is_collision_free(grid, straight_through)
