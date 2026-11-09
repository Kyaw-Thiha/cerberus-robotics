"""Tests for the seeded start-goal sampler, the source seams and the planner visualisation."""

import numpy as np
import pytest

from planning.global_planner.core.astar import astar
from planning.occupancy import BLOCKED, FREE, OccupancyGrid, OccupancyConfig
from planning.sampling import sample_pairs
from planning.sources import GroundTruthMapSource, GroundTruthPoseSource, MapSource, PoseSource
from planning.viz import grid_rgb, save_plan_png, summarise_pairs


def make_grid(mask, *, res=0.1):
    grid = np.where(mask, FREE, BLOCKED).astype(np.int8)
    return OccupancyGrid(grid=grid, origin_xy=(0.0, 0.0), resolution_m=res, floor_z=0.0,
                         config=OccupancyConfig(resolution_m=res), scene_id="toy",
                         inflated=np.zeros_like(grid, dtype=bool))


def test_sampling_is_deterministic_and_within_the_distance_bounds():
    grid = make_grid(np.ones((60, 60), dtype=bool))
    a = sample_pairs(grid, n=5, seed=3, min_distance_m=1.0, max_distance_m=3.0)
    b = sample_pairs(grid, n=5, seed=3, min_distance_m=1.0, max_distance_m=3.0)
    assert len(a.pairs) == 5
    assert [(p.start, p.goal) for p in a.pairs] == [(p.start, p.goal) for p in b.pairs]
    for pair in a.pairs:
        assert 1.0 <= pair.shortest_path_m <= 3.0
        plan = astar(grid, pair.start, pair.goal)
        assert plan.success
        assert plan.length_m == pytest.approx(pair.shortest_path_m, rel=1e-6)


def test_different_seeds_give_different_pairs():
    grid = make_grid(np.ones((60, 60), dtype=bool))
    a = sample_pairs(grid, n=5, seed=1, min_distance_m=1.0, max_distance_m=5.0)
    b = sample_pairs(grid, n=5, seed=2, min_distance_m=1.0, max_distance_m=5.0)
    assert [p.start for p in a.pairs] != [p.start for p in b.pairs]


def test_unreachable_pairs_are_counted_not_returned():
    mask = np.ones((40, 40), dtype=bool)
    mask[:, 20] = False  # two disconnected halves
    grid = make_grid(mask)
    stats = sample_pairs(grid, n=10, seed=0, min_distance_m=0.5, max_distance_m=20.0, attempts_per_pair=60)
    assert stats.rejected_unreachable > 0
    for pair in stats.pairs:
        assert astar(grid, pair.start, pair.goal).success


def test_empty_free_space_returns_no_pairs():
    grid = make_grid(np.zeros((5, 5), dtype=bool))
    stats = sample_pairs(grid, n=3, seed=0)
    assert stats.pairs == [] and stats.free_cells == 0


def test_ground_truth_pose_source_returns_floats():
    source = GroundTruthPoseSource(lambda: (1, 2, 3))
    assert source.get_pose() == (1.0, 2.0, 3.0)
    assert isinstance(source, PoseSource)


def test_ground_truth_map_source_returns_the_same_grid():
    grid = make_grid(np.ones((3, 3), dtype=bool))
    source = GroundTruthMapSource(grid)
    assert source.get_grid() is grid
    assert isinstance(source, MapSource)


def test_grid_rgb_colours_free_blocked_and_unknown():
    grid = make_grid(np.ones((3, 3), dtype=bool))
    grid.grid[0, 0] = BLOCKED
    grid.grid[1, 1] = -1
    rgb = grid_rgb(grid, show_inflation=False)
    assert tuple(rgb[0, 0]) == (0.1, 0.1, 0.1)
    assert tuple(rgb[1, 1]) == (0.65, 0.65, 0.65)
    assert tuple(rgb[2, 2]) == (1.0, 1.0, 1.0)


def test_save_plan_png_writes_a_file(tmp_path):
    grid = make_grid(np.ones((20, 20), dtype=bool))
    plan = astar(grid, grid.centre_of(0, 0), grid.centre_of(19, 19))
    out = save_plan_png(grid, tmp_path / "plan.png", paths=[plan.path], start=plan.path[0], goal=plan.path[-1],
                        title="toy")
    assert out.is_file() and out.stat().st_size > 1000


def test_summarise_pairs():
    from planning.sampling import StartGoalPair

    summary = summarise_pairs([StartGoalPair((0, 0), (1, 0), 1.0), StartGoalPair((0, 0), (3, 0), 3.0)])
    assert summary == {"pairs": 2, "mean_shortest_path_m": 2.0, "min_shortest_path_m": 1.0, "max_shortest_path_m": 3.0}
