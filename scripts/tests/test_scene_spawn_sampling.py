import math

import numpy as np
import pytest
import scene_spawn_sampling as ss
from scene_admission_logic import Thresholds

TH = Thresholds()
NAN = float("nan")


def make_spec(nx=40, ny=40, cell=0.25):
    return ss.GridSpec(0.0, 0.0, cell, nx, ny)


def stacks(spec, cell_fn):
    """Hit stacks (layers x cells): `cell_fn(ix, iy)` -> [(z, nz), ...] descending z, as a downward ray meets them."""
    n = spec.nx * spec.ny
    hit_z = np.full((TH.max_hits_per_column, n), np.nan, dtype=np.float32)
    hit_nz = np.full_like(hit_z, np.nan)
    for iy in range(spec.ny):
        for ix in range(spec.nx):
            for k, (z, nz) in enumerate(cell_fn(ix, iy)):
                hit_z[k, iy * spec.nx + ix] = z
                hit_nz[k, iy * spec.nx + ix] = nz
    return hit_z, hit_nz


ROOM = [(3.0, 1.0), (2.9, -1.0), (0.0, 1.0)]  # roof top, ceiling underside, floor top
SOLID_BOX = [(3.0, 1.0), (2.9, -1.0), (0.9, 1.0), (0.0, -1.0)]  # closed box: top face, bottom face (no floor hit)
OUTDOORS = [(0.0, 1.0)]  # open ground, nothing above


def test_empty_room_spawns_at_the_centre_facing_open_space():
    spec = make_spec()
    plan = ss.plan_spawns(*stacks(spec, lambda ix, iy: ROOM), spec, TH)
    assert plan.level_z == pytest.approx(0.0)
    first = plan.spawns[0]
    assert first.roofed and first.floor_z == pytest.approx(0.0)
    assert 3.5 < first.x < 6.5 and 3.5 < first.y < 6.5  # 10 x 10 m room: the centre has the largest free radius
    assert first.open_radius_m == pytest.approx(TH.max_open_radius_m, abs=0.3)


def test_furniture_is_avoided_and_its_underside_is_not_floor():
    spec = make_spec()

    def cell(ix, iy):
        return SOLID_BOX if 14 <= ix < 26 and 14 <= iy < 26 else ROOM  # a 3 x 3 m block in the room centre

    plan = ss.plan_spawns(*stacks(spec, cell), spec, TH)
    assert plan.spawns
    for s in plan.spawns:
        assert not (3.4 <= s.x <= 6.6 and 3.4 <= s.y <= 6.6), s  # never inside / right next to the block
        assert s.open_radius_m >= TH.min_open_radius_m


def test_a_closet_too_small_for_the_robot_leaves_no_spawn():
    spec = make_spec(4, 4)  # 1 x 1 m of floor: free half-width at most 0.375 m < min_open_radius_m
    plan = ss.plan_spawns(*stacks(spec, lambda ix, iy: ROOM), spec, TH)
    assert plan.spawns == [] and plan.free_cells == 16
    assert "open radius" in plan.notes[0]


def test_no_floor_with_enough_headroom_means_no_level():
    spec = make_spec(10, 10)
    low = [(0.6, 1.0), (0.5, -1.0)]  # only a 0.6 m slab seen from above: its underside is not floor
    cand, roofed = ss.floor_candidates(*stacks(spec, lambda ix, iy: low), TH)
    assert ss.choose_level(np.full_like(cand, np.nan), roofed, TH) is None
    assert np.isnan(cand[1]).all()  # the underside (k=1) is never a candidate


def test_roofed_floor_wins_over_larger_open_ground():
    spec = make_spec(40, 20)

    def cell(ix, iy):
        return ROOM if ix < 12 else OUTDOORS  # a 3 x 5 m room next to a 7 x 5 m yard at the same height

    plan = ss.plan_spawns(*stacks(spec, cell), spec, TH)
    assert plan.spawns[0].roofed and plan.spawns[0].x < 3.0
    assert any(not s.roofed for s in plan.spawns[1:]) or len(plan.spawns) == 1


def test_no_roof_anywhere_falls_back_to_open_sky_with_a_note():
    spec = make_spec(20, 20)
    plan = ss.plan_spawns(*stacks(spec, lambda ix, iy: OUTDOORS), spec, TH)
    assert plan.spawns and not plan.spawns[0].roofed
    assert any("no roofed floor" in n for n in plan.notes)


def test_main_level_is_chosen_by_area_and_steps_block_cells():
    spec = make_spec(40, 20)

    def cell(ix, iy):  # lower hall (z=0) 3/4 of the area, a raised platform (z=0.5) on the right quarter
        floor = 0.5 if ix >= 30 else 0.0
        return [(3.0, 1.0), (2.9, -1.0), (floor, 1.0)]

    plan = ss.plan_spawns(*stacks(spec, cell), spec, TH)
    assert plan.level_z == pytest.approx(0.0)
    assert all(s.floor_z == pytest.approx(0.0) and s.x < 7.5 - 0.5 for s in plan.spawns)


def test_ties_between_levels_prefer_the_one_nearest_zero():
    spec = make_spec(40, 20)

    def cell(ix, iy):  # two equally large floors, one at z=0.0 and one at z=3.2 (upper storey)
        floor = 0.0 if ix < 20 else 3.2
        return [(floor + 3.0, 1.0), (floor + 2.9, -1.0), (floor, 1.0)]

    plan = ss.plan_spawns(*stacks(spec, cell), spec, TH)
    assert plan.level_z == pytest.approx(0.0)
    assert all(s.x < 5.0 for s in plan.spawns)


def test_spawns_are_separated_limited_and_deterministic():
    spec = make_spec(80, 40)
    hz, hn = stacks(spec, lambda ix, iy: ROOM)
    a = ss.plan_spawns(hz, hn, spec, TH)
    b = ss.plan_spawns(hz, hn, spec, TH)
    assert a.spawns == b.spawns
    assert len(a.spawns) == TH.max_spawns
    for i, s in enumerate(a.spawns):
        assert all(math.hypot(s.x - t.x, s.y - t.y) >= TH.spawn_separation_m for t in a.spawns[:i])
    radii = [s.open_radius_m for s in a.spawns]
    assert radii == sorted(radii, reverse=True)  # best first


def test_yaw_points_toward_the_longest_free_run():
    free = np.zeros((20, 40), dtype=bool)
    free[10, 5:] = True  # a corridor along +x from column 5
    free[5:16, 5] = True
    assert ss.best_yaw(free, 10, 5, 30) == pytest.approx(0.0)
    free2 = np.zeros((40, 20), dtype=bool)
    free2[5:, 10] = True  # a corridor along +y
    assert ss.best_yaw(free2, 5, 10, 30) == pytest.approx(math.pi / 2)


def test_grid_planning_coarsens_to_the_cell_budget():
    spec = ss.plan_grid(np.array([0.0, 0.0]), np.array([10.0, 5.0]), TH)
    assert (spec.cell, spec.nx, spec.ny) == (0.25, 40, 20)
    big = ss.plan_grid(np.array([0.0, 0.0]), np.array([400.0, 400.0]), TH)
    assert big.cell > 0.25 and big.nx * big.ny <= TH.max_grid_cells * 1.01
    with pytest.raises(ValueError):
        ss.plan_grid(np.array([1.0, 1.0]), np.array([1.0, 3.0]), TH)
    xy = spec.xy()
    assert xy.shape == (800, 2) and tuple(xy[0]) == (0.125, 0.125) and tuple(xy[1]) == (0.375, 0.125)


def test_robust_bounds_ignore_a_stray_ground_plane():
    rng = np.random.default_rng(0)
    house = np.column_stack([rng.uniform(0, 10, 50_000), rng.uniform(0, 8, 50_000), rng.uniform(0, 3, 50_000)])
    plane = np.array([[-1e6, -1e6, 0.0], [1e6, -1e6, 0.0], [1e6, 1e6, 0.0], [-1e6, 1e6, 0.0]])
    lo, hi = ss.robust_xy_bounds(np.vstack([house, plane]))
    assert lo == pytest.approx([-1.0, -1.0], abs=0.1) and hi == pytest.approx([11.0, 9.0], abs=0.1)


def test_triangulate_faces_quads_ngons_and_triangles():
    tri = ss.triangulate_faces(np.array([3, 3]), np.array([0, 1, 2, 2, 1, 3]))
    assert tri.tolist() == [[0, 1, 2], [2, 1, 3]]
    quad = ss.triangulate_faces(np.array([4]), np.array([0, 1, 2, 3]))
    assert quad.tolist() == [[0, 1, 2], [0, 2, 3]]
    mixed = ss.triangulate_faces(np.array([4, 3, 5]), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    assert mixed.shape == (2 + 1 + 3, 3) and mixed.dtype == np.int32
    assert sorted(map(tuple, mixed.tolist())) == sorted(
        [(0, 1, 2), (0, 2, 3), (4, 5, 6), (7, 8, 9), (7, 9, 10), (7, 10, 11)]
    )
    assert ss.triangulate_faces(np.array([2]), np.array([0, 1])).shape == (0, 3)


def test_lidar_pattern_matches_the_sensor_cfg():
    dirs = ss.lidar_ray_dirs(TH)
    assert dirs.shape == (100 * 360, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert dirs[:, 2].min() == pytest.approx(math.sin(math.radians(-25.0)))
    assert dirs[:, 2].max() == pytest.approx(math.sin(math.radians(15.0)))


def test_next_cast_z_restarts_just_below_hits_and_parks_misses():
    z = np.array([2.0, np.nan, 0.5], dtype=np.float32)
    out = ss.next_cast_z(z, -5.0)
    assert out[0] == pytest.approx(2.0 - 0.002) and out[1] == -5.0 and out[2] == pytest.approx(0.498)
