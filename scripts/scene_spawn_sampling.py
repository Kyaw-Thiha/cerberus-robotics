"""Spawn-pose selection for scene admission (numpy only, no Isaac).

Input is what ``scripts/validate_scenes.py`` measures on the pod: for every cell of an xy grid the stack of
surfaces a vertical ray meets going DOWN from above the scene (``hit_z`` descending, ``hit_nz`` the z of the unit
winding-based face normal of the hit triangle, NaN once the ray has nothing left to hit). From the stack this module decides which cells are floor
with a clear torso/head volume, which of those lie on the main floor level, how large the free area around each is,
and returns up to ``Thresholds.max_spawns`` spawn poses best-first.

Why cast against the scene's visual triangle mesh (a Warp BVH on the GPU, the same geometry the SLAM LiDAR rays hit)
rather than an Isaac Sim physics-scene raycast: it needs no physics scene, so a scene whose PhysX warmup stalls still
gets spawns and LiDAR verdicts (the stall is then recorded by the env phase), it is one batched GPU call per surface
layer on a 1-2 GB scene, and floor/clearance/LiDAR all use one mesh. The price is that a visual surface without a
collider looks like floor here; the env phase (zero-action stepping at the first spawn, z band around ``floor_z``)
catches exactly that. Known limitation: with no roofed floor anywhere, the top of a roof counts as open ground, so a
scene whose only candidate surfaces are roofs gets roof spawns that the env phase then rejects or the SLAM eval
would show as an exterior start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scene_admission_logic import Thresholds

_EPS_RAY = 0.002  # restart offset below a hit when the column is re-cast; thinner surfaces are skipped over
_NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
_YAW_DIRECTIONS = 16


@dataclass(frozen=True)
class GridSpec:
    x0: float
    y0: float
    cell: float
    nx: int
    ny: int

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.x0 + (np.arange(self.nx) + 0.5) * self.cell,
            self.y0 + (np.arange(self.ny) + 0.5) * self.cell,
        )

    def xy(self) -> np.ndarray:
        """Cell centres, shape (ny * nx, 2), row-major (index = iy * nx + ix)."""
        xs, ys = self.centers()
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], axis=1)


@dataclass(frozen=True)
class SpawnCandidate:
    x: float
    y: float
    yaw: float
    floor_z: float
    open_radius_m: float
    roofed: bool


@dataclass
class SpawnPlan:
    level_z: float | None
    spawns: list[SpawnCandidate]
    free_cells: int
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------------------------------------
def triangulate_faces(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate a USD polygon mesh (``faceVertexCounts`` / ``faceVertexIndices``) -> (M, 3) int32."""
    counts = np.asarray(counts)
    indices = np.asarray(indices)
    if (counts == 3).all():
        return indices.reshape(-1, 3).astype(np.int32)
    starts = np.cumsum(counts) - counts
    tris = []
    for k in np.unique(counts):
        if k < 3:
            continue
        s = starts[counts == k]
        for j in range(k - 2):
            tris.append(np.stack([indices[s], indices[s + 1 + j], indices[s + 2 + j]], axis=1))
    if not tris:
        return np.zeros((0, 3), dtype=np.int32)
    return np.concatenate(tris).astype(np.int32)


def robust_xy_bounds(points: np.ndarray, pad_m: float = 1.0, pct: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """xy footprint of a vertex cloud, ignoring stray far-away geometry (a huge ground plane has only a few
    vertices): the `pct` .. 100-`pct` percentiles (on a subsample) plus `pad_m`."""
    stride = max(1, len(points) // 2_000_000)
    xy = points[::stride, :2]
    lo = np.percentile(xy, pct, axis=0) - pad_m
    hi = np.percentile(xy, 100.0 - pct, axis=0) + pad_m
    return lo, hi


def plan_grid(lo: np.ndarray, hi: np.ndarray, th: Thresholds) -> GridSpec:
    """Grid over [lo, hi] with pitch ``th.grid_cell_m``, coarsened so there are at most ``th.max_grid_cells``."""
    w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"empty footprint {lo} .. {hi}")
    cell = max(th.grid_cell_m, math.sqrt(w * h / th.max_grid_cells))
    return GridSpec(float(lo[0]), float(lo[1]), cell, max(1, math.ceil(w / cell)), max(1, math.ceil(h / cell)))


def lidar_ray_dirs(th: Thresholds) -> np.ndarray:
    """Unit directions of one sweep of the SLAM LiDAR pattern, shape (channels * horizontal steps, 3)."""
    vert = np.radians(np.linspace(th.lidar_vfov_deg[0], th.lidar_vfov_deg[1], th.lidar_channels))
    horiz = np.radians(np.arange(-180.0, 180.0, th.lidar_hres_deg))
    v, h = np.meshgrid(vert, horiz, indexing="ij")
    return np.stack([np.cos(v) * np.cos(h), np.cos(v) * np.sin(h), np.sin(v)], axis=-1).reshape(-1, 3)


def next_cast_z(hit_z: np.ndarray, z_floor: float) -> np.ndarray:
    """Start heights for the next ray layer: just under each hit; columns that already missed start below the
    mesh (``z_floor``) so they keep missing."""
    return np.where(np.isnan(hit_z), z_floor, hit_z - _EPS_RAY)


# ---------------------------------------------------------------------------------------------------------
# floor / clearance / open area
# ---------------------------------------------------------------------------------------------------------
def floor_candidates(
    hit_z: np.ndarray, hit_nz: np.ndarray, th: Thresholds
) -> tuple[np.ndarray, np.ndarray]:
    """Per layer and column: the surface height if it can be floor, else NaN; and whether a roof/ceiling is the
    next surface above it. A surface is a floor candidate when it is flat and the free space up to the next surface
    above (ceiling underside, table underside, ...) is at least ``th.clear_height_m``."""
    k, n = hit_z.shape
    above = np.vstack([np.full((1, n), np.inf), hit_z[:-1]])
    with np.errstate(invalid="ignore"):
        clear = above - hit_z
        ok = ~np.isnan(hit_z) & (hit_nz >= th.min_floor_normal_z) & (clear >= th.clear_height_m)
        roofed = (np.arange(k)[:, None] > 0) & (clear <= th.max_ceiling_m)
    return np.where(ok, hit_z, np.nan), ok & roofed


def choose_level(cand_z: np.ndarray, roofed: np.ndarray, th: Thresholds) -> float | None:
    """Height of the main floor level: the most common candidate floor height among roofed candidates (all
    candidates if nothing is roofed), using ``th.level_tol_m`` bins smoothed over neighbours; among levels with at
    least half the best count the one closest to z = 0 wins (current backends drop the robot from z = 0.4)."""
    valid = ~np.isnan(cand_z)
    sel = valid & roofed if (valid & roofed).any() else valid
    if not sel.any():
        return None
    vals = cand_z[sel]
    bins = np.round(vals / th.level_tol_m).astype(np.int64)
    uniq, counts = np.unique(bins, return_counts=True)
    by_bin = dict(zip(uniq.tolist(), counts.tolist(), strict=True))
    smoothed = np.array([by_bin.get(b - 1, 0) + by_bin.get(b, 0) + by_bin.get(b + 1, 0) for b in uniq.tolist()])
    near_best = np.flatnonzero(smoothed >= 0.5 * smoothed.max())
    best = uniq[near_best[np.argmin(np.abs(uniq[near_best]))]]
    return float(np.median(vals[bins == best]))


def floor_on_level(
    cand_z: np.ndarray, roofed: np.ndarray, level: float, spec: GridSpec, th: Thresholds
) -> tuple[np.ndarray, np.ndarray]:
    """Per cell the candidate floor nearest `level` (NaN if none within ``th.level_tol_m``) and its roofed flag,
    both shaped (ny, nx)."""
    dist = np.abs(cand_z - level)
    dist = np.where(np.isnan(dist), np.inf, dist)
    pick = dist.argmin(axis=0)
    cols = np.arange(cand_z.shape[1])
    ok = dist[pick, cols] <= th.level_tol_m
    floor = np.where(ok, cand_z[pick, cols], np.nan).reshape(spec.ny, spec.nx)
    return floor, (ok & roofed[pick, cols]).reshape(spec.ny, spec.nx)


def _shift(a: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    """out[y, x] = a[y + dy, x + dx], `fill` outside the array."""
    out = np.full_like(a, fill)
    ny, nx = a.shape
    ys, yd = slice(max(0, -dy), min(ny, ny - dy)), slice(max(0, dy), min(ny, ny + dy))
    xs, xd = slice(max(0, -dx), min(nx, nx - dx)), slice(max(0, dx), min(nx, nx + dx))
    out[ys, xs] = a[yd, xd]
    return out


def free_cells(floor: np.ndarray, th: Thresholds) -> np.ndarray:
    """Cells with a floor that is not next to a step taller than ``th.max_step_m``."""
    blocked = np.zeros(floor.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for dy, dx in _NEIGHBOURS:
            blocked |= np.abs(floor - _shift(floor, dy, dx, np.nan)) > th.max_step_m
    return ~np.isnan(floor) & ~blocked


def free_half_width_m(free: np.ndarray, cell: float, max_radius_m: float) -> np.ndarray:
    """Half-width (m) of the free square around each cell: a cell that survives i erosions of the free mask has a
    (2i+1)-cell square free around it, so (i + 0.5) * cell. 0 for blocked cells; capped at `max_radius_m`."""
    radius = np.zeros(free.shape)
    current = free.copy()
    radius[current] = 0.5 * cell
    for i in range(1, math.ceil(max_radius_m / cell) + 1):
        eroded = current.copy()
        for dy, dx in _NEIGHBOURS:
            eroded &= _shift(current, dy, dx, False)
        if not eroded.any():
            break
        radius[eroded] = (i + 0.5) * cell
        current = eroded
    return np.minimum(radius, max_radius_m)


def best_yaw(free: np.ndarray, iy: int, ix: int, max_cells: int) -> float:
    """Heading (rad, 0 = +x, counter-clockwise) whose straight line from the cell stays free the longest."""
    best, best_run = 0.0, -1
    ny, nx = free.shape
    for d in range(_YAW_DIRECTIONS):
        yaw = 2.0 * math.pi * d / _YAW_DIRECTIONS
        run = 0
        for t in range(1, max_cells + 1):
            y, x = round(iy + t * math.sin(yaw)), round(ix + t * math.cos(yaw))
            if not (0 <= y < ny and 0 <= x < nx and free[y, x]):
                break
            run = t
        if run > best_run:
            best, best_run = yaw, run
    return best


def select_spawns(
    floor: np.ndarray, roofed: np.ndarray, free: np.ndarray, spec: GridSpec, th: Thresholds
) -> list[SpawnCandidate]:
    """Up to ``th.max_spawns`` spawns, best first: roofed before open-sky cells, then larger free radius; each at
    least ``th.spawn_separation_m`` from the ones already picked. Ties break by row then column (deterministic)."""
    radius = free_half_width_m(free, spec.cell, th.max_open_radius_m)
    ok = free & (radius >= th.min_open_radius_m)
    iy, ix = np.nonzero(ok)
    order = np.lexsort((ix, iy, -radius[iy, ix], ~roofed[iy, ix]))
    xs, ys = spec.centers()
    picked: list[SpawnCandidate] = []
    for i in order:
        y, x = int(iy[i]), int(ix[i])
        px, py = float(xs[x]), float(ys[y])
        if any(math.hypot(px - c.x, py - c.y) < th.spawn_separation_m for c in picked):
            continue
        picked.append(
            SpawnCandidate(
                x=px,
                y=py,
                yaw=best_yaw(free, y, x, math.ceil(th.max_open_radius_m / spec.cell)),
                floor_z=float(floor[y, x]),
                open_radius_m=float(radius[y, x]),
                roofed=bool(roofed[y, x]),
            )
        )
        if len(picked) == th.max_spawns:
            break
    return picked


def plan_spawns(hit_z: np.ndarray, hit_nz: np.ndarray, spec: GridSpec, th: Thresholds) -> SpawnPlan:
    """Hit stacks (layers x cells, see module docstring) -> main floor level, free cells and spawn poses."""
    cand_z, roofed = floor_candidates(hit_z, hit_nz, th)
    level = choose_level(cand_z, roofed, th)
    if level is None:
        return SpawnPlan(None, [], 0, ["no floor candidate with enough clear height"])
    floor, roofed_grid = floor_on_level(cand_z, roofed, level, spec, th)
    free = free_cells(floor, th)
    spawns = select_spawns(floor, roofed_grid, free, spec, th)
    notes = [] if spawns else [f"{int(free.sum())} free cells but none with open radius >= {th.min_open_radius_m} m"]
    if spawns and not any(s.roofed for s in spawns):
        notes.append("no roofed floor found; spawns are open-sky cells")
    return SpawnPlan(level, spawns, int(free.sum()), notes)
