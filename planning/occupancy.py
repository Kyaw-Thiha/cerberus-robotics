"""Occupancy grid from a scene's static geometry: the ground-truth map the H4 planner runs on.

No Isaac, no ROS: ``pxr`` is imported inside :func:`scene_triangles_m` and numpy at module level. This is the
"ground-truth-derived occupancy map" the H4 report must name as such -- it is rasterised from the scene's own
meshes, not from a sensor.

Pipeline: world triangles (metres) -> surface samples -> per-cell floor and clearance band -> free / blocked /
unknown -> inflation by the robot radius. Parameters live in :class:`OccupancyConfig` (defaults checked in as
``planning/config/occupancy.yaml``), never as magic numbers at call sites.

Single-floor assumption: the floor level is the dominant up-facing surface height (or an explicit ``floor_z``), and
a cell's local floor must be within ``floor_window_m`` of it. An upper floor is therefore "unknown", not a second
navigable level; multi-floor navigation is out of scope (H4 says single-floor selection is acceptable if
documented). Floor height changes WITHIN the level (steps, ramps) are handled: a neighbour whose floor differs by
more than ``max_step_m`` blocks both cells.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

FREE = np.int8(0)
BLOCKED = np.int8(1)
UNKNOWN = np.int8(-1)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "occupancy.yaml"


@dataclass(frozen=True)
class OccupancyConfig:
    """Every number the grid depends on. Change these in the YAML, not in the algorithms."""

    resolution_m: float = 0.05
    band_low_m: float = 0.12
    band_high_m: float = 0.45
    floor_window_m: float = 0.20
    max_step_m: float = 0.12
    robot_radius_m: float = 0.25
    sample_spacing_m: float = 0.015
    max_samples: int = 60_000_000

    def __post_init__(self) -> None:
        if self.resolution_m <= 0 or self.sample_spacing_m <= 0:
            raise ValueError("resolution_m and sample_spacing_m must be positive")
        if not (0 <= self.band_low_m < self.band_high_m):
            raise ValueError(f"need 0 <= band_low < band_high, got {self.band_low_m}..{self.band_high_m}")
        if self.robot_radius_m < 0:
            raise ValueError("robot_radius_m must be >= 0")

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> OccupancyConfig:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OccupancyGrid:
    """A 2D grid with world-frame origin at the centre of cell (0, 0)."""

    grid: np.ndarray  # int8, shape (ny, nx); FREE / BLOCKED / UNKNOWN
    origin_xy: tuple[float, float]
    resolution_m: float
    floor_z: float
    config: OccupancyConfig
    scene_id: str = ""
    inflated: np.ndarray | None = None  # bool, blocked-after-inflation (same shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.floor((x - self.origin_xy[0]) / self.resolution_m))
        iy = int(np.floor((y - self.origin_xy[1]) / self.resolution_m))
        return iy, ix

    def centre_of(self, iy: int, ix: int) -> tuple[float, float]:
        return (
            self.origin_xy[0] + (ix + 0.5) * self.resolution_m,
            self.origin_xy[1] + (iy + 0.5) * self.resolution_m,
        )

    def inside(self, iy: int, ix: int) -> bool:
        ny, nx = self.grid.shape
        return 0 <= iy < ny and 0 <= ix < nx

    def is_free(self, x: float, y: float, *, inflated: bool = True) -> bool:
        iy, ix = self.cell_of(x, y)
        if not self.inside(iy, ix):
            return False
        if inflated and self.inflated is not None:
            return bool(self.grid[iy, ix] == FREE and not self.inflated[iy, ix])
        return bool(self.grid[iy, ix] == FREE)

    def free_fraction(self) -> float:
        known = self.grid != UNKNOWN
        return float((self.grid == FREE).sum() / max(int(known.sum()), 1))

    def extent(self) -> tuple[float, float, float, float]:
        ny, nx = self.grid.shape
        return (
            self.origin_xy[0],
            self.origin_xy[1],
            self.origin_xy[0] + nx * self.resolution_m,
            self.origin_xy[1] + ny * self.resolution_m,
        )


# ---------------------------------------------------------------------------------------------------------
# geometry -> samples -> grid
# ---------------------------------------------------------------------------------------------------------
def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Areas of the triangles in ``faces`` (metres^2)."""
    tri = np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def sample_surfaces(
    points: np.ndarray, faces: np.ndarray, config: OccupancyConfig, spacing_m: float | None = None
) -> dict[str, np.ndarray]:
    """Surface samples of ``faces`` with the orientation each came from (up-facing / vertical / slanted).

    Deterministic low-discrepancy barycentric sampling: no RNG, so two runs of the same scene produce the same grid.
    The sample count per triangle follows its area, so a floor slab is covered at roughly ``sample_spacing_m``.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = points[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    normal = np.cross(e1, e2)
    norm = np.linalg.norm(normal, axis=1)
    area = 0.5 * norm
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = normal[:, 2] / norm
    up = nz >= 0.9          # floor, table tops, step treads
    down = nz <= -0.9       # ceilings, box undersides, overhangs
    vertical = np.abs(nz) <= 0.5
    slanted = ~(up | down | vertical)  # stair noses, ramps, sills: obstacles, never floor

    spacing = float(spacing_m if spacing_m is not None else config.sample_spacing_m)
    per_tri = np.clip(np.ceil(area / (spacing ** 2)), 1, 2_000_000).astype(np.int64)
    total = int(per_tri.sum())

    tri_index = np.repeat(np.arange(len(faces)), per_tri)
    offsets = np.repeat(np.cumsum(per_tri) - per_tri, per_tri)
    j = np.arange(total) - offsets
    u = np.mod(j * 0.7548776662466927, 1.0)  # R2 low-discrepancy sequence
    v = np.mod(j * 0.5698402909980532, 1.0)
    flip = u + v > 1.0
    u = np.where(flip, 1.0 - u, u)
    v = np.where(flip, 1.0 - v, v)
    base = tri[tri_index, 0]
    sampled = base + u[:, None] * e1[tri_index] + v[:, None] * e2[tri_index]
    return {
        "xyz": sampled,
        "up": up[tri_index],
        "down": down[tri_index],
        "vertical": vertical[tri_index],
        "slanted": slanted[tri_index],
    }


def build_grid(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    config: OccupancyConfig | None = None,
    scene_id: str = "",
    floor_z: float | None = None,
    chunk_faces: int = 500_000,
) -> OccupancyGrid:
    """Rasterise world triangles (metres) into a FREE / BLOCKED / UNKNOWN grid, then inflate.

    Triangles are processed in chunks and aggregated per cell, so a 20M-triangle house never materialises all its
    surface samples at once.
    """
    config = config or OccupancyConfig()
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(points) == 0 or len(faces) == 0:
        raise ValueError("no geometry")

    # Origin and size from the 0.1-99.9 percentile of the vertices, so one stray plane cannot blow the grid up.
    lo = np.percentile(points[:, :2], 0.1, axis=0) - config.resolution_m
    hi = np.percentile(points[:, :2], 99.9, axis=0) + config.resolution_m
    ny = int(np.ceil((hi[1] - lo[1]) / config.resolution_m)) + 1
    nx = int(np.ceil((hi[0] - lo[0]) / config.resolution_m)) + 1
    origin = np.array([lo[0], lo[1]])
    n_cells = ny * nx

    # One global spacing for every chunk: a big floor slab needs hundreds of thousands of samples, and scaling per
    # triangle would leave most of its cells unsampled. If the scene's total area would exceed max_samples, widen the
    # spacing uniformly instead of dropping coverage unevenly.
    total_area = 0.0
    for start in range(0, len(faces), chunk_faces):
        total_area += float(triangle_areas(points, faces[start:start + chunk_faces]).sum())
    spacing = config.sample_spacing_m
    if total_area > 0.0:
        spacing = max(spacing, math.sqrt(total_area / config.max_samples))

    up_min = np.full(n_cells, np.inf)
    up_max = np.full(n_cells, -np.inf)
    vert_min = np.full(n_cells, np.inf)
    vert_max = np.full(n_cells, -np.inf)
    slant_min = np.full(n_cells, np.inf)
    slant_max = np.full(n_cells, -np.inf)
    down_min = np.full(n_cells, np.inf)
    down_max = np.full(n_cells, -np.inf)

    for start in range(0, len(faces), chunk_faces):
        samples = sample_surfaces(points, faces[start:start + chunk_faces], config, spacing_m=spacing)
        xyz = samples["xyz"]
        idx = np.floor((xyz[:, :2] - origin) / config.resolution_m).astype(np.int64)
        iy, ix = idx[:, 1], idx[:, 0]
        inside = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        if not inside.any():
            continue
        iy, ix, xyz = iy[inside], ix[inside], xyz[inside]
        up = samples["up"][inside]
        down = samples["down"][inside]
        vertical = samples["vertical"][inside]
        slanted = samples["slanted"][inside]
        flat = iy * nx + ix
        z = xyz[:, 2]
        np.minimum.at(up_min, flat[up], z[up])
        np.maximum.at(up_max, flat[up], z[up])
        np.minimum.at(vert_min, flat[vertical], z[vertical])
        np.maximum.at(vert_max, flat[vertical], z[vertical])
        np.minimum.at(slant_min, flat[slanted], z[slanted])
        np.maximum.at(slant_max, flat[slanted], z[slanted])
        np.minimum.at(down_min, flat[down], z[down])
        np.maximum.at(down_max, flat[down], z[down])

    level = estimate_floor_level(up_min[np.isfinite(up_min)], floor_z)
    has_up = np.isfinite(up_min)
    local_floor = np.where(has_up & (np.abs(up_min - level) <= config.floor_window_m), up_min, np.nan)
    # Sampling can leave a cell without a hit even on a continuous slab; fill those holes from their neighbours so
    # "unknown" means "no geometry here", not "the sampler missed".
    local_floor = _fill_floor_holes(local_floor.reshape(ny, nx)).reshape(-1)

    grid = np.full(n_cells, UNKNOWN, dtype=np.int8)
    free = np.isfinite(local_floor)
    band_lo = local_floor + config.band_low_m
    band_hi = local_floor + config.band_high_m
    blocked = (vert_min <= band_hi) & (vert_max >= band_lo)          # walls, legs, frames crossing the band
    blocked |= (up_max > band_lo) & (up_max < band_hi)               # a step/kerb surface inside the band
    blocked |= (slant_min <= band_hi) & (slant_max >= band_lo)       # slanted surfaces entering the band
    blocked |= (down_max > band_lo) & (down_min < band_hi)           # an overhang inside the band
    blocked &= free
    grid[free] = FREE
    grid[blocked] = BLOCKED

    grid2 = grid.reshape(ny, nx)
    # A surface thinner than a cell can leave an unblocked cell between its two face rows (a 0.1 m wall at 5 cm
    # cells is exactly that), and a planner would thread a 1-cell tunnel through the wall. One 4-neighbour
    # thickening closes it; it also grows furniture by 5 cm, which the inflation already covers.
    grid2[_thicken(grid2 == BLOCKED)] = BLOCKED
    grid2 = _block_steep_neighbours(grid2, local_floor.reshape(ny, nx), config.max_step_m)

    out = OccupancyGrid(
        grid=grid2,
        origin_xy=(float(origin[0]), float(origin[1])),
        resolution_m=float(config.resolution_m),
        floor_z=float(level),
        config=config,
        scene_id=scene_id,
    )
    out.inflated = inflate(out.grid, config.robot_radius_m, config.resolution_m)
    return out


def estimate_floor_level(up_heights: np.ndarray, floor_z: float | None = None) -> float:
    """The dominant up-facing surface height (the floor), unless the caller knows it."""
    if floor_z is not None:
        return float(floor_z)
    if len(up_heights) == 0:
        raise ValueError("no up-facing geometry: cannot estimate a floor level")
    bins = np.round(up_heights / 0.02) * 0.02  # 2 cm histogram, ties resolved toward the lower height
    values, counts = np.unique(bins, return_counts=True)
    return float(values[int(np.argmax(counts))])


def _fill_floor_holes(floor: np.ndarray, min_neighbours: int = 4) -> np.ndarray:
    """Fill a NaN floor cell from the median of its finite 8-neighbours when at least ``min_neighbours`` agree.

    Sampling a continuous slab can still miss a cell; without this, those cells would read UNKNOWN and the planner
    would treat a solid floor as a hole. NaNs are preserved through the shifts, so a floor at exactly 0.0 is not
    confused with "no sample".
    """
    out = floor.copy()
    nan = np.isnan(out)
    if not nan.any():
        return out
    stacked, counts = [], np.zeros_like(out, dtype=int)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy, dx) == (0, 0):
                continue
            shifted = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            stacked.append(shifted)
            counts += ~np.isnan(shifted)
    fill = nan & (counts >= min_neighbours)
    if fill.any():
        median = np.nanmedian(np.stack(stacked)[:, fill], axis=0)
        out[fill] = median
    return out


def _thicken(blocked: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Grow a blocked mask by ``iterations`` cells in the 4-neighbourhood (closes sub-cell-thick surfaces)."""
    out = blocked.copy()
    for _ in range(iterations):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _block_steep_neighbours(grid: np.ndarray, local_floor: np.ndarray, max_step_m: float) -> np.ndarray:
    """Block both cells of any 4-neighbour pair whose floor heights differ by more than ``max_step_m``."""
    out = grid.copy()
    for dy, dx in ((0, 1), (1, 0)):
        a = local_floor[: local_floor.shape[0] - dy, : local_floor.shape[1] - dx]
        b = local_floor[dy:, dx:]
        steep = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > max_step_m)
        head = out[: out.shape[0] - dy, : out.shape[1] - dx]
        tail = out[dy:, dx:]
        head[steep] = BLOCKED
        tail[steep] = BLOCKED
    return out


def inflate(grid: np.ndarray, radius_m: float, resolution_m: float) -> np.ndarray:
    """True where a cell is within ``radius_m`` of a blocked cell (circle kernel, includes the cell itself)."""
    blocked = grid == BLOCKED
    if radius_m <= 0:
        return blocked.copy()
    r_cells = int(np.ceil(radius_m / resolution_m))
    ny, nx = grid.shape
    out = blocked.copy()
    for dy in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            if dy * dy + dx * dx > r_cells * r_cells or (dy == 0 and dx == 0):
                continue
            ys_src = slice(max(0, -dy), ny - max(0, dy))
            xs_src = slice(max(0, -dx), nx - max(0, dx))
            ys_dst = slice(max(0, dy), ny - max(0, -dy))
            xs_dst = slice(max(0, dx), nx - max(0, -dx))
            out[ys_dst, xs_dst] |= blocked[ys_src, xs_src]
    return out


# ---------------------------------------------------------------------------------------------------------
# USD input and file I/O
# ---------------------------------------------------------------------------------------------------------
def scene_triangles_m(usd_path: str | Path, *, budget_s: float = 1800.0) -> tuple[np.ndarray, np.ndarray]:
    """World triangles of the scene USD in METRES (converts by the stage's ``metersPerUnit``)."""
    from pxr import Usd, UsdGeom

    from common.cerberus_worlds.scene_geometry import gather_world_triangles

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"cannot open {usd_path}")
    if str(UsdGeom.GetStageUpAxis(stage)) != "Z":
        raise ValueError(f"scene stage is up_axis={UsdGeom.GetStageUpAxis(stage)!r}; occupancy needs Z-up")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    points, faces = gather_world_triangles(stage, "/", frozenset(), budget_s)
    return np.asarray(points, dtype=np.float64) * mpu, faces


def from_usd(usd_path: str | Path, *, scene_id: str = "", config: OccupancyConfig | None = None,
             floor_z: float | None = None) -> OccupancyGrid:
    points, faces = scene_triangles_m(usd_path)
    return build_grid(points, faces, config=config, scene_id=scene_id or Path(usd_path).parent.name, floor_z=floor_z)


def save_npz(grid: OccupancyGrid, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": grid.grid,
        "inflated": grid.inflated if grid.inflated is not None else np.zeros_like(grid.grid, dtype=bool),
        "origin_xy": np.array(grid.origin_xy, dtype=np.float64),
        "resolution_m": np.float64(grid.resolution_m),
        "floor_z": np.float64(grid.floor_z),
        "scene_id": np.array(grid.scene_id),
        "config_json": np.array(json.dumps(grid.config.to_dict())),
    }
    np.savez_compressed(path, **payload)
    return path


def load_npz(path: str | Path) -> OccupancyGrid:
    with np.load(Path(path), allow_pickle=False) as data:
        config = OccupancyConfig(**json.loads(str(data["config_json"])))
        return OccupancyGrid(
            grid=np.array(data["grid"], dtype=np.int8),
            origin_xy=tuple(float(v) for v in data["origin_xy"]),
            resolution_m=float(data["resolution_m"]),
            floor_z=float(data["floor_z"]),
            config=config,
            scene_id=str(data["scene_id"]),
            inflated=np.array(data["inflated"], dtype=bool),
        )


def save_preview_png(grid: OccupancyGrid, path: str | Path, *, title: str | None = None) -> Path:
    """Small PNG: free white, blocked black, unknown grey, inflated zone hatched red."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((*grid.grid.shape, 3), dtype=float)
    rgb[grid.grid == FREE] = (1.0, 1.0, 1.0)
    rgb[grid.grid == BLOCKED] = (0.1, 0.1, 0.1)
    rgb[grid.grid == UNKNOWN] = (0.6, 0.6, 0.6)
    if grid.inflated is not None:
        mask = grid.inflated & (grid.grid == FREE)
        rgb[mask] = (0.95, 0.75, 0.75)
    fig, ax = plt.subplots(figsize=(6, 6 * grid.grid.shape[0] / max(grid.grid.shape[1], 1)), dpi=110)
    ax.imshow(rgb, origin="lower",
              extent=(grid.origin_xy[0], grid.origin_xy[0] + grid.grid.shape[1] * grid.resolution_m,
                      grid.origin_xy[1], grid.origin_xy[1] + grid.grid.shape[0] * grid.resolution_m))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"{grid.scene_id}: free {grid.free_fraction():.0%} (red = inflated)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
