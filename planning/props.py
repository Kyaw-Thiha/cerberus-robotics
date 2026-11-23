"""Prop set for the household mission: the pick table, the cube on it, and the place surface.

Both props are axis-aligned boxes in the scene's world frame, so the planner can take their (inflated) footprint
and a mission can be planned entirely on the occupancy grid. The cube is a state machine (on_table / in_gripper /
on_place / dropped); it is the only thing the mission's success criterion looks at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from planning.skills import CUBE_SIZE_M

TABLE_TOP_M = 0.35
TABLE_FOOTPRINT_M = 0.60
PLACE_SURFACE_TOP_M = 0.35
PLACE_SURFACE_FOOTPRINT_M = 0.60
INFLATE_M = 0.10  # the planner sees props slightly larger than they are: the base is wider than a point


@dataclass
class PropSurface:
    name: str
    centre_xy: tuple[float, float]
    footprint_m: float
    top_z: float
    yaw_rad: float = 0.0

    def corners(self) -> list[tuple[float, float]]:
        half = self.footprint_m / 2.0
        c, s = np.cos(self.yaw_rad), np.sin(self.yaw_rad)
        pts = []
        for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)):
            pts.append((self.centre_xy[0] + c * dx - s * dy, self.centre_xy[1] + s * dx + c * dy))
        return pts

    def footprint_cells(self, origin_xy: tuple[float, float], resolution_m: float,
                        inflate_m: float = INFLATE_M) -> list[tuple[int, int]]:
        """Cells covered by the (inflated) footprint, as ``(iy, ix)``."""
        half = self.footprint_m / 2.0 + inflate_m + CUBE_SIZE_M
        x0, x1 = self.centre_xy[0] - half, self.centre_xy[0] + half
        y0, y1 = self.centre_xy[1] - half, self.centre_xy[1] + half
        cells = []
        ix0 = int(np.floor((x0 - origin_xy[0]) / resolution_m))
        ix1 = int(np.ceil((x1 - origin_xy[0]) / resolution_m))
        iy0 = int(np.floor((y0 - origin_xy[1]) / resolution_m))
        iy1 = int(np.ceil((y1 - origin_xy[1]) / resolution_m))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                cells.append((iy, ix))
        return cells

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "centre_xy": list(self.centre_xy), "footprint_m": self.footprint_m,
                "top_z": self.top_z, "yaw_rad": self.yaw_rad, "corners": self.corners()}


@dataclass
class CubeState:
    pos: tuple[float, float, float]
    attached: bool = False
    on_surface: str | None = "pick_table"
    history: list[str] = field(default_factory=list)

    def move_to(self, pos: tuple[float, float, float], *, surface: str | None) -> None:
        self.pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        self.on_surface = surface
        self.history.append(f"->{surface or 'air'}")

    def as_dict(self) -> dict[str, Any]:
        return {"pos": list(self.pos), "attached": self.attached, "on_surface": self.on_surface,
                "history": list(self.history)}


@dataclass
class MissionProps:
    """The two surfaces plus the cube; ``approach_xy`` is where the robot stops to face the pick table."""

    pick_table: PropSurface
    place_surface: PropSurface
    cube_xy: tuple[float, float]
    approach_xy: tuple[float, float]
    approach_yaw: float = 0.0
    place_approach_xy: tuple[float, float] = (0.0, 0.0)
    place_approach_yaw: float = 0.0

    def cube(self) -> CubeState:
        return CubeState(pos=(self.cube_xy[0], self.cube_xy[1], self.pick_table.top_z + CUBE_SIZE_M / 2.0))

    def as_dict(self) -> dict[str, Any]:
        return {"pick_table": self.pick_table.as_dict(), "place_surface": self.place_surface.as_dict(),
                "cube_xy": list(self.cube_xy), "approach_xy": list(self.approach_xy),
                "approach_yaw": self.approach_yaw, "place_approach_xy": list(self.place_approach_xy),
                "place_approach_yaw": self.place_approach_yaw}


def approach_pose(table: PropSurface, *, distance_m: float = 0.6, facing_deg: float = 0.0) -> tuple[float, float, float]:
    """A pose ``distance_m`` from the table edge, facing its centre (x, y, yaw)."""
    edge = table.footprint_m / 2.0
    ang = np.radians(facing_deg)
    x = table.centre_xy[0] + (edge + distance_m) * np.cos(ang)
    y = table.centre_xy[1] + (edge + distance_m) * np.sin(ang)
    yaw = float(np.arctan2(table.centre_xy[1] - y, table.centre_xy[0] - x))
    return float(x), float(y), yaw


def free_approach(grid, surface: PropSurface, *, yaw_rad: float | None = None, min_m: float = 0.46,
                  max_m: float = 0.60, step_m: float = 0.02, n_angles: int = 24) -> tuple[float, float, float] | None:
    """A free base pose inside the arm's reach, looking at the surface centre.

    The table's inflated footprint blocks up to ~0.44 m from its centre and the arm reaches 0.62 m, so the usable
    standoff is a narrow band. When ``yaw_rad`` is given only that direction is tried; otherwise the ring is sampled
    (a table in a corner has one usable side, and the mission should find it rather than fail).
    """
    angles = [yaw_rad] if yaw_rad is not None else [2 * np.pi * i / n_angles for i in range(n_angles)]
    best: tuple[float, float, float] | None = None
    best_score = -1.0
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        distance = min_m
        while distance <= max_m:
            x = surface.centre_xy[0] + distance * dx
            y = surface.centre_xy[1] + distance * dy
            if grid.is_free(x, y, inflated=True):
                facing = float(np.arctan2(surface.centre_xy[1] - y, surface.centre_xy[0] - x))
                # prefer a pose with free floor around it: that is what makes the approach navigable
                # close is better (the arm reaches 0.62 m); free floor around it decides ties
                score = (sum(grid.is_free(x + 0.15 * np.cos(a), y + 0.15 * np.sin(a), inflated=True)
                             for a in np.linspace(0, 2 * np.pi, 8, endpoint=False))
                         - 20.0 * (distance - min_m))
                if score > best_score:
                    best_score, best = score, (float(x), float(y), facing)
                break
            distance += step_m
        if yaw_rad is not None and best is not None:
            break
    return best


def free_start_behind(grid, approach: tuple[float, float], yaw_rad: float, *, min_m: float = 0.6,
                      max_m: float = 3.0, step_m: float = 0.1) -> tuple[float, float]:
    """A free base pose ``min..max`` metres behind ``approach`` (opposite ``yaw_rad``); the approach if none is free."""
    distance = max_m
    while distance >= min_m:
        x = approach[0] - distance * np.cos(yaw_rad)
        y = approach[1] - distance * np.sin(yaw_rad)
        if grid.is_free(x, y, inflated=True):
            return float(x), float(y)
        distance -= step_m
    return float(approach[0]), float(approach[1])


def build_props(pick_centre: tuple[float, float], place_centre: tuple[float, float]) -> MissionProps:
    """Two surfaces several metres apart, the cube on the pick table, an approach waypoint facing it."""
    table = PropSurface("pick_table", pick_centre, TABLE_FOOTPRINT_M, TABLE_TOP_M)
    place = PropSurface("place_surface", place_centre, PLACE_SURFACE_FOOTPRINT_M, PLACE_SURFACE_TOP_M)
    ax, ay, yaw = approach_pose(table, distance_m=0.20)
    px, py, pyaw = approach_pose(place, distance_m=0.20)
    return MissionProps(pick_table=table, place_surface=place, cube_xy=pick_centre,
                        approach_xy=(ax, ay), approach_yaw=yaw, place_approach_xy=(px, py),
                        place_approach_yaw=pyaw)


def block_grid(grid, props: MissionProps, *, inflate_m: float = INFLATE_M):
    """Return a copy of ``grid`` with both prop footprints marked blocked (and inflated)."""
    import copy

    blocked = copy.deepcopy(grid)
    for surface in (props.pick_table, props.place_surface):
        for iy, ix in surface.footprint_cells(blocked.origin_xy, blocked.resolution_m, inflate_m):
            if blocked.inside(iy, ix):
                from planning.occupancy import BLOCKED

                blocked.grid[iy, ix] = BLOCKED
                if blocked.inflated is not None:
                    blocked.inflated[iy, ix] = True
    return blocked
