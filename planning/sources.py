"""Pose and map sources: the two seams where H4c's SLAM plugs in.

Pure Python, no ROS, no Isaac. H4 runs with the ground-truth implementations; a SLAM-derived pose or map is a drop-in
replacement for the same two methods, and nothing downstream (planner, controller, metrics) changes.

``PoseSource.get_pose()`` returns ``(x, y, yaw)`` in the world frame, metres and radians.
``MapSource.get_grid()`` returns a :class:`planning.occupancy.OccupancyGrid` (free / blocked / unknown + inflation).

Documented in ``docs/interfaces.md``.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from planning.occupancy import OccupancyGrid

Pose = tuple[float, float, float]


@runtime_checkable
class PoseSource(Protocol):
    """Where the robot is. Ground truth now; SLAM later."""

    def get_pose(self) -> Pose: ...


@runtime_checkable
class MapSource(Protocol):
    """The occupancy grid the planner runs on. Ground truth now; SLAM/mapping later."""

    def get_grid(self) -> OccupancyGrid: ...


class GroundTruthPoseSource:
    """Reads the simulator's own base pose (H4: ground truth, not an estimate).

    ``pose_fn`` is any callable returning ``(x, y, yaw)``; the pod driver passes one that reads
    ``scene_env.read_poses(env)[env_index]``. Kept as a callable so this module stays import-free of torch.
    """

    def __init__(self, pose_fn: Callable[[], Pose]) -> None:
        self._pose_fn = pose_fn

    def get_pose(self) -> Pose:
        x, y, yaw = self._pose_fn()
        return (float(x), float(y), float(yaw))


class GroundTruthMapSource:
    """Holds the occupancy grid rasterised from the scene's static geometry (``planning/occupancy.py``)."""

    def __init__(self, grid: OccupancyGrid) -> None:
        self._grid = grid

    def get_grid(self) -> OccupancyGrid:
        return self._grid
