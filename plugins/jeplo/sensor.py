"""JEPLO's LiDAR depth image, built in OUR scene from an Isaac Lab ray caster.

Geometry is the authors' (25 vertical rays over -4.5..45.5 deg, 60 horizontal over -60..60 deg, 2.0 m range,
sensor at (0.275, 0, 0.148) on the base, rotated (w,x,y,z)=(0,1,0,0) into the Livox frame). The implementation is
ours (this package is GPL-3.0; the core never imports it).

Per policy step: project the ray hits into a 25x60 spherical grid (minimum range per bin), accumulate 10 frames and
publish the per-bin minimum every 5 steps (10 Hz), normalised by the 2.0 m range. Gaussian noise and the bottom-row
dropout come from the authors' training setup; the training-only random occlusion masks are NOT added (the released
deployment does not have them, and inventing obstacles would be dishonest perception).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

V, H = 25, 60
V_FOV = (-4.5, 45.5)
H_FOV = (-60.0, 60.0)
MAX_RANGE_M = 2.0
NOISE_STD_M = 0.005
BOTTOM_DROPOUT_FRACTION = 0.05
HISTORY = 10
UPDATE_EVERY = 5


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class SphericalDepth:
    """Accumulate LiDAR hits into the 25x60 spherical depth image JEPLO consumes."""

    def __init__(self, *, max_range_m: float = MAX_RANGE_M, noise_std_m: float = NOISE_STD_M,
                 bottom_dropout_fraction: float = BOTTOM_DROPOUT_FRACTION, history: int = HISTORY,
                 update_every: int = UPDATE_EVERY, seed: int | None = None) -> None:
        self.max_range_m = float(max_range_m)
        self.noise_std_m = float(noise_std_m)
        self.bottom_dropout_fraction = float(bottom_dropout_fraction)
        self.update_every = int(update_every)
        self._hist: deque[np.ndarray] = deque(maxlen=int(history))
        self._depth = np.ones((V, H), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._frames = 0
        self._theta_low = math.radians(V_FOV[0])
        self._theta_res = math.radians(V_FOV[1] - V_FOV[0]) / V
        self._phi_low = math.radians(H_FOV[0])
        self._phi_res = math.radians(H_FOV[1] - H_FOV[0]) / H

    def reset(self) -> None:
        self._hist.clear()
        self._depth = np.ones((V, H), dtype=np.float32)

    def project(self, hits_w: np.ndarray, sensor_pos, sensor_quat_wxyz) -> np.ndarray:
        """One frame: (rays, 3) world hits -> (25, 60) range in metres, unfilled = max range."""
        hits = np.asarray(hits_w, dtype=np.float64).reshape(-1, 3)
        origin = np.asarray(sensor_pos, dtype=np.float64).reshape(3)
        rot = quat_wxyz_to_matrix(sensor_quat_wxyz)
        local = (hits - origin) @ rot  # R^T (p - o) == (p - o) @ R
        r = np.linalg.norm(local, axis=1)
        valid = np.isfinite(r) & (r > 0.1) & (r <= self.max_range_m)
        grid = np.full((V, H), self.max_range_m, dtype=np.float64)
        if not valid.any():
            return grid
        pts = local[valid]
        rr = r[valid]
        theta = np.arcsin(np.clip(pts[:, 2] / rr, -1.0, 1.0))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        ti = np.rint((theta - self._theta_low) / self._theta_res).astype(int)
        pi = np.rint((phi - self._phi_low) / self._phi_res).astype(int)
        ok = (ti >= 0) & (ti < V) & (pi >= 0) & (pi < H)
        np.minimum.at(grid, (ti[ok], pi[ok]), rr[ok])
        return grid

    def project_rays(self, hits_w, sensor_pos) -> np.ndarray:
        """Ranges in RAY ORDER (V x H), which is exact for a grid ray caster: no frame conventions involved.

        The ray caster returns one hit per pattern ray, and our pattern is (25, 60) in row-major order, so the
        depth image is the per-ray range from the sensor origin. Unfilled/far rays become max range.
        """
        hits = np.asarray(hits_w, dtype=np.float64).reshape(-1, 3)
        origin = np.asarray(sensor_pos, dtype=np.float64).reshape(3)
        r = np.linalg.norm(hits - origin, axis=1)
        r = np.where(np.isfinite(r) & (r > 0.1) & (r <= self.max_range_m), r, self.max_range_m)
        return r.reshape(V, H)

    def update_rays(self, hits_w, sensor_pos) -> np.ndarray | None:
        """Per-ray depth update, same accumulation/noise/dropout as :meth:`update`."""
        return self._publish(self.project_rays(hits_w, sensor_pos))

    def update(self, hits_w, sensor_pos, sensor_quat_wxyz) -> np.ndarray | None:
        """Push one frame; every ``update_every`` calls publish a new normalised depth image (else ``None``)."""
        return self._publish(self.project(hits_w, sensor_pos, sensor_quat_wxyz))

    def _publish(self, frame: np.ndarray) -> np.ndarray | None:
        frame = frame + self._rng.normal(0.0, self.noise_std_m, size=frame.shape)
        frame = np.clip(frame, 0.0, self.max_range_m)
        self._hist.append(frame)
        self._frames += 1
        if len(self._hist) < self._hist.maxlen:
            return None
        if (self._frames - self._hist.maxlen) % self.update_every != 0:
            return None
        aggregated = np.min(np.stack(list(self._hist), axis=0), axis=0)
        rows = max(1, math.ceil(V * self.bottom_dropout_fraction))
        drop = self._rng.random((rows, H)) < 0.5
        aggregated[:rows] = np.where(drop, self.max_range_m, aggregated[:rows])
        self._depth = (aggregated / self.max_range_m).astype(np.float32)
        return self._depth

    @property
    def depth(self) -> np.ndarray:
        """Latest normalised image, always (25, 60) float32 in [0, 1]."""
        return self._depth.copy()
