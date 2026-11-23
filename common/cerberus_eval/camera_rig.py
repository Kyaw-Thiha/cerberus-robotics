"""Camera rig for the scene: an EGO camera rigidly on the base, a yaw-only CHASE camera, a fixed OVERVIEW camera.

Pure geometry (poses, pose logging) is separated from the Isaac side (prim creation, capture) so the parts that can
be wrong silently -- the camera-to-base transform and the pose log -- are testable without a GPU.

Conventions
-----------
* the robot base frame is the Go2 body frame: +X forward, +Y left, +Z up;
* a camera looks along its own -Z with +Y up, so its orientation is built from the desired forward and up vectors
  rather than from a hand-rolled Euler chain (the source of most "the camera points the wrong way" bugs);
* every frame logs the base and camera world poses (``cerberus.poses.v1``), which is what ``scripts/host/clip_qa.py``
  checks: EGO/CHASE rigid within 2 cm / 2 degrees, OVERVIEW fixed, and the robot visible in CHASE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

POSE_SCHEMA = "cerberus.poses.v1"


# ---- pure geometry ---------------------------------------------------------------------------------------------

def rotation_from_forward(forward: np.ndarray, world_up: np.ndarray | None = None) -> np.ndarray:
    """3x3 rotation whose columns are the camera axes (X right, Y up, Z backward)."""
    up = np.array([0.0, 0.0, 1.0]) if world_up is None else np.asarray(world_up, dtype=float)
    f = np.asarray(forward, dtype=float)
    f = f / (np.linalg.norm(f) or 1.0)
    if abs(float(np.dot(f, up))) > 0.999:  # looking straight up/down: pick another reference
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(f, up)
    right = right / (np.linalg.norm(right) or 1.0)
    true_up = np.cross(right, f)
    return np.column_stack([right, true_up, -f])


def forward_from_yaw_pitch(yaw_rad: float, pitch_down_rad: float) -> np.ndarray:
    """Unit forward vector for a yaw (about +Z, 0 = +X) and a downward pitch."""
    return np.array([math.cos(pitch_down_rad) * math.cos(yaw_rad),
                     math.cos(pitch_down_rad) * math.sin(yaw_rad),
                     -math.sin(pitch_down_rad)])


def quat_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) from a rotation matrix."""
    m = np.asarray(R, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    # normalise and canonicalise the sign (w >= 0) so two logs of the same pose compare equal
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return (w, x, y, z)


def quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, ...]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def yaw_of(quat_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def chase_pose(base_pos: np.ndarray, base_yaw: float, *, distance: float = 1.6, height: float = 0.9,
               pitch_down_deg: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """Eye behind the base along its yaw, looking forward and down: world position + rotation matrix."""
    fwd = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])
    eye = np.asarray(base_pos, dtype=float) + np.array([0.0, 0.0, height]) - distance * fwd
    look = forward_from_yaw_pitch(base_yaw, math.radians(pitch_down_deg))
    return eye, rotation_from_forward(look)


@dataclass
class PoseLog:
    """Per-frame world poses of the base and every camera, in the schema clip_qa reads."""

    camera: str
    width: int
    height: int
    hfov_deg: float
    frames: list[dict[str, Any]] = field(default_factory=list)

    def append(self, *, t: float, base_pos, base_quat_wxyz, cam_pos, cam_quat_wxyz) -> None:
        self.frames.append({
            "t": round(float(t), 4),
            "base_pos": [round(float(v), 5) for v in base_pos],
            # stored xyzw, the convention pxr and the QA gate use
            "base_quat_xyzw": [round(float(v), 6) for v in (base_quat_wxyz[1], base_quat_wxyz[2],
                                                            base_quat_wxyz[3], base_quat_wxyz[0])],
            "cam_pos": [round(float(v), 5) for v in cam_pos],
            "cam_quat_xyzw": [round(float(v), 6) for v in (cam_quat_wxyz[1], cam_quat_wxyz[2],
                                                           cam_quat_wxyz[3], cam_quat_wxyz[0])],
        })

    def payload(self) -> dict[str, Any]:
        return {"schema": POSE_SCHEMA, "camera": self.camera, "width": self.width, "height": self.height,
                "hfov_deg": self.hfov_deg, "frames": self.frames}

    def write(self, path: Path | str) -> Path:
        from common.cerberus_eval.atomic_io import write_json_atomic

        return write_json_atomic(path, self.payload())


# ---- Isaac side ------------------------------------------------------------------------------------------------

@dataclass
class CameraSpec:
    name: str
    kind: str                     # "ego" | "chase" | "overview" | "closeup"
    resolution: tuple[int, int]
    hfov_deg: float = 90.0
    offset: tuple[float, float, float] = (0.30, 0.0, 0.08)   # ego/closeup: in the base frame
    pitch_down_deg: float = 17.0
    distance: float = 1.6         # chase
    height: float = 0.9
    world_pose: tuple[float, float, float, float, float] | None = None  # overview: x, y, z, yaw, pitch


def focal_length_for(hfov_deg: float, horizontal_aperture_mm: float = 20.955) -> float:
    return (horizontal_aperture_mm / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def _current_stage():
    """The open USD stage: ``omni.usd``'s context when Kit is running, ``pxr.Usd``'s context otherwise.

    ``pxr.Usd.GetContext`` does not exist in the USD build that ships with Isaac Sim 6.2 (measured 2026-11-23:
    ``AttributeError: module 'pxr.Usd' has no attribute 'GetContext'``), and importing ``pxr`` before ``omni.usd``
    breaks the Kit context anyway, so ask omni.usd first.
    """
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            return stage
    except (ImportError, AttributeError):
        pass
    from pxr import Usd

    return Usd.GetContext().GetStage()


class CameraRig:
    """Creates the camera prims, keeps the chase camera behind the robot, and captures RGB frames.

    ``robot_prim_path`` is the base-link prim (e.g. ``/World/envs/env_0/Robot/Geometry/base``): the EGO camera is
    parented *under* it, so its transform to the base is constant by construction and the QA gate can verify it.
    """

    def __init__(self, *, robot_prim_path: str, specs: list[CameraSpec], stage: Any = None):
        from pxr import Gf, Usd, UsdGeom

        self._Usd, self._UsdGeom, self._Gf = Usd, UsdGeom, Gf
        self.stage = stage or _current_stage()
        self.robot_prim_path = robot_prim_path
        self.specs = {spec.name: spec for spec in specs}
        self.paths: dict[str, str] = {}
        self._render_products: dict[str, Any] = {}
        self._annotators: dict[str, Any] = {}
        self._chase_yaw: float | None = None
        self._create()

    # -- prims
    def _create(self) -> None:
        UsdGeom = self._UsdGeom
        for spec in self.specs.values():
            # The EGO camera is defined *under* the base-link prim: the transform to the base is then exact by
            # construction and never drifts, which is exactly what the clip QA gate checks.
            path = (f"{self.robot_prim_path}/{spec.name}_camera" if spec.kind == "ego"
                    else f"/World/cerberus_cameras/{spec.name}")
            camera = UsdGeom.Camera.Define(self.stage, path)
            camera.CreateFocalLengthAttr(focal_length_for(spec.hfov_deg))
            camera.CreateHorizontalApertureAttr(20.955)
            camera.CreateClippingRangeAttr(self._Gf.Vec2f(0.05, 1000.0))
            xform = UsdGeom.Xformable(camera)
            xform.AddTranslateOp()
            xform.AddOrientOp()
            self.paths[spec.name] = path
            self._place_static(spec)

    def _place_static(self, spec: CameraSpec) -> None:
        if spec.kind == "ego":
            self._set_local(spec.name, np.array(spec.offset), spec.pitch_down_deg)
        elif spec.kind == "closeup":
            # a fixed pose relative to the robot, to the front-right, looking at the arm
            self._set_local(spec.name, np.array(spec.offset), spec.pitch_down_deg)
        elif spec.kind == "overview" and spec.world_pose is not None:
            x, y, z, yaw, pitch = spec.world_pose
            self._set_world(spec.name, np.array([x, y, z]), yaw, pitch)

    def _set_local(self, name: str, offset: np.ndarray, pitch_down_deg: float) -> None:
        """Position/orient the camera in its parent's frame (used for the EGO camera under the base)."""
        R = rotation_from_forward(forward_from_yaw_pitch(0.0, math.radians(pitch_down_deg)))
        q = quat_from_matrix(R)
        self._write_xform(name, offset, q)

    def _set_world(self, name: str, eye: np.ndarray, yaw: float, pitch_down_deg: float) -> None:
        R = rotation_from_forward(forward_from_yaw_pitch(yaw, math.radians(pitch_down_deg)))
        q = quat_from_matrix(R)
        self._write_xform(name, eye, q)

    def _write_xform(self, name: str, pos: np.ndarray, quat_wxyz: tuple[float, ...]) -> None:
        prim = self.stage.GetPrimAtPath(self.paths[name])
        xform = self._UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
        translate = ops.get("xformOp:translate") or xform.AddTranslateOp()
        orient = ops.get("xformOp:orient") or xform.AddOrientOp()
        translate.Set(self._Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        w, x, y, z = quat_wxyz
        orient.Set(self._Gf.Quatf(float(w), float(x), float(y), float(z)))

    # -- per-step
    def world_pose(self, name: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        prim = self.stage.GetPrimAtPath(self.paths[name])
        m = self._UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        imag = q.GetImaginary()
        return np.array([t[0], t[1], t[2]]), (q.GetReal(), imag[0], imag[1], imag[2])

    def base_pose(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        prim = self.stage.GetPrimAtPath(self.robot_prim_path)
        m = self._UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        imag = q.GetImaginary()
        return np.array([t[0], t[1], t[2]]), (q.GetReal(), imag[0], imag[1], imag[2])

    def update(self, *, low_pass: float = 0.25) -> None:
        """Move the chase camera behind the base with a low-passed yaw (no snapping when the robot turns)."""
        base_pos, base_quat = self.base_pose()
        yaw = yaw_of(base_quat)
        if self._chase_yaw is None:
            self._chase_yaw = yaw
        else:
            delta = math.atan2(math.sin(yaw - self._chase_yaw), math.cos(yaw - self._chase_yaw))
            self._chase_yaw += low_pass * delta
        for spec in self.specs.values():
            if spec.kind != "chase":
                continue
            eye, R = chase_pose(base_pos, self._chase_yaw, distance=spec.distance, height=spec.height,
                                pitch_down_deg=spec.pitch_down_deg)
            self._write_xform(spec.name, eye, quat_from_matrix(R))

    # -- capture
    def _ensure_products(self) -> None:
        if self._render_products:
            return
        import omni.replicator.core as rep

        for spec in self.specs.values():
            rp = rep.create.render_product(self.paths[spec.name], spec.resolution)
            annotator = rep.annotators.get("rgb")
            annotator.attach(rp)
            self._render_products[spec.name] = rp
            self._annotators[spec.name] = annotator

    def capture(self, name: str) -> np.ndarray:
        """One RGB frame (H, W, 3) uint8; requires a render step after the camera moved."""
        self._ensure_products()
        return rgb_from_annotator_data(self._annotators[name].get_data())

    def close(self) -> None:
        for rp in self._render_products.values():
            try:
                rp.destroy()
            except Exception:  # noqa: BLE001 - teardown must never mask the run's verdict
                pass
        self._render_products.clear()
        self._annotators.clear()


def rgb_from_annotator_data(data) -> np.ndarray:
    """Normalise what a Replicator RGB annotator returns to (H, W, 3) uint8.

    Replicator 1.13 (Sim 6.2) hands back the array itself; older versions wrap it as ``{"data": array}``.
    Measured 2026-11-23: ``data["data"]`` on an ndarray raised ``IndexError: only integers ... are valid indices``.
    """
    import numpy as np

    if isinstance(data, dict):
        data = data.get("data", data)
    rgb = np.asarray(data)
    if rgb.dtype.kind == "f":
        scale = 255.0 if float(np.nanmax(rgb, initial=0.0)) <= 1.5 else 1.0
        rgb = np.clip(rgb * scale, 0, 255).round().astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[1] > 8 and rgb.shape[2] > 8:
        rgb = np.moveaxis(rgb, 0, -1)  # channels-first layouts
    if rgb.ndim == 3:
        if rgb.shape[2] > 3:
            rgb = rgb[..., :3]
        elif rgb.shape[2] < 3:
            rgb = rgb[..., 0]
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def pose_log_for(spec: CameraSpec) -> PoseLog:
    return PoseLog(camera=spec.kind, width=spec.resolution[0], height=spec.resolution[1], hfov_deg=spec.hfov_deg)


def write_poses(path: Path | str, log: PoseLog) -> Path:
    return log.write(path)
