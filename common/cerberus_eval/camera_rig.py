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


def quat_to_matrix(quat_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotate_vector(vec, quat_wxyz) -> np.ndarray:
    """Rotate a vector by a WXYZ quaternion (three.js-style, no scipy dependency)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    vx, vy, vz = (float(v) for v in vec)
    # t = 2 * cross(q.xyz, v); v' = v + w*t + cross(q.xyz, t)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([vx + w * tx + (y * tz - z * ty),
                     vy + w * ty + (z * tx - x * tz),
                     vz + w * tz + (x * ty - y * tx)])


def yaw_of(quat_wxyz: tuple[float, float, float, float]) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def clear_boom(base_pos: np.ndarray, base_yaw: float, *, distance: float, height: float, occupancy,
               min_distance: float = 0.6, step: float = 0.1) -> tuple[float, bool]:
    """Shorten the chase boom so the camera stays in free space: never through a wall, never closer than 0.6 m.

    ``occupancy`` is a :class:`planning.occupancy.OccupancyGrid` (only ``is_free`` is used). Returns the usable
    distance and whether it was pulled in.
    """
    fwd = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])
    usable = min(float(min_distance), float(distance))
    d = usable
    while d <= distance + 1e-9:
        eye = np.asarray(base_pos, dtype=float) + np.array([0.0, 0.0, height]) - d * fwd
        if occupancy.is_free(float(eye[0]), float(eye[1]), inflated=True):
            usable = d
        else:
            break
        d += step
    return usable, usable < distance - 1e-9


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
    near_clip_m: float = 0.05     # walls/cushions closer than this are hidden instead of filling the frame
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
        self._authored: dict[str, tuple[np.ndarray, tuple[float, ...]]] = {}
        self._create()

    # -- prims
    def _create(self) -> None:
        UsdGeom = self._UsdGeom
        for spec in self.specs.values():
            # All cameras are world prims placed from the articulation pose each frame. Parenting the EGO camera
            # under the base prim inherited that prim's own rotation (the fused ARX5 base link is rolled 90 deg),
            # which produced sideways frames; a world prim with an explicit body-frame offset does not.
            path = f"/World/cerberus_cameras/{spec.name}"
            camera = UsdGeom.Camera.Define(self.stage, path)
            camera.CreateFocalLengthAttr(focal_length_for(spec.hfov_deg))
            camera.CreateHorizontalApertureAttr(20.955)
            camera.CreateClippingRangeAttr(self._Gf.Vec2f(max(0.02, float(spec.near_clip_m)), 1000.0))
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
        self._authored[name] = (np.array([float(pos[0]), float(pos[1]), float(pos[2])]), tuple(quat_wxyz))
        translate.Set(self._Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        w, x, y, z = quat_wxyz
        orient.Set(self._Gf.Quatf(float(w), float(x), float(y), float(z)))

    # -- per-step
    def world_pose(self, name: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """The camera's world pose. On Isaac Lab 3.0 the articulation poses live in Fabric and the USD xforms are not
        refreshed, so the transform this rig authored is authoritative (the stage read is only a fallback)."""
        if name in self._authored:
            pos, quat = self._authored[name]
            return np.array(pos), tuple(quat)
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

    def update(self, *, low_pass: float = 0.25, base_pos=None, base_quat_wxyz=None, occupancy=None) -> None:
        """Place the moving cameras from the robot's live pose.

        `base_pos`/`base_quat_wxyz` should come from the articulation's physics state: on Isaac Lab 3.0 the USD xform
        of the robot is not refreshed, so reading the stage (the fallback) leaves the chase camera parked (measured
        2026-11-23: the logged camera never moved while the robot walked).
        """
        if base_pos is None or base_quat_wxyz is None:
            base_pos, base_quat = self.base_pose()
        else:
            base_pos, base_quat = np.asarray(base_pos, dtype=float), tuple(float(v) for v in base_quat_wxyz)
        yaw = yaw_of(base_quat)
        if self._chase_yaw is None:
            self._chase_yaw = yaw
        else:
            delta = math.atan2(math.sin(yaw - self._chase_yaw), math.cos(yaw - self._chase_yaw))
            self._chase_yaw += low_pass * delta
        for spec in self.specs.values():
            if spec.kind != "chase":
                continue
            boom = spec.distance
            if occupancy is not None:
                boom, _pulled = clear_boom(base_pos, self._chase_yaw, distance=spec.distance,
                                           height=spec.height, occupancy=occupancy)
            # scale the height with a pulled-in boom so the look angle (and the robot in frame) stays the same
            height = spec.height * (boom / spec.distance) if spec.distance > 0 else spec.height
            eye, R = chase_pose(base_pos, self._chase_yaw, distance=boom, height=max(0.3, height),
                                pitch_down_deg=spec.pitch_down_deg)
            self._write_xform(spec.name, eye, quat_from_matrix(R))
        for spec in self.specs.values():
            if spec.kind != "ego":
                continue
            # body-mounted: world pose = base pose * fixed local offset, with a fixed downward pitch
            R_base = quat_to_matrix(base_quat)
            world = np.asarray(base_pos, dtype=float) + R_base @ np.asarray(spec.offset, dtype=float)
            R_local = rotation_from_forward(forward_from_yaw_pitch(0.0, math.radians(spec.pitch_down_deg)))
            q_world = quat_from_matrix(R_base @ R_local)
            self._write_xform(spec.name, world, q_world)
            self._authored[spec.name] = (np.asarray(world, dtype=float), tuple(float(v) for v in q_world))

    # -- capture
    def _ensure_products(self) -> None:
        if self._render_products or getattr(self, "_vp_windows", None):
            return
        rep = None
        for attempt in range(3):
            try:
                import omni.replicator.core as rep
                break
            except ModuleNotFoundError:
                # the extension is loaded asynchronously after app launch; give it app updates, then retry
                try:
                    import omni.kit.app

                    for _ in range(100):
                        omni.kit.app.get_app().update()
                except Exception:  # noqa: BLE001
                    pass
                rep = None
        if rep is None:
            # Some images ship replicator in extscache without enabling it; a headless viewport capture works
            # without it (verified 2026-11-24: capture_viewport_to_file, ~10 app updates per frame).
            from omni.kit.viewport.utility import create_viewport_window

            self._vp_windows = {}
            for spec in self.specs.values():
                window = create_viewport_window(f"capture_{spec.name}", width=int(spec.resolution[0]),
                                                height=int(spec.resolution[1]))
                window.viewport_api.camera_path = self.paths[spec.name]
                self._vp_windows[spec.name] = window
            print("[camera_rig] using headless viewport capture (no replicator)", flush=True)
            return
        for spec in self.specs.values():
            rp = rep.create.render_product(self.paths[spec.name], spec.resolution)
            annotator = rep.annotators.get("rgb")
            annotator.attach(rp)
            self._render_products[spec.name] = rp
            self._annotators[spec.name] = annotator

    def _capture_viewport(self, name: str) -> np.ndarray:
        import os
        import tempfile

        import omni.kit.app
        from PIL import Image

        app = omni.kit.app.get_app()
        window = self._vp_windows[name]
        window.viewport_api.camera_path = self.paths[name]
        for _ in range(2):
            app.update()
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        from omni.kit.viewport.utility import capture_viewport_to_file

        capture_viewport_to_file(window.viewport_api, path)
        for _ in range(120):
            app.update()
            if os.path.getsize(path) > 0:
                break
        try:
            frame = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        finally:
            os.unlink(path)
        return frame

    def capture(self, name: str) -> np.ndarray:
        """One RGB frame (H, W, 3) uint8 (Replicator annotator or the headless viewport fallback).

        `sim.render()` alone leaves the RGB annotator empty on Sim 6.2 (measured 2026-11-23: shape (0,)), so the
        frame is requested through the orchestrator, which blocks until the render products have new data.
        """
        self._ensure_products()
        import omni.kit.app

        # Measured 2026-11-23 on Sim 6.2: rep.orchestrator.step(rt_subframes=1) costs 15.0 s per frame in this
        # 137-light scene, while pumping the app after sim.render() delivers the same 1280x720x4 frame in 0.11 s.
        # The orchestrator is kept only as a fallback for a stack where the pump yields nothing.
        app = omni.kit.app.get_app()
        app.update()
        data = self._annotators[name].get_data()
        if getattr(data, "size", None) == 0:
            import omni.replicator.core as rep

            rep.orchestrator.step(rt_subframes=2)
            data = self._annotators[name].get_data()
        return rgb_from_annotator_data(data)

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
