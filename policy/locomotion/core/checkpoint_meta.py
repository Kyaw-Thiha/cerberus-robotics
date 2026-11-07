"""Checkpoint metadata sidecars: build, validate, write -- and never overwrite a checkpoint.

A ``.pt`` file records weights, not the world those weights were trained in. The GA retrain (H2 P2)
changes the world (Isaac Sim 6.2 / Lab 3.0, rsl-rl 5.x, PhysX vs Newton) while keeping the recipe
identical, so a checkpoint without a sidecar saying *which* stack produced it cannot be compared to
anything, and the Phase 0.5 gate would be gating an unidentified artifact.

Design constraints, from the H2 handoff:
  * new files only -- every write here refuses to clobber an existing path (a silently replaced
    checkpoint destroys the before/after comparison the gate report is required to contain);
  * the resolved physics backend is recorded, not assumed ("stock defaults to Newton on Lab 3.0");
  * wall-clock is recorded, because the retrain has a 2x-RUN_002 cap and exceeding it is a reportable
    outcome rather than something to recompute later.

Pure python (json/hashlib/struct/git). No torch, no Isaac: importable and testable anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_FIELDS = (
    "checkpoint",
    "git_hash",
    "task_id",
    "physics_backend",
    "iterations",
    "seed",
    "num_envs",
    "config_hash",
    "wall_clock_s",
    "final_reward",
)

# Fields whose absence makes a sidecar useless for the gate/report even if not strictly required.
RECOMMENDED_FIELDS = ("env_report", "terrain_levels", "created_at", "onnx", "run_dir")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_new_paths(paths) -> list[str]:
    """Raise if any path already exists. Returns the paths (unchanged) when all are free."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing artifact(s): " + ", ".join(existing)
        )
    return [str(p) for p in paths]


def config_hash(paths) -> str:
    """Stable hash of a run's dumped config (``params/*.yaml``), file names included.

    Hashes file *names and bytes*, not parsed values: the question it answers is "were these the same
    config files", which is what a rerun of the same recipe looks like on disk.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        if path.is_dir():
            members = sorted(p for p in path.rglob("*") if p.is_file())
        else:
            members = [path]
        for member in members:
            digest.update(str(member.name).encode())
            digest.update(b"\0")
            digest.update(member.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()[:16]


def git_hash(cwd: str | os.PathLike | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _iter_tfrecords(data: bytes):
    """Yield the raw payload of each TFRecord in ``data`` (length + masked-crc framing)."""
    offset = 0
    total = len(data)
    while offset + 12 <= total:
        (length,) = struct.unpack("<Q", data[offset : offset + 8])
        start = offset + 12
        end = start + length
        if end > total:
            return  # truncated write (a training run still in progress): keep what parsed
        yield data[start:end]
        offset = end + 4


def _read_varint(record: bytes, i: int):
    """Read a protobuf varint at ``i`` -> (value, next_index) or (None, i) when truncated."""
    shift = 0
    value = 0
    while i < len(record):
        byte = record[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7
    return None, i


def _event_wall_time(record: bytes):
    """wall_time (Event field 1, double, wire type 1) from one serialized Event protobuf, or None."""
    i = 0
    while i < len(record):
        key = record[i]
        field, wire = key >> 3, key & 0x07
        i += 1
        if wire == 1:  # 64-bit: this is where wall_time lives
            if i + 8 > len(record):
                return None
            if field == 1:
                return struct.unpack("<d", record[i : i + 8])[0]
            i += 8
        elif wire == 2:  # length-delimited: skip payload
            length, i = _read_varint(record, i)
            if length is None:
                return None
            i += length
        elif wire == 0:  # varint
            _, i = _read_varint(record, i)
        elif wire == 5:  # 32-bit
            i += 4
        else:
            return None
    return None


def tfevents_wall_time_span(path: str | os.PathLike) -> dict:
    """First/last wall_time in a tensorboard event file -> {first, last, span_s, events}.

    Used to recover a run's wall-clock without asking anyone to have timed it: the event file is
    written by the trainer itself. Returns ``{"span_s": None}`` when nothing parseable is found.
    """
    data = Path(path).read_bytes()
    times = []
    for record in _iter_tfrecords(data):
        wall = _event_wall_time(record)
        if wall is not None:
            times.append(wall)
    if not times:
        return {"first": None, "last": None, "span_s": None, "events": 0}
    return {"first": min(times), "last": max(times), "span_s": max(times) - min(times), "events": len(times)}


def build_meta(
    *,
    checkpoint: str,
    task_id: str,
    physics_backend: str | dict,
    iterations: int,
    seed: int | None,
    num_envs: int,
    wall_clock_s: float,
    final_reward: float,
    config_hash_value: str,
    git_hash_value: str | None,
    onnx: str | None = None,
    env_report: str | dict | None = None,
    terrain_levels: dict | None = None,
    run_dir: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble a sidecar dict, validating the fields the gate and report depend on."""
    meta = {
        "checkpoint": checkpoint,
        "onnx": onnx,
        "git_hash": git_hash_value,
        "task_id": task_id,
        "physics_backend": physics_backend,
        "iterations": iterations,
        "seed": seed,
        "num_envs": num_envs,
        "config_hash": config_hash_value,
        "wall_clock_s": round(float(wall_clock_s), 1),
        "final_reward": float(final_reward),
        "terrain_levels": terrain_levels,
        "env_report": env_report,
        "run_dir": run_dir,
        "created_at": _utc_now(),
    }
    if extra:
        meta["extra"] = extra
    validate_meta(meta)
    return meta


def validate_meta(meta: dict) -> None:
    """Raise ValueError on a sidecar that would be unreadable or un-attributable later."""
    missing = [f for f in REQUIRED_FIELDS if f not in meta or meta[f] is None]
    if missing:
        raise ValueError(f"sidecar missing required field(s): {missing}")
    if not isinstance(meta["physics_backend"], (str, dict)) or not meta["physics_backend"]:
        raise ValueError("physics_backend must be a non-empty string or dict")
    if int(meta["iterations"]) <= 0:
        raise ValueError(f"iterations must be positive, got {meta['iterations']!r}")
    if float(meta["wall_clock_s"]) <= 0:
        raise ValueError(f"wall_clock_s must be positive, got {meta['wall_clock_s']!r}")
    if not str(meta["config_hash"]):
        raise ValueError("config_hash must be non-empty")


def write_meta(path: str | os.PathLike, meta: dict, *, checkpoint_paths=()) -> str:
    """Validate and write a sidecar atomically.

    Two guards: the sidecar itself must not exist yet (never clobber an earlier candidate's metadata),
    and every checkpoint it describes must already be on disk -- a sidecar pointing at a file that is
    not there is worse than no sidecar, because it looks like provenance.
    """
    validate_meta(meta)
    ensure_new_paths([path])
    if isinstance(checkpoint_paths, (str, os.PathLike)):
        checkpoint_paths = [checkpoint_paths]
    missing = [str(p) for p in checkpoint_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"sidecar names checkpoint(s) that do not exist: {', '.join(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; sidecars are read by eval scripts and synced to R2
        os.replace(tmp, path)  # atomic: a half-written sidecar never appears
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return str(path)


def load_meta(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------------
# Tensorboard scalars (no tensorboard dependency)
# --------------------------------------------------------------------------------------------

_WIRE_VARINT, _WIRE_I64, _WIRE_LEN, _WIRE_I32 = 0, 1, 2, 5


def _iter_proto_fields(buf: bytes):
    """Yield ``(field_number, wire_type, value)`` for one protobuf message body.

    ``value`` is an int for varint fields, bytes for length-delimited fields, and raw bytes for the
    fixed-width types. Enough protobuf to read a tensorboard Event/Summary without pulling the
    ``tensorboard`` package into the image.
    """
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        if key is None:
            return
        field, wire = key >> 3, key & 0x07
        if wire == _WIRE_VARINT:
            value, i = _read_varint(buf, i)
            if value is None:
                return
            yield field, wire, value
        elif wire == _WIRE_I64:
            if i + 8 > len(buf):
                return
            yield field, wire, buf[i : i + 8]
            i += 8
        elif wire == _WIRE_LEN:
            length, i = _read_varint(buf, i)
            if length is None or i + length > len(buf):
                return
            yield field, wire, buf[i : i + length]
            i += length
        elif wire == _WIRE_I32:
            if i + 4 > len(buf):
                return
            yield field, wire, buf[i : i + 4]
            i += 4
        else:
            return


def _summary_scalars(record: bytes) -> dict:
    """{tag: simple_value} from one serialized Event (only scalar summaries are decoded)."""
    out = {}
    for field, wire, value in _iter_proto_fields(record):
        if field != 5 or wire != _WIRE_LEN:  # Event.summary is field 5; field 2 is Event.step
            continue
        for sub_field, sub_wire, sub_value in _iter_proto_fields(value):
            if sub_field != 1 or sub_wire != _WIRE_LEN:  # Summary.value
                continue
            tag = None
            scalar = None
            for v_field, v_wire, v_value in _iter_proto_fields(sub_value):
                if v_field == 1 and v_wire == _WIRE_LEN:
                    tag = v_value.decode("utf-8", "replace")
                elif v_field == 2 and v_wire == _WIRE_I32:
                    scalar = struct.unpack("<f", v_value)[0]
            if tag is not None and scalar is not None:
                out[tag] = scalar
    return out


def tfevents_last_scalars(path: str | os.PathLike, tags) -> dict:
    """Last logged value of each tag in ``tags`` (exact tag names) from a tensorboard event file.

    Returns ``{tag: {"value": float, "step": int}}`` for the tags that appeared; missing tags are
    absent rather than None, so a caller can tell "not logged" from "logged as null".
    """
    wanted = set(tags)
    data = Path(path).read_bytes()
    latest: dict = {}
    for record in _iter_tfrecords(data):
        step = 0
        for field, wire, value in _iter_proto_fields(record):
            if field == 2 and wire == _WIRE_VARINT:  # Event.step
                step = value
        for tag, scalar in _summary_scalars(record).items():
            if tag in wanted:
                latest[tag] = {"value": scalar, "step": step}
    return latest


# Training tags logged by rsl-rl / Isaac Lab; the sidecar's "final reward and terrain-level stats"
# come from these rather than from someone re-typing numbers off a log.
TAG_MEAN_REWARD = "Train/mean_reward"
TAG_TERRAIN_LEVELS = "Curriculum/terrain_levels"
TAG_TERRAIN_LEVEL_MAX = "Curriculum/terrain_level_max"
TAG_TERRAIN_FRAC_AT_MAX = "Curriculum/terrain_level_frac_at_max"
TAG_BASE_CONTACT = "Episode_Termination/base_contact"


def training_summary(path: str | os.PathLike) -> dict:
    """Final reward, terrain-level distribution and fall rate at the end of a run.

    Keys are absent when the run did not log that tag (e.g. the diagnostic terrain terms only exist
    on runs started after they were added), so a caller can distinguish "not logged" from "zero".
    """
    scalars = tfevents_last_scalars(
        path,
        [TAG_MEAN_REWARD, TAG_TERRAIN_LEVELS, TAG_TERRAIN_LEVEL_MAX, TAG_TERRAIN_FRAC_AT_MAX, TAG_BASE_CONTACT],
    )
    summary: dict = {}
    if TAG_MEAN_REWARD in scalars:
        summary["final_reward"] = scalars[TAG_MEAN_REWARD]["value"]
        summary["final_iteration"] = scalars[TAG_MEAN_REWARD]["step"]
    terrain = {}
    if TAG_TERRAIN_LEVELS in scalars:
        terrain["mean"] = scalars[TAG_TERRAIN_LEVELS]["value"]
    if TAG_TERRAIN_LEVEL_MAX in scalars:
        terrain["max"] = scalars[TAG_TERRAIN_LEVEL_MAX]["value"]
    if TAG_TERRAIN_FRAC_AT_MAX in scalars:
        terrain["frac_at_max"] = scalars[TAG_TERRAIN_FRAC_AT_MAX]["value"]
    if terrain:
        summary["terrain_levels"] = terrain
    if TAG_BASE_CONTACT in scalars:
        summary["fall_rate"] = scalars[TAG_BASE_CONTACT]["value"]
    return summary
