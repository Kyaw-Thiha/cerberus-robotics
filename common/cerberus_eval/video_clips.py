"""Clip selection, naming and the headless writer every H4 driver shares.

Two halves, on purpose:

* **Selection and naming** are pure Python and tested without Isaac. The rule (handoff A6): keep the first success
  and the first failure when both occur, then fill up to ``max_clips`` in episode order, always chronologically
  sorted for playback. ``max_clips=0`` disables recording for pure-metrics runs.
* **Recording** wraps the repo's existing viewport capture (``policy.locomotion.core.script_utils.
  capture_viewport_frame``, which handles the Lab 2.x / 3.0 difference) and ``imageio``. Both are imported inside
  the methods, so this module imports locally; only :meth:`ClipRecorder.frame` needs the Isaac app.

Naming is ``<scene>_<seed>_<outcome>.mp4`` with the metrics JSON at the same stem (``..._<outcome>.json``), so a
clip and its numbers cannot drift apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Self, Sequence

DEFAULT_MAX_CLIPS = 3
DEFAULT_FPS = 30
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ClipRequest:
    """One planned clip: which episode to record and what to call it."""

    episode_index: int
    filename: str
    outcome: str


def safe_name(text: str) -> str:
    """Filesystem-safe form of a scene id (``grscenes:MV7..._usd`` -> ``MV7..._usd``)."""
    text = text.split(":", 1)[-1]
    return _UNSAFE.sub("_", text).strip("_") or "scene"


def clip_filename(scene: str, seed: int, outcome: str) -> str:
    return f"{safe_name(scene)}_{int(seed)}_{safe_name(outcome)}.mp4"


def metrics_filename(clip_name: str) -> str:
    return Path(clip_name).with_suffix(".json").name


def plan_clips(outcomes: Sequence[str], max_clips: int = DEFAULT_MAX_CLIPS) -> list[int]:
    """Indices to record (see the module docstring for the rule); chronological order.

    With ``max_clips == 1`` and both a success and a failure present, the first success is kept and the failure is
    dropped -- one clip cannot honour both halves of the rule. Callers that must show the failure use
    ``max_clips >= 2``.
    """
    if max_clips <= 0:
        return []
    success = next((i for i, o in enumerate(outcomes) if o == "success"), None)
    failure = next((i for i, o in enumerate(outcomes) if o != "success"), None)
    keep: list[int] = []
    if success is not None:
        keep.append(success)
    if failure is not None and len(keep) < max_clips:
        keep.append(failure)
    for i in range(len(outcomes)):
        if len(keep) >= max_clips:
            break
        if i not in keep:
            keep.append(i)
    return sorted(keep)


def plan_clip_requests(scene: str, seed: int, outcomes: Sequence[str], max_clips: int = DEFAULT_MAX_CLIPS) -> list[ClipRequest]:
    """``plan_clips`` plus the file each clip is written to."""
    return [
        ClipRequest(episode_index=i, filename=clip_filename(scene, seed, outcomes[i]), outcome=outcomes[i])
        for i in plan_clips(outcomes, max_clips)
    ]


class ClipRecorder:
    """Writes one clip at a time from a live Isaac Lab env; a no-op when ``enabled`` is False.

    ``downscale`` decimates frames by integer strides (``2`` halves each dimension) -- the pod's headless frames are
    1080p-class and the report only needs them readable.
    """

    def __init__(self, output_dir: str | Path, *, fps: int = DEFAULT_FPS, enabled: bool = True, downscale: int = 1):
        if downscale < 1:
            raise ValueError("downscale must be >= 1")
        self.output_dir = Path(output_dir)
        self.fps = int(fps)
        self.enabled = bool(enabled)
        self.downscale = int(downscale)
        self._writer: Any = None
        self._path: Path | None = None
        self.written: list[Path] = []

    def begin(self, filename: str) -> Path | None:
        """Open a writer for ``filename``; returns the path (or None when disabled)."""
        if not self.enabled:
            return None
        import imageio

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / filename
        self._writer = imageio.get_writer(str(self._path), fps=self.fps, macro_block_size=None)
        return self._path

    def frame(self, raw_env: Any, env_index: int) -> None:
        """Capture one viewport frame of ``env_index`` (Isaac required)."""
        if self._writer is None:
            return
        from policy.locomotion.core.script_utils import capture_viewport_frame, point_viewport_at_env

        point_viewport_at_env(raw_env, env_index)
        frame = capture_viewport_frame(raw_env)
        if self.downscale > 1:
            frame = frame[:: self.downscale, :: self.downscale]
        self._writer.append_data(frame)

    def end(self) -> Path | None:
        """Close the current clip; returns its path."""
        if self._writer is None:
            return None
        self._writer.close()
        self._writer = None
        path, self._path = self._path, None
        if path is not None:
            self.written.append(path)
        return path

    def close(self) -> None:
        self.end()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def clip_and_metrics_paths(output_dir: str | Path, clip: ClipRequest) -> tuple[Path, Path]:
    """The clip path and the JSON path beside it (same stem)."""
    output_dir = Path(output_dir)
    return output_dir / clip.filename, output_dir / metrics_filename(clip.filename)


def summarise_clips(requests: Iterable[ClipRequest], written: Iterable[str | Path]) -> dict[str, Any]:
    """Manifest entry: planned clips, written clips and any planned clip that produced no file."""
    planned = [r.filename for r in requests]
    written_names = [Path(p).name for p in written]
    return {
        "planned": planned,
        "written": written_names,
        "missing": [n for n in planned if n not in written_names],
        "count": len(written_names),
    }


def outcome_counts(outcomes: Iterable[str]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    return counts
