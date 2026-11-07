#!/usr/bin/env python3
"""Write the Cerberus environment report for a pod session (H2 P0).

Pure python, no Isaac import: collects what is knowable *without* launching Kit, so it runs in
seconds before any GPU time is spent. The physics backend that the env actually resolved is not
knowable statically (it is chosen by an env cfg's ``__post_init__``) -- that half is written by
``scripts/check_isaac_quat_convention.py``, which merges its own findings into the same file.

Sources, in order of preference per field:
  1. ``/etc/cerberus_versions.json`` -- written into the image at build time (Sim / Lab / Arena /
     rsl-rl / torch), so it is the image's own claim, not a runtime guess.
  2. installed distribution metadata (``importlib.metadata``).
  3. environment variables (``ISAACLAB_PATH`` etc.) and ``git`` for the Lab checkout's commit.

Usage (either interpreter is fine -- no Isaac needed):

    python3 scripts/env_report.py                     # -> outputs/env_report.json
    python3 scripts/env_report.py --also docs/reports/env_report_2026-11-07.json

Exit codes: 0 report written, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_VERSIONS_FILE = "/etc/cerberus_versions.json"

# Distribution names to try, in order: the Isaac Sim pip/wheel name changed across releases.
ISAAC_SIM_DISTS = ("isaacsim", "isaac-sim", "isaacsim-rl")
ISAAC_LAB_DISTS = ("isaaclab", "isaaclab_tasks", "isaaclab-assets")
RSL_RL_DISTS = ("rsl-rl-lib", "rsl_rl")


def _dist_version(names: tuple[str, ...]) -> dict:
    """First importable distribution among ``names`` -> {name, version, tried}."""
    tried = []
    for name in names:
        try:
            return {"name": name, "version": importlib.metadata.version(name), "tried": tried}
        except importlib.metadata.PackageNotFoundError:
            tried.append(name)
    return {"name": None, "version": None, "tried": tried}


def _image_versions() -> dict:
    """Contents of the image-baked versions file, or {} when absent/unreadable."""
    try:
        with open(IMAGE_VERSIONS_FILE) as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        return {}


def _git(args: list[str], cwd: str | os.PathLike | None = None) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _torch_info() -> dict:
    """torch / CUDA / GPU, or an 'unavailable' marker. Imported lazily: torch is a 2 GB import."""
    info: dict = {"available": False}
    try:
        import torch  # noqa: PLC0415 -- deliberately deferred
    except Exception as e:  # noqa: BLE001 -- any import failure means 'not usable here'
        info["error"] = f"{type(e).__name__}: {e}"
        return info
    info.update(
        {
            "available": True,
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    )
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": props.name,
            "vram_gb": round(props.total_memory / 1024**3, 2),
            "capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
    return info


def _nvidia_smi() -> dict | None:
    """Driver version + GPU name/VRAM straight from the host driver, independent of torch."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            rows.append(
                {
                    "name": parts[0],
                    "vram_mib": float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else parts[1],
                    "driver_version": parts[2],
                }
            )
    return {"gpus": rows} if rows else None


def _isaaclab_commit() -> dict:
    """Commit of the ISAACLAB_PATH checkout (what the image actually built against)."""
    path = os.environ.get("ISAACLAB_PATH") or "/workspace/isaaclab"
    commit = _git(["rev-parse", "HEAD"], cwd=path) if os.path.isdir(path) else None
    describe = _git(["describe", "--tags", "--always", "--dirty"], cwd=path) if os.path.isdir(path) else None
    return {"path": path, "commit": commit, "describe": describe}


def _cerberus_git() -> dict:
    return {
        "commit": _git(["rev-parse", "HEAD"], cwd=_REPO_ROOT),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=_REPO_ROOT),
        "describe": _git(["describe", "--tags", "--always", "--dirty"], cwd=_REPO_ROOT),
        "dirty": bool(_git(["status", "--porcelain"], cwd=_REPO_ROOT)),
    }


def build_report(repo_root: Path = _REPO_ROOT) -> dict:
    """Assemble the report dict. Pure function of the environment (no writes)."""
    image = _image_versions()
    torch_info = _torch_info()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "hostname": platform.node(),
        "image_versions_file": image or None,
        "isaac_sim": _dist_version(ISAAC_SIM_DISTS),
        "isaac_lab": {**_dist_version(ISAAC_LAB_DISTS), "checkout": _isaaclab_commit()},
        "rsl_rl": _dist_version(RSL_RL_DISTS),
        "torch": torch_info,
        "nvidia_smi": _nvidia_smi(),
        "cerberus_git": _cerberus_git(),
        "env": {
            key: os.environ.get(key)
            for key in ("ISAACLAB_PATH", "TERM", "CERBERUS_ALLOW_BAD_DRIVER_RENDER", "CERBERUS_QUAT_CONVENTION")
        },
        # Filled in by scripts/check_isaac_quat_convention.py -- a static report cannot resolve it.
        "resolved_physics": None,
        "quat_convention": None,
        "runtime_probe": None,
    }
    # The image file is authoritative for the versions it pins; surface them at top level too.
    for src, dst in (
        ("isaac_sim", "isaac_sim_version"),
        ("isaaclab", "isaac_lab_version"),
        ("isaac_lab", "isaac_lab_version"),
        ("rsl_rl", "rsl_rl_version"),
        ("torch", "torch_version"),
    ):
        value = image.get(src)
        if value and dst not in report:
            report[dst] = value
    return report


def merge_runtime_probe(report_path: Path, probe: dict) -> dict:
    """Fold a runtime probe (physics backend, quat convention) into an existing report file."""
    report = json.loads(report_path.read_text())
    for key in ("resolved_physics", "quat_convention", "runtime_probe"):
        if key in probe:
            report[key] = probe[key]
    report_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=_REPO_ROOT / "outputs" / "env_report.json")
    parser.add_argument(
        "--also",
        action="append",
        default=[],
        type=Path,
        help="Additional path to write the same JSON to (e.g. the committed docs/reports copy). Repeatable.",
    )
    args = parser.parse_args(argv)

    report = build_report()
    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    for path in [args.output, *args.also]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"[env_report] wrote {path}")

    # One-line summary for the log; the JSON has everything.
    sim = report.get("isaac_sim_version") or (report["isaac_sim"] or {}).get("version")
    lab = report.get("isaac_lab_version") or (report["isaac_lab"] or {}).get("version")
    gpu = (report.get("nvidia_smi") or {}).get("gpus") or [{}]
    print(
        f"[env_report] sim={sim} lab={lab} rsl_rl={report['rsl_rl'].get('version')} "
        f"torch={(report['torch'] or {}).get('version')} gpu={gpu[0].get('name')} "
        f"commit={report['cerberus_git'].get('commit')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
