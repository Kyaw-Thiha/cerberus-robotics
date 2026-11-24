#!/usr/bin/env python3
"""J2 validation: our 25x60 spherical depth against a direct port of JEPLO's projection.

Their ``SphericalDepthGenerator._project_to_spherical_grid`` (third_party/jeplo, GPL-3) does, per hit:
sensor pose = base pose * offset; point in sensor frame; r = |p|; theta = asin(z/r); phi = atan2(y, x);
bins = round((angle - low)/res); min-range scatter. This script repeats that arithmetic independently in NumPy
and compares it with ``plugins.jeplo.sensor.SphericalDepth.project`` on a synthetic room (floor, walls, a box).

Honest limitation: this is a geometry-level comparison, not a run of their Isaac Lab 2.3 simulator (which is not
installed here). The real end-to-end check is the zero-shot walk (J4).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plugins.jeplo.sensor import (
    H_FOV,
    MAX_RANGE_M,
    V_FOV,
    H,
    SphericalDepth,
    V,
    quat_wxyz_to_matrix,
)


def reference_projection(hits_w: np.ndarray, origin, quat_wxyz, *, max_range_m=MAX_RANGE_M) -> np.ndarray:
    """Their projection, transcribed: same bins, same rounding, same unfilled value."""
    hits = np.asarray(hits_w, dtype=np.float64).reshape(-1, 3)
    R = quat_wxyz_to_matrix(quat_wxyz)
    local = (hits - np.asarray(origin, dtype=np.float64)) @ R
    r = np.linalg.norm(local, axis=-1)
    theta_low, theta_res = math.radians(V_FOV[0]), math.radians(V_FOV[1] - V_FOV[0]) / V
    phi_low, phi_res = math.radians(H_FOV[0]), math.radians(H_FOV[1] - H_FOV[0]) / H
    grid = np.full((V, H), float(max_range_m))
    for point, distance in zip(local, r):
        if not np.isfinite(distance) or distance <= 0.1 or distance > max_range_m:
            continue
        theta = math.asin(max(-1.0, min(1.0, point[2] / distance)))
        phi = math.atan2(point[1], point[0])
        ti = round((theta - theta_low) / theta_res)
        pi = round((phi - phi_low) / phi_res)
        if 0 <= ti < V and 0 <= pi < H:
            grid[ti, pi] = min(grid[ti, pi], distance)
    return grid


def synthetic_room_hits(n: int = 12000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Hits of a small room around the sensor: floor, four walls, a box. Deterministic."""
    rng = np.random.default_rng(seed)
    origin = np.array([0.275, 0.0, 1.548])   # sensor on a standing robot in world z
    quat = (0.0, 1.0, 0.0, 0.0)             # the authors' 180-deg x rotation
    xs = rng.uniform(-3.0, 3.0, n)
    ys = rng.uniform(-2.0, 2.0, n)
    hits = np.stack([xs, ys, np.zeros(n)], axis=1)
    # walls
    wall_x = np.where(rng.random(n) < 0.5, -3.0, 3.0)
    wall_z = rng.uniform(0.0, 2.5, n)
    hits[::4] = np.stack([wall_x[::4], rng.uniform(-2, 2, len(hits[::4])), wall_z[::4]], axis=1)
    hits[1::4] = np.stack([rng.uniform(-3, 3, len(hits[1::4])), np.where(rng.random(len(hits[1::4])) < 0.5, -2.0, 2.0),
                           wall_z[::4][: len(hits[1::4])]], axis=1)
    # a front wall at x=2.0, filling the 120-degree forward FOV
    front = rng.uniform(0, 1, (n // 4, 2))
    hits[2::4] = np.stack([np.full(len(front), 2.0), -1.5 + 3.0 * front[:, 0],
                           0.2 + 1.0 * front[:, 1]], axis=1)
    return hits, origin, quat


def compare(hits_w, origin, quat_wxyz, *, noise_std=0.0) -> dict:
    sensor = SphericalDepth(noise_std_m=noise_std, seed=0)
    ours = sensor.project(hits_w, origin, quat_wxyz)
    ref = reference_projection(hits_w, origin, quat_wxyz)
    diff = np.abs(ours - ref)
    filled = ref < MAX_RANGE_M - 1e-9
    return {
        "ours": ours, "reference": ref, "abs_error_m": diff,
        "max_error_m": float(diff.max()), "mean_error_m": float(diff.mean()),
        "filled_bins": int(filled.sum()), "filled_fraction": float(filled.mean()),
        "bins_disagreeing": int((diff > 1e-6).sum()),
        "range_hist_ours": np.histogram(ours[ours < MAX_RANGE_M], bins=10, range=(0, MAX_RANGE_M))[0],
        "range_hist_ref": np.histogram(ref[ref < MAX_RANGE_M], bins=10, range=(0, MAX_RANGE_M))[0],
    }


def save_figure(result: dict, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
    im = axes[0].imshow(result["reference"] / MAX_RANGE_M, vmin=0, vmax=1, cmap="turbo", aspect="auto")
    axes[0].set_title("JEPLO reference projection (metres)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im = axes[1].imshow(result["ours"] / MAX_RANGE_M, vmin=0, vmax=1, cmap="turbo", aspect="auto")
    axes[1].set_title("ours (plugins/jeplo/sensor.py)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].bar(np.arange(10), result["range_hist_ref"], width=0.4, label="reference")
    axes[2].bar(np.arange(10) + 0.4, result["range_hist_ours"], width=0.4, label="ours")
    axes[2].set_title("range histogram")
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xlabel("azimuth bin")
        ax.set_ylabel("elevation bin")
    fig.suptitle(f"{title}  |  max |err| = {result['max_error_m']:.4f} m, "
                 f"{result['bins_disagreeing']} bins differ", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", default=None, help="optional npz with 'hits','origin','quat_wxyz'")
    parser.add_argument("--out", default="outputs/scratch/jeplo_sensor_validation.png")
    args = parser.parse_args(argv)
    if args.hits:
        data = np.load(args.hits)
        hits, origin, quat = data["hits"], data["origin"], tuple(data["quat_wxyz"])
    else:
        hits, origin, quat = synthetic_room_hits()
    result = compare(hits, origin, quat)
    save_figure(result, Path(args.out), "JEPLO spherical depth: reference vs ours (synthetic room)")
    print({k: v for k, v in result.items() if not isinstance(v, np.ndarray)})
    return 0 if result["max_error_m"] < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
