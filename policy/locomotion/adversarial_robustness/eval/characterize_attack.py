#!/usr/bin/env python3
"""Characterise a trained adversarial attack: what it actually does to the robot, step by step.

The gate answers "at what magnitude does the candidate break". This answers "what is the attack doing
there" -- the time series of the command override and the base force, how fast it switches, how much it
reverses, and whether the resulting failures are genuine base-contact terminations or an artefact of
some other termination term.

Eval only: the adversary is frozen, the policy is frozen, nothing is trained.

    $ISAACLAB_PATH/isaaclab.sh -p policy/locomotion/adversarial_robustness/eval/characterize_attack.py \
        --checkpoint policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt \
        --adversary_dir policy/locomotion/checkpoints/adversary_20261108T051752Z \
        --scales 0.568,0.629 --modalities combined,command_only --envs 100 --steps 400

Writes `<out>/characterization.json` and `<out>/timeseries.png`, plus per-env CSVs when --csv is given.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUT = Path(__file__).resolve().parent / "results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0")
    parser.add_argument("--checkpoint", required=True, help="Frozen locomotion candidate.")
    parser.add_argument("--adversary_dir", required=True, help="Trained adversary run directory.")
    parser.add_argument("--scales", default="0.568,0.629", help="Comma-separated attack scales to record.")
    parser.add_argument("--modalities", default="combined,command_only")
    parser.add_argument("--envs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=400, help="Steps per episode (400 = 8 s at 50 Hz).")
    parser.add_argument("--deadband_mps", type=float, default=0.05, help="Ignore command sign changes below this.")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--csv", action="store_true", help="Also write the raw per-env time series as CSV.")
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record each condition as an mp4 (single env) with Isaac Lab's own recorder, for visual "
        "confirmation that the failures are real falls.",
    )
    from common.cerberus_compat.launcher import add_launcher_args

    add_launcher_args(parser)
    parser.set_defaults(headless=True, enable_cameras=True)
    return parser.parse_args(argv)


class RecordingHooks:
    """Wrap ManagerBasedHooks and keep the last applied force/command for the time series."""

    def __init__(self, inner):
        self.inner = inner
        self.last_force = None
        self.last_command = None

    def apply_force(self, force_xy):
        self.inner.apply_force(force_xy)
        self.last_force = force_xy.detach().clone()

    def apply_command(self, command_xy):
        self.inner.apply_command(command_xy)
        self.last_command = command_xy.detach().clone()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def series_stats(series, deadband: float, dt: float) -> dict:
    """Amplitude / switching statistics for one channel of one env.

    ``series`` is a list of 2-element rows (x, y) in physical units. Sign changes are counted per axis
    with a deadband so sensor-level jitter is not mistaken for switching; a direction reversal is a step
    whose dot product with the previous vector is negative (the command actually turned around).
    """
    import math

    if not series:
        return {}
    xs = [row[0] for row in series]
    ys = [row[1] for row in series]
    mags = [math.hypot(row[0], row[1]) for row in series]

    def flips(values):
        count = 0
        last_sign = 0
        for value in values:
            if abs(value) < deadband:
                continue
            sign = 1 if value > 0 else -1
            if last_sign and sign != last_sign:
                count += 1
            last_sign = sign
        return count

    reversals = 0
    for prev, cur in itertools.pairwise(series):
        dot = prev[0] * cur[0] + prev[1] * cur[1]
        if dot < 0 and (math.hypot(*prev) > deadband) and (math.hypot(*cur) > deadband):
            reversals += 1

    duration = len(series) * dt
    flips_x = flips(xs)
    flips_y = flips(ys)
    return {
        "steps": len(series),
        "duration_s": duration,
        "mean_abs_x": sum(abs(v) for v in xs) / len(xs),
        "mean_abs_y": sum(abs(v) for v in ys) / len(ys),
        "max_abs_x": max(abs(v) for v in xs),
        "max_abs_y": max(abs(v) for v in ys),
        "rms": math.sqrt(sum(m * m for m in mags) / len(mags)),
        "sign_flips_x": flips_x,
        "sign_flips_y": flips_y,
        "sign_flips_per_s": (flips_x + flips_y) / duration if duration else 0.0,
        "direction_reversals": reversals,
        "direction_reversals_per_s": reversals / duration if duration else 0.0,
    }


def _summarise(per_env, key):
    """Mean/min/max over envs of a per-env statistic, rounded for the report."""
    values = [e[key] for e in per_env if key in e and e[key] is not None]
    if not values:
        return None
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def main(argv=None) -> int:
    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # noqa: F841
    try:
        return _run(args)
    except Exception:  # noqa: BLE001 -- SimulationApp.close() exits 0; a traceback is the only evidence
        import traceback

        traceback.print_exc()
        return 1


def _run(args) -> int:
    import torch

    from policy.locomotion.adversarial_robustness import (
        attack_env,
        train_adversary,
    )

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    results = {}

    for modality in modalities:
        for scale in scales:
            key = f"{modality}@{scale:g}"
            print(f"\n[characterize] {key}: {args.envs} envs, {args.steps} steps")
            video_cfg = _video_configurer(args, out_dir, key) if args.video else None
            wrapper = attack_env.build_attack_env(
                args.task,
                policy=None,
                bounds=_bounds(),
                num_envs=args.envs,
                modality=modality,
                play=True,
                configure_env_cfg=video_cfg,
            )
            hooks = RecordingHooks(wrapper.hooks)
            wrapper.hooks = hooks
            wrapper.scale = scale
            # The frozen locomotion policy, loaded against this env, then installed on the wrapper.
            wrapper.policy = train_adversary.load_frozen_policy(args.task, args.checkpoint)
            adversary = train_adversary.load_adversary_policy(wrapper.env, args.adversary_dir)

            raw_env = wrapper.env.unwrapped
            dt = float(raw_env.step_dt)
            termination_terms = list(raw_env.termination_manager.active_terms)

            force_series = [[] for _ in range(args.envs)]
            command_series = [[] for _ in range(args.envs)]
            term_hits = {name: 0 for name in termination_terms}
            first_failure_step = [None] * args.envs

            obs = wrapper.reset()
            finished = torch.zeros(args.envs, dtype=torch.bool, device=wrapper.device)
            for step in range(args.steps):
                with torch.no_grad():
                    action = adversary(obs)
                obs, _reward, dones, extras = wrapper.step(action)
                if hooks.last_force is not None:
                    for env_index, row in enumerate(hooks.last_force.detach().cpu().tolist()):
                        force_series[env_index].append(row)
                if hooks.last_command is not None:
                    for env_index, row in enumerate(hooks.last_command.detach().cpu().tolist()):
                        command_series[env_index].append(row)
                time_outs = extras.get("time_outs")
                terminated = dones.bool() & (~time_outs.bool() if time_outs is not None else True)
                if bool(terminated.any()):
                    for name in termination_terms:
                        fired = raw_env.termination_manager.get_term(name)
                        hits = (fired.bool() & terminated).sum().item()
                        term_hits[name] += int(hits)
                    for env_index in terminated.nonzero(as_tuple=True)[0].tolist():
                        if first_failure_step[env_index] is None:
                            first_failure_step[env_index] = step
                finished |= dones.bool()
                if bool(finished.all()):
                    break

            per_env = []
            for env_index in range(args.envs):
                row = {"env": env_index, "failed_at_step": first_failure_step[env_index]}
                row["force"] = series_stats(force_series[env_index], deadband=1.0, dt=dt)
                row["command"] = series_stats(command_series[env_index], deadband=args.deadband_mps, dt=dt)
                per_env.append(row)

            results[key] = {
                "modality": modality,
                "scale": scale,
                "num_envs": args.envs,
                "steps": args.steps,
                "dt_s": dt,
                "termination_hits": term_hits,
                "envs_failed": sum(1 for e in per_env if e["failed_at_step"] is not None),
                "command_stats": {
                    metric: _summarise([e["command"] for e in per_env], metric)
                    for metric in (
                        "mean_abs_x", "mean_abs_y", "max_abs_x", "max_abs_y", "rms",
                        "sign_flips_x", "sign_flips_y", "sign_flips_per_s",
                        "direction_reversals", "direction_reversals_per_s", "duration_s",
                    )
                },
                "force_stats": {
                    metric: _summarise([e["force"] for e in per_env], metric)
                    for metric in ("mean_abs_x", "mean_abs_y", "max_abs_x", "max_abs_y", "rms", "duration_s")
                },
                "examples": _examples(per_env, force_series, command_series),
                "per_env": per_env if args.csv else None,
            }
            print(f"[characterize] {key}: {results[key]['envs_failed']}/{args.envs} failed, terms={term_hits}")
            print(f"[characterize] {key}: command sign flips/s "
                  f"{results[key]['command_stats']['sign_flips_per_s']}, "
                  f"reversals/s {results[key]['command_stats']['direction_reversals_per_s']}")
            wrapper.env.close()

    (out_dir / "characterization.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(f"[characterize] wrote {out_dir / 'characterization.json'}")
    _plot(results, out_dir / "timeseries.png")
    return 0


def _examples(per_env, force_series, command_series, count: int = 2):
    """Raw time series for a few representative envs: failures first, then survivors."""
    failed = [e["env"] for e in per_env if e["failed_at_step"] is not None][:count]
    survived = [e["env"] for e in per_env if e["failed_at_step"] is None][:count]
    out = []
    for env_index in failed + survived:
        out.append(
            {
                "env": env_index,
                "failed_at_step": per_env[env_index]["failed_at_step"],
                "force": [[round(v, 4) for v in row] for row in force_series[env_index]],
                "command": [[round(v, 4) for v in row] for row in command_series[env_index]],
            }
        )
    return out


def _video_configurer(args, out_dir, tag):
    """Isaac Lab 3.0 records through env_cfg.video_recorders, which must be set before gym.make."""
    from types import SimpleNamespace

    from policy.locomotion.core.script_utils import (
        configure_video_recorders,
    )

    video_args = SimpleNamespace(video=True, video_length=int(args.steps), enable_cameras=True, headless=True)

    def configure(env_cfg):
        ok = configure_video_recorders(env_cfg, video_args, str(out_dir / "videos"), phase=tag)
        print(f"[characterize] video recorder configured for {tag}: {ok}")

    return configure


def _bounds():
    from policy.locomotion.adversarial_robustness import attack_math
    from policy.locomotion.core.platforms.go2.platform import GO2

    class _Ranges:  # the trained command envelope; the Play task's own ranges are not needed here
        lin_vel_x = (-1.0, 1.0)
        lin_vel_y = (-1.0, 1.0)
        ang_vel_z = (-1.0, 1.0)

    return attack_math.PerturbationBounds.from_config(GO2, _Ranges)


def _plot(results: dict, path: Path) -> None:
    """Command and force time series for one representative env per condition, plus flips per episode."""
    try:
        import matplotlib
    except ImportError:
        print("[characterize] matplotlib not installed on the pod; skipping the plot")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with_series = {k: v for k, v in results.items() if v.get("examples")}
    if not with_series:
        return
    fig, axes = plt.subplots(len(with_series), 3, figsize=(15, 3.4 * len(with_series)), squeeze=False)
    for row, (key, data) in enumerate(with_series.items()):
        example = data["examples"][0]
        steps = list(range(len(example["command"])))
        fail = example["failed_at_step"]
        ax = axes[row][0]
        ax.set_title(f"{key}: command override (env {example['env']})")
        ax.plot(steps, [c[0] for c in example["command"]], label="cmd x")
        ax.plot(steps, [c[1] for c in example["command"]], label="cmd y")
        ax.axhline(0, color="grey", lw=0.5)
        if fail is not None:
            ax.axvline(fail, color="red", lw=1, label=f"fall @ {fail}")
        ax.set_xlabel("step")
        ax.set_ylabel("m/s")
        ax.legend(fontsize=7)

        ax = axes[row][1]
        ax.set_title(f"{key}: base force magnitude")
        ax.plot(steps, [math.hypot(f[0], f[1]) for f in example["force"]])
        if fail is not None:
            ax.axvline(fail, color="red", lw=1)
        ax.set_xlabel("step")
        ax.set_ylabel("N")

        ax = axes[row][2]
        ax.set_title("sign flips per second across envs")
        flips = [e["sign_flips_per_s"] for e in data["per_env"]] if data.get("per_env") else []
        if flips:
            ax.hist(flips, bins=20)
            ax.axvline(sum(flips) / len(flips), color="red", lw=1, label="mean")
            ax.legend(fontsize=7)
        ax.set_xlabel("flips / s")
        ax.set_ylabel("envs")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[characterize] wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
