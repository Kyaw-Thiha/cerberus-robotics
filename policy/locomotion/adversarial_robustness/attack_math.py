"""Phase 0.5 gate arithmetic: perturbation bounds, rate limiting, magnitudes, reward, gate decision.

Everything here is pure python so it can be tested without Isaac, and so the *definitions* the gate
rests on are reviewable in one place. The rules come from ``plans/handoff_phase0.5.md``:

  * the adversary emits a base push force (x/y) and a velocity-command override;
  * its output is rate-limited to 0.1x the channel's max range per simulation timestep (paper value);
  * its reward is the paper's Eq. 2-3 form (alive penalty against the robot, orientation / shake /
    torque-limit auxiliaries, minus a Lipschitz penalty on the network's layer weights);
  * the gate is *relative*: the minimum combined magnitude that induces a fall must exceed the push
    range the candidate was trained and evaluated against -- no absolute newton figure is copied from
    the paper, because 15 N / +-0.5 m/s were chosen for a ~30-50 kg ANYmal, not a ~15 kg Go2.

Magnitude convention (the one thing the spec leaves open, so it is defined here explicitly and used
everywhere downstream): every channel is normalised by the range the policy was trained against, and
the combined magnitude is the Euclidean norm of the per-channel normalised magnitudes.

    force_norm = |F_xy| / F_peak_n          (F_peak_n = the Hwangbo push curriculum's peak)
    cmd_norm   = |dcmd_xy| / cmd_peak_mps   (command space the policy was trained to track)
    combined   = hypot(force_norm, cmd_norm)

So ``combined == 1.0`` means "as strong as the trained push, combined with a command deflection as
large as the trained command range" -- the boundary the gate compares against. The raw newtons and
m/s are always reported alongside, because a normalised number alone is not inspectable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------
# Bounds -- read from config, never hardcoded at the call site
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PerturbationBounds:
    """What the adversary is allowed to ask for, expressed in the policy's own trained units."""

    force_peak_n: float
    command_peak_mps: float
    yaw_peak_rad_s: float = 0.0
    force_peak_torque_nm: float | None = None
    rate_limit_fraction: float = 0.1
    """Per-timestep cap as a fraction of the channel's max range. 0.1 is the paper's value."""

    def __post_init__(self):
        for name in ("force_peak_n", "command_peak_mps"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
        if not 0 < self.rate_limit_fraction <= 1.0:
            raise ValueError(f"rate_limit_fraction must be in (0, 1], got {self.rate_limit_fraction!r}")

    @property
    def max_force_delta_per_step(self) -> float:
        return self.force_peak_n * self.rate_limit_fraction

    @property
    def max_command_delta_per_step(self) -> float:
        return self.command_peak_mps * self.rate_limit_fraction

    @classmethod
    def from_config(cls, platform_cfg, command_ranges, *, rate_limit_fraction: float = 0.1):
        """Build from the platform's push curriculum + the task's trained command ranges.

        ``platform_cfg``: the PlatformCfg instance (``GO2``) -- its ``push_peak_force_n`` is the peak
        the Hwangbo curriculum ramps to, i.e. the largest push the policy ever trained against.
        ``command_ranges``: ``env_cfg.commands.base_velocity.ranges`` (or an equivalent mapping).
        """
        peak_force = getattr(platform_cfg, "push_peak_force_n", None)
        if peak_force is None:
            raise ValueError("platform_cfg has no push_peak_force_n: cannot derive the trained push range")
        lin_x = getattr(command_ranges, "lin_vel_x", None)
        lin_y = getattr(command_ranges, "lin_vel_y", None)
        ang_z = getattr(command_ranges, "ang_vel_z", None)
        if lin_x is None or lin_y is None:
            raise ValueError("command_ranges is missing lin_vel_x / lin_vel_y")
        # The trained command envelope: the largest |(vx, vy)| a sample can take is the norm of the
        # per-axis maxima (corner of the box). That is the honest denominator for "as large as the
        # command range the policy was trained with".
        command_peak = math.hypot(max(abs(lin_x[0]), abs(lin_x[1])), max(abs(lin_y[0]), abs(lin_y[1])))
        yaw_peak = max(abs(ang_z[0]), abs(ang_z[1])) if ang_z is not None else 0.0
        return cls(
            force_peak_n=float(peak_force),
            command_peak_mps=float(command_peak),
            yaw_peak_rad_s=float(yaw_peak),
            force_peak_torque_nm=getattr(platform_cfg, "push_peak_torque_nm", None),
            rate_limit_fraction=rate_limit_fraction,
        )


# --------------------------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------------------------


@dataclass
class RateLimitedChannel:
    """One 2-D perturbation channel (force or command) with a per-step rate cap.

    The adversary's raw output is *not* applied directly: it is clamped to the channel's range and to
    the per-step delta cap. This is what keeps a found attack physical rather than a teleport -- with
    the Lipschitz penalty on the network, it is what stops degenerate bang-bang outputs.
    """

    peak: float
    max_delta_per_step: float
    state: list = field(default_factory=lambda: [0.0, 0.0])

    def __post_init__(self):
        if self.peak <= 0:
            raise ValueError(f"peak must be positive, got {self.peak}")
        if not 0 < self.max_delta_per_step <= self.peak:
            raise ValueError(
                f"max_delta_per_step must be in (0, {self.peak}], got {self.max_delta_per_step}"
            )
        if len(self.state) != 2:
            raise ValueError("channel state must be 2-D")

    def step(self, raw_xy) -> list:
        """Apply a raw adapter output -> clamped, rate-limited 2-D value; updates ``state``."""
        out = rate_limited_step(
            self.state, raw_xy, peak=self.peak, max_delta_per_step=self.max_delta_per_step
        )
        self.state = out
        return list(out)

    def reset(self) -> None:
        self.state = [0.0, 0.0]


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else min(value, high)


def rate_limited_step(prev_xy, raw_xy, *, peak: float, max_delta_per_step: float) -> list:
    """One step of a 2-D channel: clamp to ``peak``, then cap the move from ``prev_xy``.

    Pure python (lists/floats) so the exact semantics are testable without torch. The wrapper calls
    it per env with that env's own previous value: rate limiting is per-env state, not global.
    """
    if len(prev_xy) != 2 or len(raw_xy) != 2:
        raise ValueError(f"expected 2-D prev and raw values, got {len(prev_xy)} and {len(raw_xy)}")
    out = []
    for axis in range(2):
        target = _clip(float(raw_xy[axis]), -peak, peak)
        delta = _clip(target - float(prev_xy[axis]), -max_delta_per_step, max_delta_per_step)
        out.append(float(prev_xy[axis]) + delta)
    return out


# --------------------------------------------------------------------------------------------
# Magnitude (the quantity the gate compares)
# --------------------------------------------------------------------------------------------


def normalised_magnitude(force_xy, command_xy, bounds: PerturbationBounds) -> dict:
    """Per-channel normalised magnitudes, the combined score, and the raw values."""
    force_mag = math.hypot(float(force_xy[0]), float(force_xy[1]))
    command_mag = math.hypot(float(command_xy[0]), float(command_xy[1]))
    force_norm = force_mag / bounds.force_peak_n
    command_norm = command_mag / bounds.command_peak_mps
    return {
        "force_n": force_mag,
        "command_mps": command_mag,
        "force_norm": force_norm,
        "command_norm": command_norm,
        "combined": math.hypot(force_norm, command_norm),
    }


def modality_evidence(force_xy, command_xy, bounds: PerturbationBounds) -> dict:
    """Magnitudes split per modality, for the report's "single modality fails, combination succeeds"."""
    per_channel = normalised_magnitude(force_xy, command_xy, bounds)
    return {
        "combined": per_channel["combined"],
        "force_only": abs(per_channel["force_norm"]),
        "command_only": abs(per_channel["command_norm"]),
        **per_channel,
    }


# --------------------------------------------------------------------------------------------
# Adversary reward (paper Eq. 2-3)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdversaryRewardConfig:
    """Weights for the paper's Eq. 2-3.

    The paper states the *structure* and the sign conventions but not these coefficients; the values
    below are OUR calibration choices for Go2, and the report must label them as such rather than
    implying they are from the paper.
    """

    c_orient: float = 1.0
    c_shake: float = 0.05
    c_torque: float = 0.1
    alive_penalty: float = 1.0
    lipschitz_lambda: float = 1e-4


DEFAULT_REWARD_CONFIG = AdversaryRewardConfig()


def adversary_reward(
    *,
    alive: bool,
    projected_gravity_z: float,
    base_angular_xy,
    joint_torques,
    joint_torque_limits,
    layer_inf_norms=(),
    config: AdversaryRewardConfig = DEFAULT_REWARD_CONFIG,
) -> dict:
    """Paper Eq. 2-3: ``-1_alive + r_aux - lambda * sum(||theta_i||_inf)``, broken into its parts.

    ``alive`` True means the robot has NOT failed: that is what is penalised, so a surviving robot
    yields a negative reward and the adversary is pushed toward causing a fall.
    ``projected_gravity_z``: +1 lying, -1 upright -- causing a fall pushes it up.
    Termination/fall definition is not redefined here: the caller passes the env's own ``alive``.
    """
    if len(joint_torques) != len(joint_torque_limits):
        raise ValueError(
            f"joint_torques ({len(joint_torques)}) and joint_torque_limits ({len(joint_torque_limits)}) differ"
        )
    orient = config.c_orient * float(projected_gravity_z)
    shake = config.c_shake * (float(base_angular_xy[0]) ** 2 + float(base_angular_xy[1]) ** 2)
    torque_excess = sum(
        max(0.0, abs(float(t)) / limit - 1.0) for t, limit in zip(joint_torques, joint_torque_limits, strict=True)
    )
    torque = config.c_torque * torque_excess
    lipschitz = config.lipschitz_lambda * sum(abs(float(w)) for w in layer_inf_norms)
    r_aux = orient + shake + torque
    alive_term = -config.alive_penalty if alive else 0.0
    return {
        "reward": alive_term + r_aux - lipschitz,
        "alive_term": alive_term,
        "r_aux": r_aux,
        "orient": orient,
        "shake": shake,
        "torque": torque,
        "lipschitz": lipschitz,
    }


# --------------------------------------------------------------------------------------------
# Reduce-until-survives, and the gate decision
# --------------------------------------------------------------------------------------------


def find_minimum_inducing(
    evaluate,
    *,
    max_scale: float,
    fall_threshold: float = 0.5,
    min_scale: float = 0.0,
    tolerance: float = 0.02,
    max_iterations: int = 12,
) -> dict:
    """Smallest perturbation scale whose fall rate reaches ``fall_threshold`` (paper Sec. VI-B).

    ``evaluate(scale) -> Mapping`` must return at least ``{"fall_rate": float, "trials": int}``.
    The search is a bisection on the scale, which assumes the fall rate is (weakly) monotone in the
    scale -- true by construction here, since a larger allowed perturbation contains every smaller
    one. Returns ``{"found": False}`` when even ``max_scale`` does not induce enough falls: that means
    the adversary failed to attack the policy at all, which is itself a gate-relevant result.
    """
    if max_scale <= 0:
        raise ValueError(f"max_scale must be positive, got {max_scale}")
    if not 0 <= fall_threshold <= 1:
        raise ValueError(f"fall_threshold must be in [0, 1], got {fall_threshold}")

    top = evaluate(max_scale)
    if float(top.get("fall_rate", 0.0)) < fall_threshold:
        return {
            "found": False,
            "reason": "adversary did not reach the fall threshold at the maximum allowed scale",
            "max_scale": max_scale,
            "best_fall_rate": float(top.get("fall_rate", 0.0)),
            "trials": int(top.get("trials", 0)),
        }
    low_result = evaluate(min_scale)
    if float(low_result.get("fall_rate", 0.0)) >= fall_threshold:
        return {
            "found": True,
            "scale": min_scale,
            "fall_rate": float(low_result["fall_rate"]),
            "trials": int(low_result.get("trials", 0)),
            "iterations": 0,
            "note": "even the minimum scale induces failure -- the adversary is not the limiting factor",
        }

    low, high = min_scale, max_scale
    history = []
    boundary_result = top
    for iteration in range(1, max_iterations + 1):
        mid = (low + high) / 2.0
        result = evaluate(mid)
        history.append({"iteration": iteration, "scale": mid, **{k: result.get(k) for k in ("fall_rate", "trials")}})
        if float(result.get("fall_rate", 0.0)) >= fall_threshold:
            high = mid
            boundary_result = result
        else:
            low = mid
        if high - low <= tolerance:
            break
    return {
        "found": True,
        "scale": high,
        "fall_rate": float(boundary_result.get("fall_rate", 0.0)),
        "trials": int(boundary_result.get("trials", 0)),
        "iterations": len(history),
        "history": history,
        "tolerance": tolerance,
    }


def gate_decision(
    minimum_inducing_score, trained_range_score: float = 1.0, searched_max_scale: float | None = None
) -> dict:
    """PASS iff the minimum inducing magnitude strictly exceeds the trained range.

    Strictly: a boundary exactly at the trained range means the attack works at the strength the
    policy was trained and evaluated against, which is the failure the gate exists to catch.

    ``minimum_inducing_score`` of None means the adversary never broke the policy at any scale that was
    tried. That is not automatically a failure -- if the largest tried scale exceeds the trained range,
    the boundary is above it, which is exactly what a pass requires. It is reported as a lower bound so
    the report cannot present a search limit as a measured boundary.
    """
    if minimum_inducing_score is None:
        if searched_max_scale is not None and searched_max_scale > trained_range_score:
            return {
                "verdict": "PASS",
                "reason": (
                    f"no fall induced up to {searched_max_scale:g}x the trained range, so the boundary lies "
                    "above it (lower bound, not a measured boundary)"
                ),
                "minimum_inducing_score": None,
                "searched_max_scale": searched_max_scale,
                "boundary_is_lower_bound": True,
                "trained_range_score": trained_range_score,
                "margin": None,
            }
        return {
            "verdict": "FAIL",
            "reason": "no fall induced, and the search never exceeded the trained range: inconclusive",
            "trained_range_score": trained_range_score,
            "searched_max_scale": searched_max_scale,
        }
    value = float(minimum_inducing_score)
    passed = value > trained_range_score
    return {
        "verdict": "PASS" if passed else "FAIL",
        "minimum_inducing_score": value,
        "trained_range_score": trained_range_score,
        "margin": value - trained_range_score,
        "relative_margin": (value - trained_range_score) / trained_range_score if trained_range_score else None,
        "reason": (
            "minimum attacking magnitude exceeds the trained push range"
            if passed
            else "the adversary can break the policy at or below the trained push range"
        ),
    }


def summarise_modalities(results: dict) -> dict:
    """Per-modality gate verdicts -> the report's headline claim.

    ``results``: {"force_only": {"minimum_score": float|None, "fall_rate": ...}, ...}
    Expected shape of the source-paper finding: single modalities fail to break a push-randomised
    policy, the combination succeeds.
    """
    out = {}
    for name, payload in results.items():
        score = payload.get("minimum_score")
        out[name] = {
            "minimum_score": score,
            "breaks_policy": bool(score is not None and score <= 1.0),
            "detail": payload,
        }
    singles = [name for name in ("force_only", "command_only") if name in out]
    out["combination_succeeds_where_singles_fail"] = bool(
        singles and all(not out[name]["breaks_policy"] for name in singles) and out.get("combined", {}).get("breaks_policy")
    )
    return out


# --------------------------------------------------------------------------------------------
# Modalities
# --------------------------------------------------------------------------------------------

MODALITIES = ("force_only", "command_only", "combined")
"""The three attack configurations the gate report must show side by side.

The source paper's Table IV finds that a policy trained with standard push-domain randomisation
resists single-modality attacks and is broken by combinations. Running all three is what lets the
report say whether that holds for this checkpoint, rather than assuming it.
"""


def channels_for(modality: str) -> frozenset:
    """Which channels a modality may perturb. Disabled channels are driven to zero by the wrapper."""
    if modality == "force_only":
        return frozenset({"force"})
    if modality == "command_only":
        return frozenset({"command"})
    if modality == "combined":
        return frozenset({"force", "command"})
    raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}")


def scale_bounds(bounds: PerturbationBounds, scale: float) -> PerturbationBounds:
    """Shrink (or grow) every channel's allowed range by ``scale``, keeping the rate fraction.

    This is the "reduce-until-survives" knob from the paper's Section VI-B: the adversary is
    retrained/rolled out with a smaller allowed perturbation, and the smallest scale that still
    breaks the policy is the boundary the gate compares against the trained range.
    """
    if not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    return PerturbationBounds(
        force_peak_n=bounds.force_peak_n * scale,
        command_peak_mps=bounds.command_peak_mps * scale,
        yaw_peak_rad_s=bounds.yaw_peak_rad_s * scale,
        force_peak_torque_nm=bounds.force_peak_torque_nm,
        rate_limit_fraction=bounds.rate_limit_fraction,
    )
