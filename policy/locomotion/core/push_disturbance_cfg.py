"""Shared push-disturbance EventTerm + CurriculumTerm, added to both the flat and
rough Go2 configs. See REFERENCES.md for the method and where these numbers come
from; see curriculum.py for the curriculum formula itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from .curriculum import push_disturbance_curriculum
from .push_impulse_event import apply_push_impulse, clear_expired_push_impulses

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

# Peak values anchored to isaaclab-go2-locomotion's validated Go2 numbers (87.5%
# recovery at 120N peak vs 0% for the untrained baseline) — see REFERENCES.md.
# Torque has no verified source anywhere in the Hwangbo/Lee/Miki lineage or the
# reference repo; this is a reasoned ESTIMATE (peak force x ~0.15m, an
# approximate Go2 body-frame lever arm), not a verified figure.
PEAK_FORCE_N = 120.0
PEAK_TORQUE_NM = 18.0  # ESTIMATE -- see REFERENCES.md

# Interval endpoints: sparse start matches the reference repo's impulse trigger
# interval (6-10s); dense end is our own conservative interpretation, since
# Miki et al.'s exact interval (if published at all) was in supplementary
# material we could not verify -- see REFERENCES.md. Raised from an earlier
# (3.0, 5.0) after a real training run showed poor baseline walking even at
# 0 push force -- pushing every 3-5s at full curriculum difficulty was
# structurally squeezing out the training time needed to consolidate basic
# walking, not just push-recovery specifically.
SPARSE_INTERVAL_RANGE_S = (8.0, 12.0)
DENSE_INTERVAL_RANGE_S = (5.0, 7.0)

# The interval curriculum saturates at DENSE_INTERVAL_RANGE_S once k_c crosses
# this fraction of its own range (0.5 = halfway), instead of continuing to
# interpolate all the way out to k_c=1.0 like force/torque do. Force still
# ramps the full literature-grounded Hwangbo curve to k_c=1.0 -- only the
# interval's approach to its (now less extreme) floor saturates early, so the
# back half of training happens at a stable push frequency instead of one
# that's still slowly changing throughout. Our own design choice, not
# literature-verified -- see REFERENCES.md.
INTERVAL_SATURATION_K_C = 0.5

# How long a single push actually shoves for before self-clearing back to zero force.
# Matches Hwangbo 2019 / isaaclab-go2-locomotion's own "Impulse" duration (0.15-0.25s);
# we use the midpoint. Isaac Lab's `permanent_wrench_composer` (what
# apply_external_force_torque and, previously, our own push terms used) only resets at
# episode reset -- never between an interval EventTerm's triggers -- so without this,
# a "push" was actually a continuous, unending force lasting from one trigger to the
# next (5-12s), not a shove. See REFERENCES.md's "True impulse, not a sustained force"
# section and push_impulse_event.py for the full investigation and fix.
IMPULSE_DURATION_S = 0.2


def add_push_disturbance(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Adds the push_disturbance EventTerm (+ its clear pair) and curriculum to env_cfg
    in place."""
    env_cfg.events.push_disturbance = EventTerm(
        func=apply_push_impulse,
        mode="interval",
        interval_range_s=SPARSE_INTERVAL_RANGE_S,
        params={
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
            "impulse_duration_s": IMPULSE_DURATION_S,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    # Fires every env.step() (degenerate step_dt interval) to zero out any push whose
    # IMPULSE_DURATION_S has elapsed -- see push_impulse_event.py.
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.events.push_disturbance_clear = EventTerm(
        func=clear_expired_push_impulses,
        mode="interval",
        interval_range_s=(step_dt, step_dt),
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    env_cfg.curriculum.push_disturbance_curriculum = CurrTerm(
        func=push_disturbance_curriculum,
        params={
            "event_term_name": "push_disturbance",
            "peak_force": PEAK_FORCE_N,
            "peak_torque": PEAK_TORQUE_NM,
            "sparse_interval_range_s": SPARSE_INTERVAL_RANGE_S,
            "dense_interval_range_s": DENSE_INTERVAL_RANGE_S,
            "interval_saturation_k_c": INTERVAL_SATURATION_K_C,
        },
    )
