"""Shared push-disturbance EventTerm + CurriculumTerm, added to both the flat and
rough Go2 configs. See REFERENCES.md for the method and where these numbers come
from; see curriculum.py for the curriculum formula itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from .curriculum import push_disturbance_curriculum

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
# material we could not verify -- see REFERENCES.md.
SPARSE_INTERVAL_RANGE_S = (8.0, 12.0)
DENSE_INTERVAL_RANGE_S = (3.0, 5.0)


def add_push_disturbance(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Adds the push_disturbance EventTerm and its curriculum to env_cfg in place."""
    env_cfg.events.push_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=SPARSE_INTERVAL_RANGE_S,
        params={
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    env_cfg.curriculum.push_disturbance_curriculum = CurrTerm(
        func=push_disturbance_curriculum,
        params={
            "event_term_name": "push_disturbance",
            "peak_force": PEAK_FORCE_N,
            "peak_torque": PEAK_TORQUE_NM,
            "sparse_interval_range_s": SPARSE_INTERVAL_RANGE_S,
            "dense_interval_range_s": DENSE_INTERVAL_RANGE_S,
        },
    )
