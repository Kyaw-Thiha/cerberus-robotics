"""Pure observation-assembly and action-postprocessing logic for the
locomotion_policy inference node -- kept separate from the rclpy node so
it's unit-testable without a live ROS graph. Term order and action
post-processing come from the actual training run's params/env.yaml
(confirmed 2026-09-08 for the Rough candidate) -- do not reorder without
re-checking that file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHAPE_BLIND = "proprioception_only"
SHAPE_PERCEPTIVE = "proprioception_plus_height_scan"


@dataclass
class ProprioSample:
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    velocity_commands: np.ndarray
    joint_pos_rel: np.ndarray
    joint_vel_rel: np.ndarray
    last_action: np.ndarray


def assemble_observation(
    observation_shape: str,
    proprio: ProprioSample,
    height_scan: np.ndarray | None,
) -> np.ndarray:
    """Concatenates observation terms in training's exact order:
    base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
    joint_pos_rel, joint_vel_rel, last_action[, height_scan].
    """
    base_terms = np.concatenate(
        [
            proprio.base_lin_vel,
            proprio.base_ang_vel,
            proprio.projected_gravity,
            proprio.velocity_commands,
            proprio.joint_pos_rel,
            proprio.joint_vel_rel,
            proprio.last_action,
        ]
    ).astype(np.float32)

    if observation_shape == SHAPE_BLIND:
        return base_terms
    elif observation_shape == SHAPE_PERCEPTIVE:
        if height_scan is None:
            raise ValueError(f"observation_shape={observation_shape!r} requires a height_scan array, got None")
        return np.concatenate([base_terms, height_scan.astype(np.float32)])
    else:
        raise ValueError(f"unknown observation_shape {observation_shape!r}, expected {SHAPE_BLIND!r} or {SHAPE_PERCEPTIVE!r}")


def _build_permutation(from_names: list[str], to_names: list[str]) -> np.ndarray:
    """Builds an index array `perm` such that, for an array `a` whose entries
    are ordered like `from_names`, `a[perm]` is reordered to match
    `to_names`. Used to remap joint-state data (`msg.name`'s order) into a
    platform's declared `joint_names` order, and its inverse (via
    `_build_permutation(to_names, from_names)` or `np.argsort(perm)`) to map
    action-vector output back out to `msg.name`'s order.

    Both inputs must contain exactly the same set of names (order may
    differ) -- raises a clear RuntimeError naming both sets otherwise.
    """
    from_set = set(from_names)
    to_set = set(to_names)
    if from_set != to_set or len(from_names) != len(to_names):
        raise RuntimeError(
            "joint name mismatch while building a remapping permutation: "
            f"from_names={list(from_names)!r}, to_names={list(to_names)!r} -- "
            f"only in from_names: {sorted(from_set - to_set)!r}, "
            f"only in to_names: {sorted(to_set - from_set)!r}"
        )
    index_in_from = {name: i for i, name in enumerate(from_names)}
    return np.array([index_in_from[name] for name in to_names], dtype=np.int64)


def apply_action_postprocessing(
    raw_network_output: np.ndarray,
    default_joint_pos: np.ndarray,
    scale: float = 0.25,
) -> np.ndarray:
    """JointPositionAction post-processing: scale the network's raw output
    and add each joint's default position, matching training's
    use_default_offset=true. Never publish raw_network_output directly."""
    return raw_network_output * scale + default_joint_pos
