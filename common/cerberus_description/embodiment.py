"""Robot embodiment contract for the world layer.

Pure python on purpose: the articulation config is referenced by a lazy
"module:ATTR" string so importing this file never pulls in Isaac Lab.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmbodimentCfg:
    """A robot body (legs, arm, or a composition) as the world layer sees it."""

    name: str  # e.g. "go2"
    articulation_cfg_entry_point: str  # lazy "module:ATTR"
    joint_groups: Mapping[str, tuple[str, ...]]  # group -> joint NAMES
    checkpoint_prefix: str = ""  # defaults to name

    def __post_init__(self) -> None:
        if not self.checkpoint_prefix:
            # frozen dataclass: bypass the frozen __setattr__ for the default.
            object.__setattr__(self, "checkpoint_prefix", self.name)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """All joints, in group order."""
        return tuple(j for group in self.joint_groups.values() for j in group)

    def resolve(self) -> Any:
        """Import and return the ArticulationCfg object (deferred so the sim
        is only needed when actually building an env)."""
        module_name, sep, attr = self.articulation_cfg_entry_point.partition(":")
        if not sep or not module_name or not attr:
            raise ValueError(
                f"Bad entry point {self.articulation_cfg_entry_point!r}; expected 'module:ATTR'."
            )
        return getattr(importlib.import_module(module_name), attr)


def compose_embodiments(
    name: str,
    legs: EmbodimentCfg,
    arm: EmbodimentCfg,
    articulation_cfg_entry_point: str,
) -> EmbodimentCfg:
    """Merge a legged base and an arm into one embodiment (e.g. go2_arx5).

    Joint groups are kept as authored, so a multi-group arm such as
    arm+gripper survives composition. Joint names must be disjoint, since
    policies address joints by name.
    """
    clash = sorted(set(legs.joint_names) & set(arm.joint_names))
    if clash:
        raise ValueError(f"Joint name collision between legs and arm: {clash}")
    return EmbodimentCfg(
        name=name,
        articulation_cfg_entry_point=articulation_cfg_entry_point,
        joint_groups={**legs.joint_groups, **arm.joint_groups},
        checkpoint_prefix=name,
    )
