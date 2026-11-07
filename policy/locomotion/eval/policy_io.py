"""Observation plumbing shared by the policy wrappers (pure python).

Isaac Lab's vector-env wrapper returns either a tensor or a ``{group: tensor}`` mapping depending on the
rsl-rl version and how many observation groups the task defines. A policy callable that assumes one
shape breaks on the other with an unhelpful error deep inside a numeric conversion, so the selection is
done once, here, where it is testable without Isaac or torch.
"""

from __future__ import annotations

DEFAULT_GROUP = "policy"


def select_policy_obs(obs, group: str = DEFAULT_GROUP):
    """Return the tensor a policy should consume.

    - mapping with ``group``: that entry;
    - mapping with exactly one entry: it (a task with a single observation group is unambiguous);
    - anything else (tensor, array): returned unchanged.
    """
    if not isinstance(obs, dict):
        return obs
    if group in obs:
        return obs[group]
    if len(obs) == 1:
        return next(iter(obs.values()))
    raise KeyError(
        f"observation mapping has no {group!r} entry and is not single-group "
        f"(groups: {sorted(obs)}); pass the group name explicitly"
    )
