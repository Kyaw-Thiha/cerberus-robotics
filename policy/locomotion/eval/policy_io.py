"""Observation plumbing shared by the policy wrappers (pure python).

Isaac Lab's vector-env wrapper returns either a tensor or a ``{group: tensor}`` mapping depending on the
rsl-rl version and how many observation groups the task defines. A policy callable that assumes one
shape breaks on the other with an unhelpful error deep inside a numeric conversion, so the selection is
done once, here, where it is testable without Isaac or torch.
"""

from __future__ import annotations

DEFAULT_GROUP = "policy"


def _is_group_container(obs) -> bool:
    """True for a mapping of observation groups.

    A plain ``dict`` is the obvious case; rsl-rl 5 hands over a ``TensorDict``, which is not a ``dict``
    subclass but does expose ``keys()`` and a ``batch_size``. Both are treated the same way here --
    ``.numpy()`` on a TensorDict returns a plain dict, which is how a policy wrapper ends up calling
    array methods on a mapping.
    """
    return isinstance(obs, dict) or (hasattr(obs, "keys") and hasattr(obs, "batch_size"))


def select_policy_obs(obs, group: str = DEFAULT_GROUP):
    """Return the tensor a policy should consume.

    - mapping (dict or TensorDict) with ``group``: that entry;
    - mapping with exactly one entry: it (a single-group observation space is unambiguous);
    - nested mappings: unwrapped repeatedly (a wrapper may nest its groups);
    - anything else (tensor, array): returned unchanged.
    """
    seen = []
    while _is_group_container(obs):
        keys = list(obs.keys())
        seen.append(keys)
        if group in keys:
            obs = obs[group]
            continue
        if len(keys) == 1:
            obs = obs[keys[0]]
            continue
        raise KeyError(
            f"observation mapping has no {group!r} entry and is not single-group "
            f"(nesting so far: {seen}); pass the group name explicitly"
        )
    if hasattr(obs, "batch_size"):
        raise TypeError(
            f"observation still holds groups ({sorted(obs.keys())}) after unwrapping; "
            "the policy needs a single tensor"
        )
    return obs
