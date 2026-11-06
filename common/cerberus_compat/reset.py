"""Full-environment reset that works with Arena's reset events on Isaac Lab 3.0 GA.

Lab 3.0 GA passes ``env_ids=slice(None)`` to the reset events when ``env.reset()`` is called with no ids. Arena's
``ResetBackgroundPhysics`` (``isaaclab_arena/terms/events.py``, active on any background with nested physics, e.g.
the Lightwheel kitchens) does ``torch.as_tensor(env_ids, dtype=torch.long)`` on it and raises
``TypeError: 'slice' object cannot be interpreted as an integer`` (seen on Arena ``main`` @ 276d97313 + Lab v3.0.0).
Passing an explicit index tensor avoids the slice. Per-episode auto-resets already pass tensors, so only global
resets are affected.

gymnasium wrappers do not all forward extra kwargs: ``OrderEnforcing.reset`` is ``reset(*, seed=None, options=None)``
and raises ``TypeError: OrderEnforcing.reset() got an unexpected keyword argument 'env_ids'`` (seen on the GA pod,
2026-11-06, at arena_spike_go2.py's reset). When that happens the reset goes to the unwrapped env and the wrappers'
own "has been reset" bookkeeping is repaired, otherwise the next ``env.step()`` raises
"Cannot call env.step() before calling env.reset()". Pure python at import time (torch is imported inside).
"""

from __future__ import annotations

from typing import Any


def reset_all(env: Any, **kwargs: Any):
    """``env.reset(env_ids=<all ids>)`` as an explicit device tensor. Returns what ``env.reset`` returns.

    Falls back to a plain ``env.reset(**kwargs)`` when the env has no ``num_envs``/``device`` (fakes, old stacks
    where the plain call is the proven path).
    """
    import torch

    base = getattr(env, "unwrapped", env)
    num_envs = getattr(base, "num_envs", None)
    device = getattr(base, "device", None)
    if num_envs is None or device is None:
        return env.reset(**kwargs)
    ids = torch.arange(int(num_envs), device=device, dtype=torch.long)
    try:
        return env.reset(env_ids=ids, **kwargs)
    except TypeError as e:
        # gymnasium's OrderEnforcing (and anything with a keyword-only reset signature) rejects env_ids.
        if "env_ids" not in str(e) or base is env:
            raise
    out = base.reset(env_ids=ids, **kwargs)
    _mark_wrappers_reset(env, base)
    return out


def _mark_wrappers_reset(env: Any, base: Any) -> None:
    """Tell wrappers wrapping `base` that a reset happened, after bypassing their reset()."""
    wrapper = env
    while wrapper is not None and wrapper is not base:
        if hasattr(wrapper, "_has_reset"):
            wrapper._has_reset = True
        wrapper = getattr(wrapper, "env", None)
