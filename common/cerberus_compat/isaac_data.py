"""Isaac Lab 2.x / 3.x data-access shims (pure python; torch not imported)."""

from __future__ import annotations


def as_tensor(x):
    """Return a plain tensor from an Isaac Lab ``.data`` field.

    Lab 3.0 returns a ProxyArray exposing ``.torch``; Lab 2.x returns tensors.
    Anything without a ``torch`` attribute (tensors, arrays, tuples) passes through.
    """
    proxy = getattr(x, "torch", None)
    if proxy is not None and not callable(proxy):
        return proxy
    return x


def physx_view(asset):
    """``asset.root_view`` (Lab 3.0) if present, else ``asset.root_physx_view`` (Lab 2.x)."""
    view = getattr(asset, "root_view", None)
    if view is not None:
        return view
    return asset.root_physx_view
