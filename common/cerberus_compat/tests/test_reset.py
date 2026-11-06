"""Tests for common.cerberus_compat.reset (no Isaac; torch is stubbed per test when absent).

`reset_all` imports torch inside the function, so a stub installed in ``sys.modules`` for the duration of a single
test is enough. It must be scoped with ``monkeypatch``: assigning ``sys.modules["torch"]`` at import time leaks into
every later test module in the same pytest session (it broke 7 torch-using tests on 2026-11-06).
"""

import sys
import types

import pytest

from common.cerberus_compat.reset import reset_all

try:  # real torch on a pod-like env
    import torch as _real_torch
except ImportError:  # the default pixi env has none
    _real_torch = None


class _StubTensor(list):
    """Only what ``reset_all`` and these assertions touch."""

    dtype = "long"

    def tolist(self):
        return list(self)


@pytest.fixture
def torch_mod(monkeypatch):
    """Real torch when installed, otherwise a minimal stub scoped to this test."""
    if _real_torch is not None:
        return _real_torch
    mod = types.ModuleType("torch")
    mod.Tensor = _StubTensor
    mod.long = "long"
    mod.arange = lambda n, device=None, dtype=None: _StubTensor(range(int(n)))
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


class _Base:
    num_envs = 3
    device = "cpu"


class _Env:
    unwrapped = _Base()

    def __init__(self):
        self.calls = []

    def reset(self, **kwargs):
        self.calls.append(kwargs)
        return "obs", {}


def test_passes_explicit_id_tensor(torch_mod):
    env = _Env()
    assert reset_all(env) == ("obs", {})
    ids = env.calls[0]["env_ids"]
    assert isinstance(ids, torch_mod.Tensor) and ids.dtype == "long"
    assert ids.tolist() == [0, 1, 2]


def test_forwards_kwargs(torch_mod):
    env = _Env()
    reset_all(env, seed=7)
    assert env.calls[0]["seed"] == 7


def test_plain_reset_when_env_has_no_size(torch_mod):
    class Bare:
        def __init__(self):
            self.calls = []

        def reset(self, **kwargs):
            self.calls.append(kwargs)

    env = Bare()
    reset_all(env, seed=1)
    assert env.calls == [{"seed": 1}]


def test_falls_back_to_unwrapped_when_wrapper_rejects_env_ids(torch_mod):
    """gymnasium's OrderEnforcing.reset(*, seed=None, options=None) rejects env_ids (GA pod, 2026-11-06)."""

    class Enforcing:
        _has_reset = False

        def __init__(self, inner):
            self.env = inner
            self.unwrapped = inner  # gym's env.unwrapped is the raw env that accepts env_ids

        def reset(self, *, seed=None, options=None):
            raise TypeError("Enforcing.reset() got an unexpected keyword argument 'env_ids'")

    base = _Env()
    base.num_envs, base.device = 3, "cpu"  # the raw env carries these, as Isaac Lab's does
    env = Enforcing(base)
    assert reset_all(env) == ("obs", {})
    assert base.calls[0]["env_ids"].tolist() == [0, 1, 2]
    assert env._has_reset is True  # step() would otherwise refuse to run


def test_reraises_other_type_errors(torch_mod):
    class Broken:
        unwrapped = _Base()

        def reset(self, **kwargs):
            raise TypeError("something else entirely")

    with pytest.raises(TypeError, match="something else"):
        reset_all(Broken())
