"""Tests for common.cerberus_compat.reset (torch only, no Isaac)."""

import pytest

torch = pytest.importorskip("torch")

from common.cerberus_compat.reset import reset_all  # noqa: E402


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


def test_passes_explicit_id_tensor():
    env = _Env()
    assert reset_all(env) == ("obs", {})
    ids = env.calls[0]["env_ids"]
    assert isinstance(ids, torch.Tensor) and ids.dtype == torch.long
    assert ids.tolist() == [0, 1, 2]


def test_forwards_kwargs():
    env = _Env()
    reset_all(env, seed=7)
    assert env.calls[0]["seed"] == 7


def test_plain_reset_when_env_has_no_size():
    class Bare:
        def __init__(self):
            self.calls = []

        def reset(self, **kwargs):
            self.calls.append(kwargs)

    env = Bare()
    reset_all(env, seed=1)
    assert env.calls == [{"seed": 1}]
