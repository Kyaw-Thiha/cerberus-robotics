from common.cerberus_compat.isaac_data import as_tensor, physx_view


class FakeProxy:
    def __init__(self, t):
        self.torch = t


class FakeTensor:
    """Stands in for torch.Tensor: no `.torch` attribute."""


def test_as_tensor_unwraps_proxy():
    t = FakeTensor()
    assert as_tensor(FakeProxy(t)) is t


def test_as_tensor_passthrough():
    t = FakeTensor()
    assert as_tensor(t) is t
    assert as_tensor((1, 2)) == (1, 2)


def test_as_tensor_real_torch_passthrough():
    import pytest

    torch = pytest.importorskip("torch")
    t = torch.zeros(2)
    assert as_tensor(t) is t


def test_physx_view_prefers_root_view():
    class New:
        root_view = "new"
        root_physx_view = "old"

    class Old:
        root_physx_view = "old"

    assert physx_view(New()) == "new"
    assert physx_view(Old()) == "old"
