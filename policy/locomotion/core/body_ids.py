"""How many bodies a resolved `SceneEntityCfg.body_ids` selects.

Dependency-free on purpose (no torch / isaaclab import) so it is unit-testable off-pod.

The wrench-composer push events size their force tensors as [n_envs, n_bodies, 3] and pass
`body_ids` straight through, so n_bodies must equal the number of selected bodies. Isaac Lab
resolves `body_ids` to a list on 2.x, but to a tensor on 3.x (manager terms finalize it), and to
`slice(None)` when no `body_names` were given. Counting only the list case (the pattern in Isaac
Lab's own `apply_external_force_torque`) sized the tensor for ALL bodies whenever `body_ids` was a
tensor, which no longer matches the single selected root body.
"""

from __future__ import annotations


def num_target_bodies(body_ids, num_bodies: int) -> int:
    """Number of bodies selected by `body_ids` on an asset with `num_bodies` bodies."""
    if isinstance(body_ids, slice):
        return len(range(*body_ids.indices(num_bodies)))
    if isinstance(body_ids, (list, tuple)):
        return len(body_ids)
    if hasattr(body_ids, "numel"):  # torch tensor
        return int(body_ids.numel())
    return num_bodies
