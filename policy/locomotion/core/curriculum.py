"""Hwangbo et al. 2019 disturbance curriculum. See REFERENCES.md for the paper
citation, the verified formula/constants, and why the concrete magnitude/interval
endpoints come from isaaclab-go2-locomotion rather than the paper itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Verified constants from Hwangbo et al. 2019 (arXiv:1901.08652, Materials and
# Methods, "For all training sessions, we use k0 = 0.3 and kd = 0.997").
K0 = 0.3
KD = 0.997


class push_disturbance_curriculum(ManagerTermBase):
    """Scales a push EventTerm's force/torque range and trigger interval each
    iteration via Hwangbo's k_{j+1} = k_j^{k_d} curriculum factor.

    k_c starts at K0 and asymptotically approaches 1. force/torque ranges scale
    from 0 up to their peak as k_c -> 1; the trigger interval interpolates from
    a sparse early range down to a dense late range over the same k_c.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.event_manager.get_term_cfg(cfg.params["event_term_name"])
        self._k_c = K0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        event_term_name: str,
        peak_force: float,
        peak_torque: float,
        sparse_interval_range_s: tuple[float, float],
        dense_interval_range_s: tuple[float, float],
    ) -> torch.Tensor:
        self._k_c = self._k_c**KD

        # Symmetric, not (0, peak): apply_external_force_torque samples x, y,
        # AND z independently from this same range, so (0, peak) would always
        # push into the same fixed positive-x/y/z octant instead of a random
        # direction. (-peak, peak) applies to all three axes uniformly,
        # including z -- deliberately, not a bug: Shi et al. 2024 describes
        # base perturbation as simulating both "pushes and payload," and
        # "payload" is inherently a vertical force. See REFERENCES.md.
        current_peak_force = peak_force * self._k_c
        current_peak_torque = peak_torque * self._k_c
        self._term_cfg.params["force_range"] = (-current_peak_force, current_peak_force)
        self._term_cfg.params["torque_range"] = (-current_peak_torque, current_peak_torque)
        self._term_cfg.interval_range_s = (
            sparse_interval_range_s[0] + (dense_interval_range_s[0] - sparse_interval_range_s[0]) * self._k_c,
            sparse_interval_range_s[1] + (dense_interval_range_s[1] - sparse_interval_range_s[1]) * self._k_c,
        )

        # Stashed on env (not just logged to TensorBoard) so best_checkpoint_runner.py
        # can bucket "best" checkpoints by difficulty without reaching into
        # CurriculumManager internals -- see REFERENCES.md / best_checkpoint_runner.py.
        env.push_curriculum_k_c = self._k_c

        return torch.tensor(self._k_c, device=env.device)
