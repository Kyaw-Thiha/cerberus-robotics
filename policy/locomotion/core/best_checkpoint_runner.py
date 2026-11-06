"""Adds difficulty-stratified best-checkpoint tracking on top of rsl_rl's
OnPolicyRunner, which only saves periodic snapshots (see rsl_rl/runners/on_policy_runner.py
-- `if it % self.save_interval == 0: self.save(...)`) and has no concept of
"best" at all.

"Best" is bucketed by the push-disturbance curriculum's difficulty factor
(curriculum.py's k_c), not a single global best, because k_c keeps rising over
training -- a policy late in training facing a much larger push range will
often show a *lower* raw reward than an equally-good policy early on facing
almost no disturbance. Comparing reward only within a bucket keeps the
comparison meaningful, and gives a fallback checkpoint at each difficulty
level (not just one, likely stuck in the easy zone) if training destabilizes
later.
"""

from __future__ import annotations

import os
import statistics

from rsl_rl.runners import OnPolicyRunner

from common.cerberus_compat.rsl_rl_compat import rsl_rl_major

# k_c bucket boundaries (k_c runs from K0=0.3 up to ~1.0, see curriculum.py).
# Upper bound of the last bucket is >1.0 so k_c==1.0 itself still falls inside it.
DIFFICULTY_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("easy", 0.30, 0.55),
    ("medium", 0.55, 0.80),
    ("hard", 0.80, 1.01),
)


def _bucket_for(k_c: float) -> str | None:
    for name, lo, hi in DIFFICULTY_BUCKETS:
        if lo <= k_c < hi:
            return name
    return None


class BestCheckpointOnPolicyRunner(OnPolicyRunner):
    """Same as OnPolicyRunner, but also writes model_best_<bucket>.pt whenever
    the current mean reward beats the best seen so far within that bucket.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_mean_reward: float | None = None
        self._best_reward_by_bucket: dict[str, float] = {}
        # rsl-rl >= 4 moved logging out of the runner (no `log` method any more; episode rewards live on
        # `self.logger.rewbuffer`), so `log` below is only called on rsl-rl 3.x.
        self._reward_from_logger = rsl_rl_major() >= 4

    def _refresh_mean_reward_from_logger(self) -> None:
        # UNVERIFIED: rsl-rl >= 4 branch (Logger.rewbuffer, rsl-rl v4.0.0 / v5.5.1); needs the new stack.
        rewbuffer = getattr(getattr(self, "logger", None), "rewbuffer", None)
        if rewbuffer is not None and len(rewbuffer) > 0:
            self._last_mean_reward = statistics.mean(rewbuffer)

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)
        if len(locs["rewbuffer"]) > 0:
            self._last_mean_reward = statistics.mean(locs["rewbuffer"])

    def save(self, path: str, infos: dict | None = None) -> None:
        if self._reward_from_logger:
            self._refresh_mean_reward_from_logger()
        super().save(path, infos)

        if self._last_mean_reward is None:
            return
        k_c = getattr(self.env.unwrapped, "push_curriculum_k_c", None)
        if k_c is None:
            return  # push curriculum not present on this env (e.g. a Play variant)
        bucket = _bucket_for(k_c)
        if bucket is None:
            return

        best_so_far = self._best_reward_by_bucket.get(bucket)
        if best_so_far is None or self._last_mean_reward > best_so_far:
            self._best_reward_by_bucket[bucket] = self._last_mean_reward
            best_path = os.path.join(os.path.dirname(path), f"model_best_{bucket}.pt")
            super().save(best_path, infos)
