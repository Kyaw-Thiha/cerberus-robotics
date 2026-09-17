"""Terrain-understanding candidate registry. Every known candidate is
registered directly in this one file -- add one by adding a new
`register_algorithm(AlgorithmCfg(...))` call below, not a new directory. See
docs/perception.md's Terrain Understanding section for sourcing and
plans/sensor_setup_handoff.md for how this feeds sensor-suite compatibility
checking (system/cerberus_bringup/cerberus_bringup/validate_sensor_suite.py).

Built from common/algorithm_registry.py's generic factory -- this module's
dict is fully independent of slam/slam's, planning/local_planner's, and
planning/global_planner's own core/algorithms.py, even though all four are
built from the same factory call. See that file's docstring for why sharing
the factory (pure data + closures, no domain content) doesn't compromise
the decoupling between these otherwise-peer modules.
"""

from common.algorithm_registry import AlgorithmCfg, make_algorithm_registry

register_algorithm, get_algorithm, available_algorithms = make_algorithm_registry("terrain understanding")

# AME-2 -- attention-based neural map encoding over a learning-based,
# uncertainty-aware elevation-mapping pipeline (Bayesian CNN predicts
# elevation + per-cell uncertainty from depth). Zhang, Klemm, Yang, Hutter
# (ETH RSL) 2026, arXiv:2601.08485. Dense-attention candidate in
# perception.md's terrain-understanding ablation. No implementation exists
# yet -- this registers only the declared sensor contract used for
# launch-time compatibility checking.
register_algorithm(AlgorithmCfg(
    name="ame2",
    sensor_requirements={"front_camera": "required"},
    description="Zhang et al. 2026, dense attention elevation encoding, sites.google.com/leggedrobotics.com/ame-2",
))

# DELTA -- deformable elevation-attention encoder: predicts a fixed number
# of proprioception-conditioned sampling locations, decoupling encoder cost
# from map resolution. Park, Jung, Hwangbo (KAIST) 2026, arXiv:2608.22033.
# Sparse-deformable-attention candidate in perception.md's terrain-
# understanding ablation; also the camera-only rig's real-hardware reference
# (front+rear D430, no LiDAR, no foothold planner). No implementation exists
# yet -- this registers only the declared sensor contract used for
# launch-time compatibility checking.
register_algorithm(AlgorithmCfg(
    name="delta",
    sensor_requirements={"front_camera": "required", "rear_camera": "optional"},
    description="Park et al. 2026, sparse deformable-attention elevation encoding, arxiv.org/abs/2608.22033",
))
