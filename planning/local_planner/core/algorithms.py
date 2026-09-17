"""Local-planner candidate registry. Every known candidate is registered
directly in this one file -- add one by adding a new
`register_algorithm(AlgorithmCfg(...))` call below, not a new directory. See
docs/perception.md's Local Planner section for sourcing and
plans/sensor_setup_handoff.md for how this feeds sensor-suite compatibility
checking (system/cerberus_bringup/cerberus_bringup/validate_sensor_suite.py).

Built from common/algorithm_registry.py's generic factory -- this module's
dict is fully independent of slam/slam's, perception/terrain_perception's,
and planning/global_planner's own core/algorithms.py, even though all four
are built from the same factory call. See that file's docstring for why
sharing the factory (pure data + closures, no domain content) doesn't
compromise the decoupling between these otherwise-peer modules.

Note: `planning/local_planner/` is not yet in project_structure.md's locked
tree (which only lists `planning/global_planner/` and
`planning/task_orchestrator/`) -- added here because docs/perception.md §3
treats local/body-aware collision avoidance as a module distinct from
global route planning (§4), and perception.md is the doc of record where it
revises Phase 1 scoping. Tree update flagged, not silently assumed.
"""

from common.algorithm_registry import AlgorithmCfg, make_algorithm_registry

register_algorithm, get_algorithm, available_algorithms = make_algorithm_registry("local planner")

# SCAN-Planner -- yaw-aware whole-body collision checking + projected A*
# rebound search + robot-centric sliding occupancy map. Optimization-based,
# static/quasi-static structured 3D clutter regime. Zheng, Chen, Fu, Yang,
# Qin (SJTU) 2026, arXiv:2606.19555. No implementation exists yet -- this
# registers only the declared sensor contract used for launch-time
# compatibility checking. Real-world reference rig uses a Livox Mid-360
# LiDAR; camera-only-rig equivalent is an open gap (see docs/perception.md's
# Proposed Ablations table -- both local-planner candidates documented so
# far require LiDAR).
register_algorithm(AlgorithmCfg(
    name="scan_planner",
    sensor_requirements={"lidar": "required"},
    description="Zheng et al. 2026, whole-body collision-aware planning, github.com/wuyi2121/SCAN-Planner (unconfirmed released)",
))

# VOP-Nav -- RL policy implicitly learning Velocity-Obstacle-style safety
# constraints from raw multi-frame LiDAR (VOP-Net regresses a 360 degree
# safe-velocity region). Learned, dense/dynamic-crowd regime. Wu, Liu,
# Zhang, Li, Sun, Xiong, Xi, Yu, Zou (SJTU) 2026, arXiv:2607.15036. No
# implementation exists yet -- this registers only the declared sensor
# contract used for launch-time compatibility checking.
register_algorithm(AlgorithmCfg(
    name="vop_nav",
    sensor_requirements={"lidar": "required"},
    description="Wu et al. 2026, learned velocity-obstacle planning from raw LiDAR, arxiv.org/abs/2607.15036",
))
