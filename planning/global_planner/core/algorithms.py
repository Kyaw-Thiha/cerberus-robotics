"""Global-planner candidate registry. Every known candidate is registered
directly in this one file -- add one by adding a new
`register_algorithm(AlgorithmCfg(...))` call below, not a new directory. See
docs/perception.md's Global Planner section for sourcing and
plans/sensor_setup_handoff.md for how this feeds sensor-suite compatibility
checking (system/cerberus_bringup/cerberus_bringup/validate_sensor_suite.py).

Built from common/algorithm_registry.py's generic factory -- this module's
dict is fully independent of slam/slam's, perception/terrain_perception's,
and planning/local_planner's own core/algorithms.py, even though all four
are built from the same factory call. See that file's docstring for why
sharing the factory (pure data + closures, no domain content) doesn't
compromise the decoupling between these otherwise-peer modules.
"""

from common.algorithm_registry import AlgorithmCfg, make_algorithm_registry

register_algorithm, get_algorithm, available_algorithms = make_algorithm_registry("global planner")

# RAEM -- hybrid local-global traversability: robot-centric local
# tomography map + local 3D grid map drive an incrementally-built global
# topological graph, purpose-built for multi-floor structures. Classical,
# explicit map + graph search. Yuan et al. (multi-institution HK/China)
# 2026, arXiv:2608.25366. No implementation exists yet -- this registers
# only the declared sensor contract used for launch-time compatibility
# checking.
register_algorithm(AlgorithmCfg(
    name="raem",
    sensor_requirements={"lidar": "required"},
    description="Yuan et al. 2026, multi-floor topological exploration, arxiv.org/abs/2608.25366",
))

# HiPAN -- hierarchical RL, no explicit map: a high-level policy reads
# onboard depth images directly and outputs velocity + posture commands to a
# low-level posture-adaptive locomotion policy. Mapless, learned, implicit.
# Jeong, Yoon, Choi, Shin, Yang, Yoon (KAIST) 2026, arXiv:2604.26504. The
# camera-only rig's real-hardware-validated candidate (single front-facing
# depth camera, no LiDAR, no map). No implementation exists yet -- this
# registers only the declared sensor contract used for launch-time
# compatibility checking.
register_algorithm(AlgorithmCfg(
    name="hipan",
    sensor_requirements={"front_camera": "required"},
    description="Jeong et al. 2026, mapless hierarchical RL navigation, arxiv.org/abs/2604.26504",
))
