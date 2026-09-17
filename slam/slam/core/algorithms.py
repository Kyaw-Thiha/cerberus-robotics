"""SLAM candidate registry. Every known candidate is registered directly in
this one file -- add one by adding a new `register_algorithm(AlgorithmCfg(...))`
call below, not a new directory. See docs/perception.md's SLAM section for
sourcing and plans/sensor_setup_handoff.md for how this feeds sensor-suite
compatibility checking (system/cerberus_bringup/cerberus_bringup/validate_sensor_suite.py).

Built from common/algorithm_registry.py's generic factory -- this module's
dict is fully independent of perception/terrain_perception's,
planning/local_planner's, and planning/global_planner's own
core/algorithms.py, even though all four are built from the same factory
call. See that file's docstring for why sharing the factory (pure data +
closures, no domain content) doesn't compromise the decoupling between
these otherwise-peer modules.
"""

from common.algorithm_registry import AlgorithmCfg, make_algorithm_registry

register_algorithm, get_algorithm, available_algorithms = make_algorithm_registry("SLAM")

# FAST-LIO2 -- LiDAR-inertial odometry, direct point registration, ikd-Tree
# map. LiDAR-only baseline in perception.md's SLAM ablation. Xu, Cai, He,
# Lin, Zhang, IEEE T-RO 2022 (arXiv:2107.06829). No implementation exists
# yet -- this registers only the declared sensor contract used for
# launch-time compatibility checking.
register_algorithm(AlgorithmCfg(
    name="fast_lio2",
    sensor_requirements={"lidar": "required"},
    description="Xu et al. 2022 (T-RO), LiDAR-only odometry, github.com/hku-mars/FAST_LIO",
))

# FAST-LIVO2 -- tightly-coupled LiDAR-inertial-visual odometry, fused via a
# sequential-update error-state iterated Kalman filter on one unified voxel
# map. LiDAR+camera candidate in perception.md's SLAM ablation. Zheng et al.
# 2024 (HKU-MARS), arXiv:2408.14035. No implementation exists yet -- this
# registers only the declared sensor contract used for launch-time
# compatibility checking.
register_algorithm(AlgorithmCfg(
    name="fast_livo2",
    sensor_requirements={"lidar": "required", "front_camera": "required"},
    description="Zheng et al. 2024, LiDAR+camera fused odometry, github.com/hku-mars/FAST-LIVO2",
))
