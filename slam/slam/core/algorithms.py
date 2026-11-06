"""SLAM candidate registry. Every known candidate is registered directly in
this one file -- add one by adding a new `register_algorithm(AlgorithmCfg(...))`
call below, not a new directory. See docs/perception.md's SLAM section for
sourcing. Feeds sensor-suite
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

register_algorithm, get_algorithm, available_algorithms = make_algorithm_registry(
    "SLAM"
)

# --- Sensor-config groupings ---
#
# | Sensor config          | sensor_requirements shape                          | Candidates                  |
# |-------------------------|-----------------------------------------------------|------------------------------|
# | LiDAR only               | {"lidar": "required", "imu": "required"}                               | fast_lio2 (baseline), bievr_lio (latest) |
# | LiDAR + camera (fused)   | {"lidar": "required", "imu": "required", "front_camera": "required"}   | fast_livo2 (baseline), gs_livo (latest)  |
# | Camera-only, single pair | {"front_camera": "required"}                        | orb_slam3 (baseline AND latest)          |
# | Camera-only, front+rear  | {"front_camera": "required", "rear_camera": "required"} | multicam_slam (latest, fuses onto orb_slam3) |
#
# See test_algorithms.py's test_sensor_config_groupings_match_documented_shape for the drift guard.

# LiDAR only. Xu et al. 2022 (IEEE T-RO).
#   - GitHub: https://github.com/hku-mars/FAST_LIO
#   - Paper:  https://arxiv.org/abs/2107.06829
register_algorithm(
    AlgorithmCfg(
        name="fast_lio2",
        sensor_requirements={"lidar": "required", "imu": "required"},
        description="Xu et al. 2022 (T-RO), LiDAR-only odometry, github.com/hku-mars/FAST_LIO",
    )
)

# LiDAR + front camera. Zheng et al. 2024 (IEEE T-RO).
#   - GitHub: https://github.com/hku-mars/FAST-LIVO2
#   - Paper:  https://arxiv.org/abs/2408.14035
register_algorithm(
    AlgorithmCfg(
        name="fast_livo2",
        sensor_requirements={"lidar": "required", "imu": "required", "front_camera": "required"},
        description="Zheng et al. 2024, LiDAR+camera fused odometry, github.com/hku-mars/FAST-LIVO2",
    )
)


# LiDAR only. Pfreundschuh et al. 2026 (RSS). Native ROS2 build — no ros1_bridge.
#   - GitHub: https://github.com/ethz-asl/BIEVR-LIO
#   - Paper:  https://arxiv.org/abs/2604.14421
register_algorithm(
    AlgorithmCfg(
        name="bievr_lio",
        sensor_requirements={"lidar": "required", "imu": "required"},
        description="Pfreundschuh et al. 2026 (RSS), degeneracy-robust LiDAR-inertial odometry, "
        "github.com/ethz-asl/BIEVR-LIO -- has a native ROS2 build, unlike its peers here",
    )
)

# LiDAR + front camera. Hong et al. 2025. Fork fixed a map-persistence write-back bug and migrated to ROS2 Jazzy.
#   - GitHub: https://github.com/HKUST-Aerial-Robotics/GS-LIVO
#   - Paper:  https://arxiv.org/abs/2501.08672
register_algorithm(
    AlgorithmCfg(
        name="gs_livo",
        sensor_requirements={"lidar": "required", "imu": "required", "front_camera": "required"},
        description="Hong et al. 2025, real-time Gaussian-splat LiDAR-inertial-visual "
        "odometry and mapping, IESKF backbone modified from FAST-LIVO2, "
        "github.com/HKUST-Aerial-Robotics/GS-LIVO -- fork fixed a map-persistence "
        "write-back bug and ported to ROS2 Jazzy; GS-map persistence is now an "
        "explicit gated async .ply export (off by default), not yet verified in a "
        "real pod build/run",
    )
)

# Front stereo only. Campos et al. 2021 (IEEE T-RO).
#   - GitHub: https://github.com/UZ-SLAMLab/ORB_SLAM3
#   - Paper:  https://arxiv.org/abs/2007.11898
register_algorithm(
    AlgorithmCfg(
        name="orb_slam3",
        sensor_requirements={"front_camera": "required"},
        description="Campos et al. 2021 (T-RO), stereo visual SLAM, no IMU, "
        "github.com/UZ-SLAMLab/ORB_SLAM3 (core) + github.com/zang09/ORB_SLAM3_ROS2 (wrapper)",
    )
)

# Front + rear stereo. Li, Pang, Hu 2024. Paper covers RGB-D multi-cam; repo is ORB-SLAM2-based.
#   - GitHub: https://github.com/AlterPang/Multi_ORB_SLAM
#   - Paper:  https://arxiv.org/abs/2406.06374
register_algorithm(
    AlgorithmCfg(
        name="multicam_slam",
        sensor_requirements={"front_camera": "required", "rear_camera": "required"},
        description="Front+rear stereo fusion on ORB-SLAM2, github.com/AlterPang/Multi_ORB_SLAM "
        "-- README is generic/unconfirmed, read src/ directly before implementing",
    )
)
