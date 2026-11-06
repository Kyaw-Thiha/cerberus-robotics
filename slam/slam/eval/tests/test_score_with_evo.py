from slam.slam.eval.score_with_evo import build_evo_ape_command, build_evo_rpe_command


def test_build_evo_ape_command_targets_ground_truth_and_slam_odom_topics():
    cmd = build_evo_ape_command(bag_path="bags/fast_lio2_household", backend_name="fast_lio2")
    assert cmd[0:2] == ["evo_ape", "bag2"]
    assert "bags/fast_lio2_household" in cmd
    assert "/ground_truth/odom" in cmd
    assert "/slam/odom" in cmd


def test_build_evo_ape_command_includes_backend_name_in_output_path():
    cmd = build_evo_ape_command(bag_path="bags/fast_lio2_household", backend_name="fast_lio2")
    idx = cmd.index("--save_results")
    assert "fast_lio2" in cmd[idx + 1]


def test_build_evo_rpe_command_targets_ground_truth_and_slam_odom_topics():
    cmd = build_evo_rpe_command(bag_path="bags/fast_lio2_household", backend_name="fast_lio2")
    assert cmd[0:2] == ["evo_rpe", "bag2"]
    assert "/ground_truth/odom" in cmd
    assert "/slam/odom" in cmd


def test_build_evo_rpe_command_includes_backend_name_in_output_path():
    cmd = build_evo_rpe_command(bag_path="bags/fast_lio2_household", backend_name="fast_lio2")
    idx = cmd.index("--save_results")
    assert "fast_lio2" in cmd[idx + 1]
    assert "rpe" in cmd[idx + 1]
