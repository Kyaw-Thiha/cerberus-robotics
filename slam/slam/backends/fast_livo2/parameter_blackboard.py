"""Drop-in replacement for ROS2's `demo_nodes_cpp` package's
`parameter_blackboard` executable -- that package isn't a dependency of
this backend's pixi env (confirmed by hitting "Package 'demo_nodes_cpp'
not found" against the real synced artifact on a pod), and adding it would
need a Docker image rebuild+push+re-extract cycle just for one trivial
utility. parameter_blackboard's entire job is to auto-declare whatever
parameters its own --params-file passes and do nothing else -- exactly what
rclpy's own `automatically_declare_parameters_from_overrides` does, so this
is a faithful, minimal reimplementation rather than a guess.

launch_algorithm.sh runs this via `python3 parameter_blackboard.py
--ros-args --params-file <camera_config.yaml>` in place of `ros2 run
demo_nodes_cpp parameter_blackboard`, matching the node name
mapping_avia_marslvig.launch.py's own vendored launch file uses
(`name='parameter_blackboard'`) so fastlivo_mapping's cross-node parameter
client finds it under the expected node name.
"""

import rclpy
from rclpy.node import Node


def main() -> None:
    rclpy.init()
    node = Node(
        "parameter_blackboard",
        allow_undeclared_parameters=True,
        automatically_declare_parameters_from_overrides=True,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
