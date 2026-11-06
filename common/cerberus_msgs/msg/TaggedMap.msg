# Wraps a SLAM backend's native map output with the map_type tag REQUIRED by
# the SLAM topic contract to live on the topic's actual payload, not a node
# parameter (see this task's rationale). All six candidates registered in
# slam/slam/core/algorithms.py currently produce point clouds only
# ("FAST-LIO2, BIEVR-LIO, FAST-LIVO2, LVI-Q, ORB-SLAM3,
# Multicam-SLAM all produce point clouds") -- occupancy_grid/pose_graph
# fields are deliberately NOT added here (YAGNI) until a candidate that
# produces one is actually registered.

string map_type          # "point_cloud" for every candidate registered as of this plan
sensor_msgs/PointCloud2 point_cloud
