# Height scan as the rough-terrain locomotion policy was trained on it: a yaw-aligned grid of
# relative heights around the robot base, row-major, one float per cell.
#
# Convention: identical to Isaac Lab's `height_scan` observation term --
#   value = sensor_z - terrain_hit_z - offset, clipped to [-1, 1] m, with uniform noise in training.
# That formula is transcribed from the vendored Isaac Lab source, but its sign/offset was NOT
# verified against the live term (contract `height_scan.convention_status: unverified`). A producer
# that emits a different convention must say so; the policy will not detect the mismatch.
# Grid used in training: 17 x 11 cells, 0.1 m resolution, 1.6 x 1.0 m footprint.
std_msgs/Header header
uint32 rows
uint32 cols
float32 resolution
float32[] data
