# B1 locomotion command: what any locomotion backend consumes (blind, map-based, depth E2E,
# world-model, raw-LiDAR). Typical rate 5-50 Hz. See docs/interfaces.md.
std_msgs/Header header

# Base twist in the base frame (m/s, m/s, rad/s)
float32 vx
float32 vy
float32 wz

# Optional body pose targets. Valid only when the matching has_* flag is true
# (flags instead of NaN sentinels, so the message survives serialisation and logging).
float32 body_pitch    # rad
float32 body_height   # m
bool has_body_pitch
bool has_body_height
