# B2 arm command: an EE pose / EE trajectory chunk, or an arm joint-position chunk, plus gripper.
# Typical rate 5-30 Hz. A single command is a one-step chunk. See docs/interfaces.md.
std_msgs/Header header

uint8 EE_POSE=0
uint8 EE_TRAJECTORY=1
uint8 JOINT_POSITION=2
uint8 mode

# EE_POSE: exactly one entry. EE_TRAJECTORY: one entry per chunk step. Empty for JOINT_POSITION.
geometry_msgs/Pose[] ee_poses

# JOINT_POSITION: flattened row-major [step][joint], length = steps * num_joints (rad).
# Joints are ordered by the embodiment config's arm joint_group, never by position convention.
float32[] joint_positions
uint32 num_joints

# One value per chunk step (opening width or normalised command, per embodiment config).
float32[] gripper

# Seconds between chunk steps.
float32 dt
