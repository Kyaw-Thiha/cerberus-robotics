# B3 body command: B1 + B2 in one message for a unified whole-body controller or a loco-manip
# VLA. Typical rate 5-10 Hz (chunked). See docs/interfaces.md.
#
# The IIT Go2 VLA 11-D command (base twist 3 + body pitch 1 + 6 arm joints + gripper 1) is
# representable as a one-step chunk:
#   loco.vx, loco.vy, loco.wz        <- twist (3)
#   loco.body_pitch + has_body_pitch <- pitch (1)
#   arm.mode = JOINT_POSITION, arm.num_joints = 6, arm.joint_positions (len 6)   <- arm joints (6)
#   arm.gripper (len 1)              <- gripper (1)
# For chunked output, repeat BodyCommand at `dt`, or use a longer arm chunk with arm.dt.
std_msgs/Header header
cerberus_msgs/LocoCommand loco
cerberus_msgs/ArmCommand arm
float32 dt   # seconds this command is held / between consecutive BodyCommands
