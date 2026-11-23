# Arm bring-up: Go2 + ARX5 as a locomotion platform

How the fused body is spawned, what the policy sees, and what is still open. Session evidence: 2026-11-23
(`docs/reports/first_scene_session.md`, session 2).

## What runs

```
--platform go2_arx5 --terrain rough      # train
--task Isaac-Velocity-Rough-Unitree-Go2-Arx5-Cerberus-Play-v0   # play / scene
```

`policy/locomotion/core/platforms/go2_arx5/` registers four task ids (flat/rough x train/play) with the **same PPO
agent cfg** as the legs-only platform, so the recipe is unchanged: 4096 envs, 1500 iterations, seed 42, 24 steps per
env.

## The contract the arm must not break

The frozen legs-only checkpoint is positional and 235-D wide. On the fused body:

- **action**: 12 leg joints, by name, in the contract's order (hips, thighs, calves per part);
- **observation**: the same 235-D vector; the leg joint blocks are 12 + 12, not 20 + 20;
- **arm**: `joint1..joint6` + `joint7/joint8` (gripper) are held at the stowed pose by their implicit PD actuators
  and appear in **neither** the action nor the observation.

The fused USD's joint order is per leg (FR, FL, RR, RL) because it comes from the merged URDF, while the contract is
per part. `preserve_order=True` with the explicit contract list is what keeps positional data aligned; without it
Isaac Lab resolves the joints in asset order and the policy would receive permuted joint positions. A live log line
shows the mapping:

```
Resolved joint names for the action term JointPositionAction: [... contract order ...]
  [tensor([1, 0, 3, 2, 5, 4, 7, 6, 10, 9, 12, 11])]
```

## Assets

The fused USD is generated, never committed: `data/assets/go2_arx5/go2_arx5.usd` (7.8 MB, self-contained, converted
from the merged URDF on Isaac Lab 3.0). It travels through the **private** R2 bucket at `embodiment/go2_arx5/`; the
arm source (`third_party/arx-model`, no licence file upstream) never leaves the machine and is never committed.

## Assertions

`scripts/scene_eval.py --embodiment go2_arx5` records the live joint names and the observation width, and the
contract assertions accept the 20-DoF articulation **only** when all 12 contract joints and all 8 arm joints are
present (`check_joint_order_embodiment`). The command-slice and observation-dimension checks are unchanged, so a
permuted or widened observation still fails the run.

## Open items (honest)

- **Arm randomisation events are blocked**: per-episode arm joint offsets and a payload mass both trip a PhysX
  device-side assert at the reset/startup event on this stack (three hypotheses tried: absolute vs added mass,
  `root_physx_view` vs `data.body_mass`, inertia consistency). The retrain runs with `CERBERUS_ARM_EVENTS=0`.
- **The 20-DoF body's joint order differs from the stock asset**: any new positional consumer must pin the order by
  name, exactly as the env cfg does.
- **Grasp in physics** is not implemented; the scripted skills (`planning/skills.py`) solve IK waypoints, and the
  mission's physics path is still the next milestone.
