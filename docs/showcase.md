# Showcase clips

Everything the owner watches lives in one flat folder, `outputs/videos/<date>/`, and is mirrored to R2 under
`showcase/<date>/`. Each clip is a half split: LEFT is the CHASE camera, RIGHT is the modality named in the file.
Each half is rendered natively at 960x1080 and captions say what is real, ground truth or scripted.

## What the labels mean

- **Render** — an RTX frame from the scene in Isaac Sim.
- **Ground truth** — simulator state (occupancy grid, planned path, pose, trajectory). Not perception.
- **Sensed** — what the robot's own simulated sensor provides (LiDAR depth image, camera image).
- **Scripted** — an open-loop or IK-driven motion, not a learned policy.

## Clips

| file | left | right | policy | what is real / ground truth / scripted |
|---|---|---|---|---|
| 01_chase_map.mp4 | Chase camera | Occupancy grid + A* path + trail (ground truth) | PPO Go2 (`go2_locomotion_ga_v1_candidate`) | Render + simulator ground-truth map; planned path from A* |
| 02_chase_ego.mp4 | Chase camera | Onboard front camera (sensed) | same run as 01 | Render; the ego image is the exact simulated camera view |
| 03_chase_lidar_jeplo.mp4 | Chase camera | 25x60 spherical depth as JEPLO sees it (sensed) | JEPLO released checkpoint + our sensor sim | Render + our LiDAR simulation; caption says so |
| 04_ppo_vs_jeplo.mp4 | PPO chase | JEPLO chase | PPO vs JEPLO, same seed/route/goals | Render; failures kept and captioned |
| 05_chase_wrist_mission_*.mp4 | Chase camera | Wrist camera + step strip | scripted IK (physical or magic grasp, labelled) | Render; grasp label states which |
| 00_cerberus_showcase.mp4 | full film | — | all of the above | title/chapter cards, captions, credits |

## Honest findings

- The arm-robot retrained policy (`go2_arx5_locomotion_ga_v1_candidate.pt`) does not translate in the scene on
  this stack: commands reach the observation and the height scan is sane, but base velocity stays ~0.015 m/s under
  a 0.7 m/s command. The legs-only Go2 policy tracks ~0.41 m/s in the same scene and env. The tour clips therefore
  use the Go2 baseline; this is a result, not a tuning problem hidden away.
- Session-3's "walk" clip did not move because it was open-loop and faced a couch; the current tour uses reachable
  goals, A* and closed-loop pure pursuit, and fixed the pure-pursuit turn-around singularity (alpha = pi).

## Credits

- GRScenes-100 scenes: CC BY-NC-SA 4.0, non-commercial use.
- Isaac Sim / Isaac Lab (NVIDIA).
- JEPLO: the authors' released checkpoint (Yuan et al.); our adapter and sensor simulation are labelled in every
  clip. `plugins/jeplo/` is GPL-3.0 and never imported by the core.
