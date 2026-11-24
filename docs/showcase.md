# Showcase clips (2026-11-24)

Everything the owner watches is in one flat folder, `outputs/videos/2026-11-24/`, mirrored to R2 under
`showcase/2026-11-24/`. `00_cerberus_showcase.mp4` (79 s, 1920x1080, H.264, silent) is the film; the teaser and GIF
are 60 s / 10 s cuts.

## What is real, ground truth, sensed or scripted

- **Render** — an RTX frame of the GRScenes home in Isaac Sim.
- **Ground truth** — simulator state (occupancy grid, A* path, pose). Not perception.
- **Sensed** — what the robot's own simulated sensor provides (onboard RGB, 25x60 LiDAR depth).
- **Scripted** — a scripted arm motion or open-loop path, not a learned policy.

## Clips

| file | content | policy | labels / gate |
|---|---|---|---|
| 00_cerberus_showcase.mp4 | film: in the scene, exploration map, JEPLO LiDAR, PPO baseline, results | mixed, captioned per chapter | media checks + 6 frames inspected |
| 02_chase_render.mp4 | chase render of the Go2 + ARX5, scripted arm demo (14 s) | scripted | PASS gate v2 (hue 4-6, robot visible 5/5, yaw-only pose rule) |
| 01_exploration_map.mp4 | occupancy grid, A* path, trail, goal, clock | JEPLO run | data-half PASS; ground truth |
| 03_jeplo_lidar.mp4 | the 25x60 depth image as JEPLO sees it (turbo, metres) + BEV hits | JEPLO + our sensor sim | data-half PASS; sensed |
| 04_ppo_map_falls.mp4 | same seed/goal with the PPO height-scan policy: falls at 5.3 s | PPO Go2 | data-half PASS; honest failure |
| 99_results.png/.mp4 | success, falls, path efficiency | JEPLO N=5, PPO N=1, kinematic N=1 | N in the subtitle |

## Results (same seed 0, goal (7.55,-1.92), planner and controller)

| policy | reached | net | path efficiency | falls |
|---|---|---|---|---|
| JEPLO (released ONNX + our sensor sim) | yes, 0.30 m | 11.33 m in 25.8 s | 0.955 | 0 |
| PPO Go2 height-scan | no | 3.67 m in 5.3 s | — | 1 |
| Kinematic bound | yes | 11.28 m in 42.4 s | ~0.98 | 0 |

Seeded navigation eval (5 start-goal pairs, seed 0, 25 s cap): JEPLO **5/5, no falls, SPL 1.0**. The PPO N=5 run
could not be completed (Kit crashed at startup on repeated attempts); the N=1 tour above is the PPO evidence.

## What could not be published

- **JEPLO CHASE/EGO half-split**: the chase render exists but some sampled frames are occluded by furniture and
  others are white walls with <3 hue bins. The EGO camera, once moved out of the base-link frame, produced upright
  frames that pass the gate, but the Replicator annotator returned intermittent black frames between the gate's
  samples, so that half was not published. The film uses the gated single-view chase render instead.
- **T5 grasp mission**: `scene_arm_grasp.py` exits at Kit startup on this image (the camera/RTX extension path that
  `scene_clip.py` uses works, the grasp script does not); no mission numbers or clip this session.
- **PPO N=5** as above.

## Credits

GRScenes-100: CC BY-NC-SA 4.0, non-commercial. Isaac Sim / Isaac Lab (NVIDIA). JEPLO: the authors' released
checkpoint (Yuan et al.), used with our adapter and sensor simulation; `plugins/jeplo/` is GPL-3.0.
