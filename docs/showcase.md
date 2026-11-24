# Showcase clips (2026-11-24)

Everything the owner watches is in one flat folder, `outputs/videos/2026-11-24/`, mirrored to R2 under
`showcase/2026-11-24/`. The film `00_cerberus_showcase.mp4` (98 s, 1920x1080, H.264, silent) presents the chapters;
the teaser and GIF are 60 s / 10 s cuts.

## What is real, ground truth, sensed or scripted

- **Render** — an RTX frame of the GRScenes home in Isaac Sim.
- **Ground truth** — simulator state (occupancy grid, A* path, pose). Not perception.
- **Sensed** — what the robot's own simulated sensor provides (the 25x60 LiDAR depth image).
- **Scripted** — open-loop or IK motion, not a learned policy.

## Clips

| file | content | policy | labels |
|---|---|---|---|
| 00_cerberus_showcase.mp4 | the film: home render, exploration map, JEPLO LiDAR, PPO baseline, arm robot, results | mixed, captioned per chapter | credits + "ground truth / sensed / scripted as captioned" |
| 01_exploration_map.mp4 | occupancy grid, planned A* path, travelled trail, goal, clock | JEPLO (released checkpoint) | grid and path are ground truth; 25.8 s run |
| 03_jeplo_lidar.mp4 | the 25x60 spherical depth image as JEPLO sees it, turbo scale in metres, bird's-eye hit strip | JEPLO + our sensor simulation | sensed, not ground truth |
| 04_ppo_map_falls.mp4 | the same seed/goal with the PPO height-scan policy: it falls at 5.3 s | PPO Go2 (`go2_locomotion_ga_v1_candidate`) | ground-truth map; honest failure |
| 99_results.png | success, falls, path efficiency (PPO / JEPLO / kinematic bound) | — | N and seeds printed in the subtitle |
| walk_ego.mp4, arm_ego.mp4 | session-3 RTX renders of the same home and the arm robot | scripted / open-loop | labelled "session 3" in the film |

## Results (same seed 0, same goal (7.55, -1.92), same planner and controller)

| policy | reached | net displacement | path efficiency | falls |
|---|---|---|---|---|
| JEPLO (released ONNX, our sensor simulation) | yes, 0.30 m | 11.33 m in 25.8 s | 0.955 | 0 |
| PPO Go2 height-scan | no | 3.67 m in 5.3 s | — | 1 |
| Kinematic bound (A* + pure pursuit) | yes | 11.28 m in 42.4 s | ~0.98 | 0 |

The JEPLO run uses the authors' unchanged checkpoint, our ray-caster simulation (authors' 25x60 pattern, 2 m range,
10 Hz), and no height scan. The PPO policy receives commands (verified in its observation) and a sane height scan
(min -1.0, max -0.12, std 0.30) but falls while turning; the arm-robot checkpoint does not translate at all in the
scene (base velocity 0.015 m/s under a 0.7 m/s command). Those are results, not tuning problems hidden away.

## Render blocker (honest)

The CHASE/EGO halves of the *current* runs could not be rendered on this VM: repeated Kit crashes left the GPU
context unusable and the scene was still bound to ~800 MDL materials. After converting the materials
(`scene_materials_to_preview.py` -> `scene_preview_metric.usd`, `CERBERUS_SCENE_USD_OVERRIDE`) and rebooting, a 2 s
`scene_clip.py` CHASE render succeeded, but the live JEPLO recording crashed again and its retries did not finish
inside the session. The data halves above are real, gated (`clip_qa --data-half` PASS) and use the same logs the
policy ran on; the film includes session-3 RTX renders of the same home for the visual chapters. Next step: record
the chase/ego halves in one live run on a fresh VM after the material conversion (code ready:
`jeplo_nav.py --record-dir`), then compose the half-splits.

## Credits

GRScenes-100 scenes: CC BY-NC-SA 4.0, non-commercial. Isaac Sim / Isaac Lab (NVIDIA). JEPLO: the authors' released
checkpoint (Yuan et al.), used with our adapter and sensor simulation; `plugins/jeplo/` is GPL-3.0.
