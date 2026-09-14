# Cerberus — Locomotion Policy Architecture Landscape

> **Status: provisional.** Research pass done while designing the ROS2 `locomotion_policy` inference node, to keep that node from hardcoding today's simplest setup as if it were the only option. Not a commitment to build anything beyond what's already trained (Flat, Rough). Update or delete sections as better info arrives — same pattern as `docs/manipulator.md`.

## Scope

Cerberus trains locomotion one way today: PPO via `rsl_rl`, single stage, a flat MLP actor-critic, observations built by concatenating proprioception with (for Rough) a raw simulated height-scan. Before locking the inference node's interface to that one shape, this doc maps out what else the field actually does, so the node's design doesn't foreclose better options later.

## Five architectural categories (2020-2025 literature)

Grounded in Ha et al., "Learning-based legged locomotion" (arXiv 2406.01152, 2024) plus targeted lookups.

1. **Blind / proprioception-only, history-based implicit estimation.** No exteroception — the policy infers terrain from joint/IMU history. Lee et al. 2020 (Science Robotics), DreamWaQ (Nahrendra et al., ICRA 2023), newer 2025 work (PPL, VB-Com, "Blind Dexterity"). Most robust to sensor failure; weakest at precise foothold placement (stepping stones).
2. **Teacher-student privileged distillation.** A teacher trained with perfect ground truth distills into a student with realistic noisy sensors, sometimes through a learned encoder. Miki et al. 2022 ("Wild ANYmal", Science Robotics) adds an attention-based "belief encoder" fusing noisy proprio and extero to survive dropout/occlusion that raw simulated data never modeled. RMA (Kumar et al. 2021) is the other reference. Newer work folds distillation into training online rather than as a separate offline stage.
3. **Direct concatenation of exteroceptive features, single-stage RL.** What Cerberus's Rough task does today — a raw height-scan term concatenated into a flat MLP's observation, one training stage, no teacher/student split. Earlier assumed to be a sim-only shortcut; 2023-2025 evidence (parkour lineage, DPL) shows it's a real hardware-validated paradigm **provided the training noise model realistically captures real sensor failure modes** (dropouts, occlusion, structured noise), not naive uniform noise — which is what Cerberus's current ±0.1m perturbation actually is. Real-hardware readiness is untested, not disqualifying.
4. **End-to-end jointly-trained encoders, no privileged teacher.** A CNN/transformer encoder over raw depth/images trained via RL alongside the policy, rather than distilled from a teacher. CReF, ParkourFormer, ETH Zurich's attention-based map encoding.
5. **World-model / diffusion / hybrid paradigms.** World-model perception, diffusion-conditioned control, foundation-style multi-embodiment policies (LocoFormer, "One Policy to Run Them All"). Mostly sim-validated or early demos — least mature of the five.

## What collapses at inference time

Training-time machinery (a teacher, an offline distillation pass) never runs on the robot — only the trained policy does. The five categories collapse to three runtime shapes:

- **Shape A — proprioception only** (+ maybe a short history buffer). Category 1, plus the deployed half of most category-2 setups.
- **Shape B — proprioception + a compact exteroceptive feature** (height-scan-style vector). Category 3 (current Rough checkpoint) and RMA-style latent variants.
- **Shape C — proprioception + a raw sensor tensor** (image/depth/point cloud) through an onboard encoder. Category 4 and the heavier end of category 5.

## Current Cerberus status

- **Flat checkpoint** (`unitree_go2_flat`, 1500 iters, exported on R2) — Shape A. No perception dependency.
- **Rough checkpoint** (`go2_locomotion_v1_candidate.pt`) — Shape B. Real architecture per 2023-2025 evidence, but the training noise model is simpler than deployed category-3 systems use. Open risk, not a blocker.
- **Shape C** — nothing trained, no encoder built. Not needed until a checkpoint that requires it exists.

## Is `rsl_rl` the limiting factor?

No. `rsl_rl` is what the entire research lineage above builds on, including the newer papers. The three independent choices that actually fix how complex this gets are all on top of it:

1. **Network architecture** — flat MLP vs. an encoder module (CNN/transformer) feeding the same kind of actor-critic.
2. **Training procedure** — single-stage vs. teacher-student/online distillation (still PPO underneath).
3. **Observation design** — raw concatenation vs. history-based implicit estimation vs. learned features.

Adopting a more sophisticated configuration later (realistic noise synthesis, teacher-student, Shape C encoder) doesn't require swapping training libraries — it's a config and procedure change within the same tooling.

## Recommendation

Don't build Shape C support (onboard encoder, real depth-sensor integration) now — no checkpoint needs it and it's separate engineering. Do design the `locomotion_policy` node's interface generically: what a checkpoint needs (which observation terms, in what order, whether it needs a sensor tensor at all) should be a declared property read at launch, not assumed in code. See `docs/superpowers/specs/2026-09-09-multi-platform-config-design.md`'s Part B for where that declaration lives.

## Known verification gaps

- **`platforms.yaml` `joint_names` is an unverified placeholder.** Populated from the URDF's own joint order (FR, FL, RR, RL, each hip/thigh/calf), not verified against Isaac Lab's real training-time articulation order — that requires pod access this session didn't have. To verify: on a pod with the trained env, `import` and instantiate it, then read `env.scene["robot"].joint_names` and compare against the YAML. Until checked, treat real-hardware runs as using an unverified joint mapping — the inference node's name-based remapping (`observation_assembly._build_permutation`) makes this a one-line YAML fix once verified, but doesn't make the current placeholder correct.

## Open items for whenever this gets revisited

- Whether the Rough checkpoint's simple uniform-noise randomization is enough for real deployment, or needs the realistic depth-corruption synthesis DPL and the parkour lineage use.
- Whether a teacher-student pass is worth adding for Rough, once real-world testing surfaces concrete failure modes worth distilling against.
