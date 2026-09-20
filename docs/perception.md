# Cerberus — Perception & Navigation Stack Reference

Reference doc for Phase 1's perception, SLAM, planning, and perception-conditioned
control modules. Complements `cerberus_project_context.md` (overall project scope)
and `project_structure.md` (repo layout). Where this doc's conclusions revise the
original Phase 1 scoping in `cerberus_project_context.md` — e.g. adding LiDAR,
splitting global/local planning, treating terrain understanding as feeding the
locomotion policy rather than only the nav stack — this doc is the current source
of truth; the project-context doc predates this research pass.

Candidates below are not final picks. Per the project's "synthesis over invention"
ethos, several modules deliberately carry 2+ candidates earmarked for ablation
rather than a single locked choice — see **Proposed Ablations** at the end.

All papers were surfaced via alphaXiv in Sept 2026; "GitHub" is listed only where
a repo URL was explicitly confirmed in the paper itself — entries marked
*not released* or *not found* should be re-checked before being relied on, not
assumed absent forever.

---

## Sensor Modalities

| Modality | Decision | Isaac Lab mechanism |
|---|---|---|
| **LiDAR (spinning)** | **In — primary rig only** | `RayCasterCfg` + `patterns.LidarPatternCfg`. Spinning preferred over solid-state — legged robots' omnidirectional motion benefits from the wider FoV (Boxi, ETH 2025); matches stock Go2 EDU's built-in 4D LiDAR. A single front camera is sufficient alongside it — LiDAR's near-field/underbody coverage makes a rear camera redundant; no reference Go2+LiDAR rig in the literature (SCAN-Planner, VOP-Nav, RAEM, ForEnt) pairs it with one. |
| **Front stereo camera, global shutter** | **In — both rigs** | `CameraCfg` (stereo pair). Stereo + global shutter, not RGB-D/rolling-shutter — a 2026 multi-platform quadruped SLAM study found stereo consistently outperforms RGB-D and global shutter avoids the motion-blur tracking failures rolling shutter showed under quadruped locomotion. ~15–25° downward tilt (consistent with published Go2 nav/parkour rigs). |
| **Rear stereo camera** | **In — camera-only ablation rig only** | Same `CameraCfg` type, mirrored mount. Only needed when LiDAR is absent — DELTA's own real deployment (front + rear RealSense D430) is the reference rig for this path. |
| **Wrist-adjacent RGB-D (arm camera)** | **In — Phase 2, placeholder for now** | `CameraCfg`, mounted ~10cm proximal from the ARX5 wrist joint, not on the gripper, to avoid arm self-occlusion (per a quadruped-manipulator EAS design paper). RGB-D matches the Diffusion Policy/ACT baselines Phase 2 implements faithfully — though a 2026 paper (SCDP) found single-RGB can match or beat wrist-camera+RGB-D setups on manipulation benchmarks, which is a candidate additional Phase 2 ablation, not resolved here. |
| **Radar** | **Out (for now)** | No native Isaac Lab sensor class for batched parallel RL envs (`isaacsim.sensors.experimental.rtx.Radar` is a raw Isaac Sim / Replicator API, single-prim, not batched for parallel RL envs). Requires Motion BVH for the Doppler effect, which materially raises VRAM/render cost stack-wide. Radar's actual value proposition (fog/dust/smoke penetration) doesn't organically appear in a clean sim scene without deliberately building obscurant scenarios to exercise it. **Revisit only if** a dedicated perceptual-degradation ablation scenario is built later. |

See `docs/sensor_setup_handoff.md` for concrete Isaac Lab configs, the
sensor-suite composition pattern, and how algorithms declare and validate
their required sensor inputs.

---

## Module Breakdown & SOTA Candidates

### 1. SLAM (state estimation + mapping, merged)

Three sensor configs, two candidates each (baseline vs. latest), so each config
gets its own ablation rather than one algorithm being picked project-wide. Full
implementation detail — repo links, ROS versions, the pluggable-backend
architecture, Docker/`ros1_bridge` strategy, ground-truth export, and the
eval/viz pipeline — lives in `docs/slam_handoff.md`; this is the short version.

| Sensor config | Baseline | Latest |
|---|---|---|
| LiDAR only (primary rig, LiDAR-only algorithm) | FAST-LIO2 | BIEVR-LIO |
| LiDAR + camera (primary rig, fused algorithm) | FAST-LIVO2 | LVI-Q |
| Camera-only (ablation rig) | ORB-SLAM3 (stereo, no IMU) | ORB-SLAM3 + Multicam-SLAM fusion |

RGB-D was considered as a possible fourth config and rejected: a systematic
quadruped SLAM benchmark (Corlito et al., KIT/Freiburg, 2026) found stereo
consistently outperforms RGB-D on a legged platform — this confirms the
`sensor_setup_handoff.md` stereo-over-RGB-D decision independently rather than
contradicting it.

---

### 2. Terrain Understanding

*(Tightly coupled with Perception-Conditioned Control below by design in current SOTA — the terrain encoder generally **is** the conditioning mechanism, not a separate upstream module. Listed once here; cross-referenced from §5.)*

**AME-2** — attention-based neural map encoding: CNN extracts local + global elevation-map features, attention (conditioned on proprioception + global context) weights local features; paired with a learning-based, uncertainty-aware elevation-mapping pipeline (Bayesian CNN predicts elevation + per-cell uncertainty from depth, fused via a "probabilistic winner-take-all" rule so occlusions don't falsely gain confidence). Teacher-student distillation for sim-to-real.
- Zhang, Klemm, Yang, Hutter (ETH Zurich RSL) — Jan 2026
- **Metrics:** 82.4% avg. success rate on unseen test terrains (student policy, nominal conditions). Real ANYmal-D + LimX TRON1 biped. Uses LiDAR-inertial odometry (ANYmal) / DLIO (TRON1) for state estimation, depth camera for elevation input.
- Paper: https://www.alphaxiv.org/abs/2601.08485
- Project page: https://sites.google.com/leggedrobotics.com/ame-2
- GitHub: not found as of this pass.

**DELTA** — deformable elevation-attention encoder: predicts a *fixed* number of proprioception-conditioned sampling locations and encodes only those local elevation patches, decoupling encoder cost from map resolution (vs. AME-2's dense full-map attention).
- Park, Jung, Hwangbo (KAIST) — Aug 2026
- **Metrics:** 99.5% success rate on stage-10 stepping stones; 96.6% avg. success across 4 unseen mixed terrain courses, vs. AME's 26.0% and a plain-MLP encoder's 3.5% on the same courses. 9.9× faster wall-clock training to reach 90% SR. Real-world: 100% SR across 21/21 trials on RAIBO2, front + rear Intel RealSense D430 depth cameras (no LiDAR, no foothold planner).
- Paper: https://www.alphaxiv.org/abs/2608.22033
- GitHub: not found as of this pass.

---

### 3. Local Planner (body-aware collision avoidance)

**SCAN-Planner** — yaw-aware "twin-cylinder" whole-body collision checking (not point/sphere inflation) + projected A* rebound search with z-gradient suppression + robot-centric sliding occupancy map with dead-end recovery. Optimization-based, not learned.
- Zheng, Chen, Fu, Yang, Qin (SJTU) — Jun 2026
- **Metrics:** in a cluttered-desk scene with an overhanging obstacle, finds the shortest feasible route through the under-table clearance while CMU-Planner / ART-Planner take longer bypasses; on stairs, the only compared method that stays physically valid for a ground robot (EGO-Planner-3D "succeeds" by flying over the obstacle). Real-world: Unitree Go2 + Livox Mid-360 LiDAR + Jetson Orin NX — 251 s / 149 m cross-floor inspection, 589 s / 367 m outdoor delivery run.
- Paper: https://www.alphaxiv.org/abs/2606.19555
- GitHub: https://github.com/wuyi2121/SCAN-Planner — **confirmed live** (main algorithm released, ROS1 Noetic, `catkin_make`), not a placeholder repo.
- Best regime: static/quasi-static structured 3D clutter (narrow passages, overhangs, stairs).

**VOP-Nav** — RL policy that implicitly learns Velocity-Obstacle-style safety constraints from raw multi-frame LiDAR (VOP-Net regresses a 360° "safe velocity region," used both as a policy input and a reward-shaping signal), rather than explicit geometric optimization. Learned, not optimization-based.
- Wu, Liu, Zhang, Li, Sun, Xiong, Xi, Yu, Zou (SJTU) — Jul 2026
- **Metrics** (hardest "Square" crowded dynamic scenario, obstacles up to 1.5 m/s): 83.7% success / 15.7% collision rate, vs. ORCA 76.9%/19.9%, NavRL(2T) 76.4%/22.0%, ABS 53.1%/44.2%. Real-world: Unitree Go2 + Livox Mid-360 LiDAR + RealSense D435i, FAST-LIO2 for outdoor localization — 15/15 indoor trials with people actively obstructing the robot, 12/15 outdoor (2 failures attributed to localization drift, not the planner).
- Paper: https://www.alphaxiv.org/abs/2607.15036
- GitHub: not found as of this pass.
- Best regime: dense, unpredictable, non-cooperative **dynamic** obstacles (crowds) — the regime SCAN-Planner isn't built for.

---

### 4. Global Planner (route / exploration)

**RAEM** — hybrid local-global traversability: a robot-centric local tomography map + explicitly-categorized local 3D grid map drive an incrementally-built global topological graph, purpose-built for multi-floor structures (staircases, cross-floor connectivity) that planar/2.5D representations can't model. Classical, not learned.
- Yuan, Ren, Wang, Wang, Fang, Zhang, Cheng, Chen, Ho, Zhu, Xu, Cheng, Yang (multi-institution HK/China) — Aug 2026
- **Metrics:** 20/20 successful trials completing every floor across 4 simulated multi-floor buildings, vs. baselines (TARE, FAEL, HPHS) which fail to progress past the first floor in every scene (0/20 success on floors 2+; HPHS scores 0/20 entirely on the hardest scene). Real-world: Unitree Go2 + Mid-360S LiDAR + Jetson Orin NX + FAST-LIO2 — autonomously explored a real 5-floor stairwell.
- Paper: https://www.alphaxiv.org/abs/2608.25366
- GitHub: **none found** — checked directly, no repo or project-page code link exists anywhere for this paper (authors state code "will be released upon acceptance"). A near-identical alternative, "Multi-Floor Exploration for Ground Robots via an Incremental Reachable Graph and Structural Priors" (Zhu et al., alphaXiv 2605.23350), was also checked and also has no released code.
- **Fallback if code is needed before RAEM/its twin release: TARE** (Cao, Zhu, Choset, Zhang, RSS 2021) — `github.com/caochao39/tare_planner`, real and actively maintained (RSS Best Paper + Best System Paper). Classical, explicit map + hierarchical graph exploration — the same category as RAEM, just not multi-floor-native. RAEM's own results show TARE stalls past the first floor in true multi-floor buildings; this is a known, documented limitation to report honestly rather than a disqualifying flaw, and it may not even matter if the household mission scenes turn out to be single-story.
- Paradigm: explicit map + graph search.

**HiPAN** — hierarchical RL, no explicit map: a high-level policy reads onboard depth images directly and outputs strategic navigation commands (velocity **and** body posture — crouch/tilt for confined spaces); a low-level posture-adaptive locomotion policy executes them. Trained with "Path-Guided Curriculum Learning" to escape local minima (dead-ends) without myopic behavior.
- Jeong, Yoon, Choi, Shin, Yang, Yoon (KAIST) — Apr 2026
- **Metrics:** 94.4–98.5% success rate / 83.6–93.2 SPL across 4 unstructured 3D test environments, vs. 20–88% SR for classical Bug/Wall-Following and 44–82% SR for an end-to-end flat-RL baseline. Real-world: Unitree Go1 + single front-facing RealSense D435i depth camera, no LiDAR, no map — demonstrated dead-end backtracking and posture adaptation under a height-constrained passage.
- Paper: https://www.alphaxiv.org/abs/2604.26504
- GitHub: not found as of this pass (project page referenced, URL not captured).
- Paradigm: mapless, learned, implicit.

---

### 5. Perception-Conditioned Control

Same underlying techniques as §2 (Terrain Understanding), viewed from the "how does perception actually change locomotion" angle rather than the encoding angle:
- **Footstep-level** conditioning → **AME-2**, **DELTA** (see §2 for full entries)
- **Navigation-level** conditioning (velocity + posture, not individual footsteps) → **HiPAN** (see §4 for full entry)

---

### 6. Uncertainty Modeling

**UP-Fuse** — uncertainty-guided LiDAR-camera fusion: a lightweight MLP learns to predict per-feature instability under simulated camera degradation (brightness shift, sensor dropout, out-of-domain histogram shift), and that uncertainty gates how much a deformable-attention fusion module trusts camera features vs. falling back on LiDAR. **Note: this is a 3D panoptic segmentation / autonomous-driving paper (nuScenes, SemanticKITTI, Waymo), not a legged-robot paper** — included as a transferable fusion *mechanism*, not a direct deployment reference.
- Mohan, Drews, Miron, Cattaneo, Valada (Freiburg + Bosch) — Feb 2026
- **Metrics:** 80.7% PQ on Panoptic nuScenes val (fused) vs. 74.9% LiDAR-only. Robustness is the real result: under full camera dropout, competing fusion methods drop 4.2–5.0 PQ points *below their own LiDAR-only baseline*, while UP-Fuse drops only 1.2 points. Under 5° calibration drift, degrades 4.4% vs. 8.3% for the next-best method. Under day→night domain shift, competing methods lose 2.1–3.1 PQ; UP-Fuse gains 0.1%.
- Paper: https://www.alphaxiv.org/abs/2602.19349
- GitHub / project page: http://upfuse.cs.uni-freiburg.de

**AME-2's mapping pipeline** (see §2) is the in-domain, legged-robot-native alternative: a lighter-weight Bayesian CNN + "probabilistic winner-take-all" fusion rule, purpose-built for depth-camera elevation mapping rather than general LiDAR-camera fusion. Treat as the cheaper baseline; UP-Fuse's gating mechanism as the more general, more rigorously-validated-under-degradation alternative to ablate against.

---

## Proposed Ablations

| Module | Candidate A | Candidate B | What the ablation tests |
|---|---|---|---|
| SLAM (LiDAR-only) | FAST-LIO2 | BIEVR-LIO | Does BIEVR-LIO's degeneracy handling (bump-image voxel maps) actually matter in our flat, hallway-heavy household scenes? |
| SLAM (LiDAR + camera) | FAST-LIVO2 | LVI-Q | Does LVI-Q's joint-kinematics fusion — free, already on the ROS2 bus — meaningfully help over camera+LiDAR alone? |
| SLAM (camera-only) | ORB-SLAM3 (single stereo pair) | ORB-SLAM3 + Multicam-SLAM fusion | Does true front+rear fusion beat a single stereo pair enough to justify the extra integration work? |
| Terrain understanding | AME-2 (dense attention) | DELTA (sparse deformable attention) | Does DELTA's resolution-independent cost trade away any accuracy vs. AME-2, given our compute budget? |
| Local planner | SCAN-Planner (optimization, static/structured) | VOP-Nav (learned, dynamic/crowded) | Does either degrade badly in the other's regime (SCAN-Planner with moving pedestrians; VOP-Nav on a staircase with overhangs)? If both degrade meaningfully, that argues for a switched/blended local-planner layer rather than picking one — a stronger systems finding than either paper alone claims. |
| Global planner | RAEM (or TARE, if RAEM/its twin stay codeless) | HiPAN (mapless, learned) | Classical-vs-learned navigation, directly on your stated PhD interest in what should be classical vs. learned in a deployed system. |
| Uncertainty modeling | AME-2's Bayesian winner-take-all (in-domain, cheap) | UP-Fuse-style learned uncertainty gating (general, more robust under degradation) | Is the cheaper in-domain approach "good enough," or does the more general gating mechanism earn its extra complexity? |
| **Sensor rig** | **LiDAR + front stereo camera** | **Front + rear stereo camera, no LiDAR** | What does LiDAR actually buy over a well-designed camera-only rig, on our own household scenes? See `docs/sensor_setup_handoff.md` for the concrete configs and how algorithms declare which rig they require. |

---

## Explicitly Out of Scope for This Doc

Per earlier scoping discussion, these are real components of the eventual stack but are correctly gated behind later phases and not candidates to research/lock in yet:
- **Mission-level sequencing** (task orchestrator) — a nav-only skeleton is fine now; its real value (sequencing navigate↔manipulate) can't be exercised until Phase 2 exists.
- **Last-resort safety filter** (CBF/shielding) — needs a mature, composed policy stack to wrap and concrete failure modes to constrain against; premature before Phase 1+2 are done.
- **Object-level perception** (6D pose for grasp targets) — Phase 2 (manipulation), different job from Phase 1's "don't hit things while navigating."
- **Radar** — see Sensor Modalities table above.
- **Wrist/arm camera** — Phase 2; see `docs/sensor_setup_handoff.md` for the reserved config placeholder.

---

## Open Decisions

- Final pick (or ablation scope) per module — this doc deliberately doesn't lock these in; see Proposed Ablations.
- SLAM: full architecture, ROS versions, Docker strategy, and eval/viz pipeline are in `docs/slam_handoff.md`, not here.
- Whether SCAN-Planner's now-confirmed-live repo is far enough along to build on for the Local Planner ablation, or still needs more releases to land.
- Whether RAEM (or its uncoded twin) ever releases code before Phase 1's global planner work starts, or whether TARE is the permanent stand-in.
- Whether to pursue the VOP-Nav / SCAN-Planner blended-local-planner idea as a real deliverable, or treat the ablation as purely diagnostic.
- Whether UP-Fuse's gating mechanism is worth reimplementing standalone, or whether AME-2's native uncertainty handling is sufficient given compute constraints.
- Exact stereo camera specs (baseline, resolution) not yet tuned against a specific real-hardware reference — see `docs/sensor_setup_handoff.md` §7 for current placeholders.
