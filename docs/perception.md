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
| **RGB + Depth camera** | **In** | `CameraCfg`, RTX renderer (rgb, depth, normals, segmentation) |
| **LiDAR** | **In** | `RayCasterCfg` + `patterns.LidarPatternCfg` — a documented Go2 config combines this with a front `CameraCfg` in the same scene |
| **Radar** | **Out (for now)** | No native Isaac Lab sensor class exists (`isaacsim.sensors.experimental.rtx.Radar` is a raw Isaac Sim / Replicator API, single-prim, not batched for parallel RL envs). Requires Motion BVH for the Doppler effect, which materially raises VRAM/render cost stack-wide. Radar's actual value proposition (fog/dust/smoke penetration) doesn't organically appear in a clean sim scene without deliberately building obscurant scenarios to exercise it. **Revisit only if** a dedicated perceptual-degradation ablation scenario is built later. |

---

## Module Breakdown & SOTA Candidates

### 1. SLAM (state estimation + mapping, merged)

**FAST-LIO2** — LiDAR-inertial odometry, direct point registration (no feature extraction), incremental k-d tree (ikd-Tree) map. The de facto standard practical baseline — used as the real-world localization backbone by RAEM, SCAN-Planner, and VOP-Nav below.
- Xu, Cai, He, Lin, Zhang — IEEE T-RO, Aug 2022 (arXiv: Jul 2021)
- Paper: https://arxiv.org/abs/2107.06829
- GitHub: https://github.com/hku-mars/FAST_LIO
- Role here: **LiDAR-only baseline** for the SLAM ablation below.

**FAST-LIVO2** — tightly-coupled LiDAR-inertial-**visual** odometry, direct methods for both LiDAR and camera fused via a sequential-update error-state iterated Kalman filter on one unified voxel map.
- Zheng et al. (HKU-MARS) — Aug 2024
- **Metrics:** 0.044 m avg. translational RMSE across 25 benchmark sequences (NTU-VIRAL + Hilti), vs. 0.151 m for FAST-LIO2 (LiDAR-only), 0.278 m for R3LIVE, 1.928 m for LVI-SAM. Runs on both x86 and ARM (Kryo585).
- Paper: https://www.alphaxiv.org/abs/2408.14035
- GitHub: https://github.com/hku-mars/FAST-LIVO2
- Role here: **LiDAR+camera fused** candidate.

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
- **Metrics:** 99.5% success rate on stage-10 stepping stones; 96.6% avg. success across 4 unseen mixed terrain courses, vs. AME's 26.0% and a plain-MLP encoder's 3.5% on the same courses. 9.9× faster wall-clock training to reach 90% SR. Real-world: 100% SR across 21/21 trials on RAIBO2, depth-camera only (Intel RealSense D430, front + rear), no LiDAR, no foothold planner.
- Paper: https://www.alphaxiv.org/abs/2608.22033
- GitHub: not found as of this pass.

---

### 3. Local Planner (body-aware collision avoidance)

**SCAN-Planner** — yaw-aware "twin-cylinder" whole-body collision checking (not point/sphere inflation) + projected A* rebound search with z-gradient suppression + robot-centric sliding occupancy map with dead-end recovery. Optimization-based, not learned.
- Zheng, Chen, Fu, Yang, Qin (SJTU) — Jun 2026
- **Metrics:** in a cluttered-desk scene with an overhanging obstacle, finds the shortest feasible route through the under-table clearance while CMU-Planner / ART-Planner take longer bypasses; on stairs, the only compared method that stays physically valid for a ground robot (EGO-Planner-3D "succeeds" by flying over the obstacle). Real-world: Unitree Go2 + Livox Mid-360 LiDAR + Jetson Orin NX — 251 s / 149 m cross-floor inspection, 589 s / 367 m outdoor delivery run.
- Paper: https://www.alphaxiv.org/abs/2606.19555
- GitHub: https://github.com/wuyi2121/SCAN-Planner (**stated as "will be released"** — not confirmed live as of this pass)
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
- GitHub: none published; authors state code "will be released upon acceptance."
- Paradigm: explicit map + graph search.

**HiPAN** — hierarchical RL, no explicit map: a high-level policy reads onboard depth images directly and outputs strategic navigation commands (velocity **and** body posture — crouch/tilt for confined spaces); a low-level posture-adaptive locomotion policy executes them. Trained with "Path-Guided Curriculum Learning" to escape local minima (dead-ends) without myopic behavior.
- Jeong, Yoon, Choi, Shin, Yang, Yoon (KAIST) — Apr 2026
- **Metrics:** 94.4–98.5% success rate / 83.6–93.2 SPL across 4 unstructured 3D test environments, vs. 20–88% SR for classical Bug/Wall-Following and 44–82% SR for an end-to-end flat-RL baseline. Real-world: Unitree Go1 + RealSense D435i depth camera only, no LiDAR, no map — demonstrated dead-end backtracking and posture adaptation under a height-constrained passage.
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
| SLAM | FAST-LIO2 (LiDAR-only) | FAST-LIVO2 (LiDAR+camera) | Does the camera actually buy localization accuracy in our sim scenes, or is it dead weight? |
| Terrain understanding | AME-2 (dense attention) | DELTA (sparse deformable attention) | Does DELTA's resolution-independent cost trade away any accuracy vs. AME-2, given our compute budget? |
| Local planner | SCAN-Planner (optimization, static/structured) | VOP-Nav (learned, dynamic/crowded) | Does either degrade badly in the other's regime (SCAN-Planner with moving pedestrians; VOP-Nav on a staircase with overhangs)? If both degrade meaningfully, that argues for a switched/blended local-planner layer rather than picking one — a stronger systems finding than either paper alone claims. |
| Global planner | RAEM (explicit map + graph search) | HiPAN (mapless, learned) | Classical-vs-learned navigation, directly on your stated PhD interest in what should be classical vs. learned in a deployed system. |
| Uncertainty modeling | AME-2's Bayesian winner-take-all (in-domain, cheap) | UP-Fuse-style learned uncertainty gating (general, more robust under degradation) | Is the cheaper in-domain approach "good enough," or does the more general gating mechanism earn its extra complexity? |

---

## Explicitly Out of Scope for This Doc

Per earlier scoping discussion, these are real components of the eventual stack but are correctly gated behind later phases and not candidates to research/lock in yet:
- **Mission-level sequencing** (task orchestrator) — a nav-only skeleton is fine now; its real value (sequencing navigate↔manipulate) can't be exercised until Phase 2 exists.
- **Last-resort safety filter** (CBF/shielding) — needs a mature, composed policy stack to wrap and concrete failure modes to constrain against; premature before Phase 1+2 are done.
- **Object-level perception** (6D pose for grasp targets) — Phase 2 (manipulation), different job from Phase 1's "don't hit things while navigating."
- **Radar** — see Sensor Modalities table above.

---

## Open Decisions

- Final pick (or ablation scope) per module — this doc deliberately doesn't lock these in; see Proposed Ablations.
- Whether SCAN-Planner / RAEM's "code will be released" repos are live yet — worth re-checking before committing to either as a code base to build on.
- Whether to pursue the VOP-Nav / SCAN-Planner blended-local-planner idea as a real deliverable, or treat the ablation as purely diagnostic.
- Whether UP-Fuse's gating mechanism is worth reimplementing standalone, or whether AME-2's native uncertainty handling is sufficient given compute constraints.
