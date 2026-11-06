# Cerberus — Modular Stack Options: Locomotion, Manipulation & Whole-Body

Oct 30, 2026 · @Kevin

## TL;DR

Every locomotion, manipulation and whole-body paradigm in the current literature attaches to the rest of the stack at one of three command boundaries, so Cerberus can support all of them by fixing those three interfaces and making each paradigm a swappable backend behind them (same pattern as `cerberus_worlds`).

- **Locomotion** has five live paradigms: blind proprioceptive RL, modular map-then-policy, end-to-end depth-to-joints, latent world-model / JEPA perception, and raw-LiDAR end-to-end. The strongest Go2-native, code-released option for our LiDAR rig is **JEPLO** (Sep 2026: JEPA world model on raw Mid-360 scans, trained in Isaac Lab + RSL-RL, open source). For camera-only, **WMP** and **DAWN** are the world-model picks; for the modular path, a height-scan teacher plus GPU elevation mapping.
- **Manipulation** has three: modular perception-grasp-plan (FoundationPose + GraspGen + cuRobo), imitation learning (BC / Diffusion Policy / ACT), and VLAs. The VLA shortlist is **π0.5**, **GR00T N1.7**, **X-VLA**, **SmolVLA** and **MolmoAct2**, all open-weight and runnable through LeRobot.
- **Whole-body** has four tiers: decoupled legs + arm IK, a unified RL whole-body controller (WBC), a learned high-level policy or VLA emitting body-level commands into that WBC, and a fully end-to-end joint-token VLA. **UMI-on-Legs** (Go2 + ARX5, code public) is the closest existing match to our hardware for the WBC tier, and the Oct 2026 IIT Go2 VLA deployment gives a concrete 11-D body-command interface to copy.
- **Recommendation:** ship tiers 1–3 behind the interfaces; treat the fully end-to-end joint-level VLA (tier 4) as a documented non-goal, since no quadruped VLA found runs at joint-control rates.

## Build-now lineup (v1)

Version 1 ships two to four backends per slot, chosen because each has public code I checked directly (Nov 6, 2026) that runs on our Isaac Lab 2.3 / Sim 5.1 pin or is one config change away. Every slot gets at least two backends so the B1–B3 interfaces are actually exercised rather than shaped around one model.

| Slot | Backend | Paradigm | Paper | Code (verified) | Fit with our stack |
| --- | --- | --- | --- | --- | --- |
| Locomotion | **Blind PPO** (Rudin et al.) | A blind | [arXiv](https://arxiv.org/abs/2109.11978) · [alphaXiv](https://www.alphaxiv.org/abs/2109.11978) | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) built-in Go2 rough task | Native; already Phase 0 |
| Locomotion | **Height-map policy** (Miki et al.) + GPU elevation map | B modular | [arXiv](https://arxiv.org/abs/2201.08117) · [alphaXiv](https://www.alphaxiv.org/abs/2201.08117); map: [arXiv](https://arxiv.org/abs/2204.12876) · [alphaXiv](https://www.alphaxiv.org/abs/2204.12876) | Isaac Lab height-scan observation in sim; [elevation\_mapping\_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) for a real map (ROS1 catkin) | Native in sim; real-map path needs ROS1 bridge |
| Locomotion | **JEPLO** | D′ JEPA world model, LiDAR | [arXiv](https://arxiv.org/abs/2609.15770) · [alphaXiv](https://www.alphaxiv.org/abs/2609.15770) | [ASIG-X/JEPLO](https://github.com/ASIG-X/JEPLO), GPLv3; pretrained checkpoint, [dataset](https://huggingface.co/datasets/ASIG-Lab/JEPLO-Dataset), MuJoCo sim-to-sim, Jetson deploy | Pins `isaaclab==2.3.2.post1`, our exact version; Go2 native |
| Locomotion | **DAWN** | D reconstructive world model, depth | [arXiv](https://arxiv.org/abs/2609.29092) · [alphaXiv](https://www.alphaxiv.org/abs/2609.29092) | [DocyNoah/dawn-parkour](https://github.com/DocyNoah/dawn-parkour), MIT; Isaac Lab `release/2.3.0` + Sim 5.1 | Go1 tasks ship; Go2 is an asset/config swap. Contains a WMP-style RSSM, so it also covers WMP |
| Arm (B2) | **Classical grasp**: FoundationPose → GraspGen → cuRobo | M1 | FoundationPose [arXiv](https://arxiv.org/abs/2312.08344) · [alphaXiv](https://www.alphaxiv.org/abs/2312.08344); GraspGen [arXiv](https://arxiv.org/abs/2507.13097) · [alphaXiv](https://www.alphaxiv.org/abs/2507.13097); cuRobo [arXiv](https://arxiv.org/abs/2310.17274) · [alphaXiv](https://www.alphaxiv.org/abs/2310.17274) | [GraspGenX](https://github.com/NVlabs/GraspGenX) already chains grasps → cuRobo; [cuRobo](https://github.com/NVlabs/curobo) Apache-2.0 | In sim, start from ground-truth object pose and add FoundationPose later |
| Arm (B2) | **BC, Diffusion Policy, ACT** | M2 IL | robomimic [arXiv](https://arxiv.org/abs/2108.03298) · [alphaXiv](https://www.alphaxiv.org/abs/2108.03298); DP [arXiv](https://arxiv.org/abs/2303.04137) · [alphaXiv](https://www.alphaxiv.org/abs/2303.04137); ACT [arXiv](https://arxiv.org/abs/2304.13705) · [alphaXiv](https://www.alphaxiv.org/abs/2304.13705) | robomimic (Isaac Lab Mimic's native trainer); DP and ACT via LeRobot | One converter from Mimic HDF5 to LeRobot datasets serves all three |
| Arm (B2) | **SmolVLA** | M3 VLA | [arXiv](https://arxiv.org/abs/2506.01844) · [alphaXiv](https://www.alphaxiv.org/abs/2506.01844) | LeRobot | Cheapest VLA; runs the full demo-count sweep |
| Arm (B2) | **X-VLA** (0.9B) | M3 VLA | [arXiv](https://arxiv.org/abs/2510.10274) · [alphaXiv](https://www.alphaxiv.org/abs/2510.10274) | [2toinf/X-VLA](https://github.com/2toinf/X-VLA), Apache-2.0, LoRA code + server/client | Outputs EE 6D actions, which map straight onto B2 |
| Arm (B2) | **π0.5** | M3 VLA | [arXiv](https://arxiv.org/abs/2504.16054) · [alphaXiv](https://www.alphaxiv.org/abs/2504.16054) | openpi (Apache-2.0) and LeRobot | Reference VLA; LoRA needs >22.5 GB |
| Whole-body | **T1 decoupled**: any locomotion backend + differential IK | T1 | n/a (our composition) | Isaac Lab differential-IK controller | Native |
| Whole-body | **LeggedManip\_Lab `GO2-ARX5-WBC`** | T2 unified WBC | No paper (community repo, 2026) | [zzzJie-Robot/LeggedManip\_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab): Isaac Lab, Flat + WBC modes for GO2-ARX5 | README says the Isaac Lab v2.3.2 tag is not compatible (needs a newer RSL-RL), so run it in its own env or after the Lab 3.0 upgrade |

**Port next** (code exists, but on Isaac Gym or outside Isaac Lab):

- **UMI-on-Legs** ([arXiv](https://arxiv.org/abs/2407.10353) · [alphaXiv](https://www.alphaxiv.org/abs/2407.10353), [code](https://github.com/real-stanford/umi-on-legs), MIT): Isaac Gym, but ships Go2 + ARX5 URDF and configs. Port its EE-trajectory WBC as the second T2 backend and the T3 interface.
- **RoboDuet** ([arXiv](https://arxiv.org/abs/2403.17367) · [alphaXiv](https://www.alphaxiv.org/abs/2403.17367), [code](https://github.com/locomanip-duet/RoboDuet)): Isaac Gym; trains Go1 or Go2, with a separate Go2 + ARX5 deployment repo.
- **MolmoAct2** ([arXiv](https://arxiv.org/abs/2605.02881) · [alphaXiv](https://www.alphaxiv.org/abs/2605.02881), [code](https://github.com/allenai/molmoact2)): weights, training code and data released; the T3 body-command VLA, as in the IIT Go2 work.
- **GR00T N1.x** ([arXiv](https://arxiv.org/abs/2503.14734) · [alphaXiv](https://www.alphaxiv.org/abs/2503.14734), [code](https://github.com/NVIDIA/Isaac-GR00T)): Isaac-native VLA, but ≥40 GB to fine-tune.
- **Extreme Parkour** ([arXiv](https://arxiv.org/abs/2309.14341) · [alphaXiv](https://www.alphaxiv.org/abs/2309.14341), [code](https://github.com/chengxuxin/extreme-parkour)) and **WMP** ([arXiv](https://arxiv.org/abs/2409.16784) · [alphaXiv](https://www.alphaxiv.org/abs/2409.16784), [code](https://gitee.com/ByteDance/WMP)): Isaac Gym; only needed for exact reproductions, since DAWN covers the paradigm.
- **RWM / RWM-O** ([arXiv](https://arxiv.org/abs/2501.10100) · [alphaXiv](https://www.alphaxiv.org/abs/2501.10100); [arXiv](https://arxiv.org/abs/2504.16680) · [alphaXiv](https://www.alphaxiv.org/abs/2504.16680), [code](https://github.com/leggedrobotics/rsl_rl_rwm)): an RSL-RL fork; belongs to Phase 4.
- **3D Diffusion Policy** ([arXiv](https://arxiv.org/abs/2403.03954) · [alphaXiv](https://www.alphaxiv.org/abs/2403.03954)): point-cloud IL, once wrist depth is in.

**Easy to implement ourselves** (no code, but small, well-specified modules):

- **AME-2** ([arXiv](https://arxiv.org/abs/2601.08485) · [alphaXiv](https://www.alphaxiv.org/abs/2601.08485)) and **DELTA** ([arXiv](https://arxiv.org/abs/2608.22033) · [alphaXiv](https://www.alphaxiv.org/abs/2608.22033)) map encoders: both consume the same elevation map as the height-map backend, so each is a swappable encoder inside it, not a new pipeline.
- **IIT 11-D body command** ([arXiv](https://arxiv.org/abs/2610.32271) · [alphaXiv](https://www.alphaxiv.org/abs/2610.32271)): a message convention, adopted as `BodyCommand`.

**Not in v1:** Omni-Perception's policy ([arXiv](https://arxiv.org/abs/2505.19214) · [alphaXiv](https://www.alphaxiv.org/abs/2505.19214)): its repo has released only the LiDAR simulator so far, with training code in progress; reuse its Mid-360 scan patterns instead. PointWalker, TA-WBC, FWBC-VLA, PIE and standalone CTS have no public code found (CTS training ships inside JEPLO's RSL-RL fork). QUART stays out as tier T4.

### How backends ship so others can swap them

- **Plugin packages.** Each backend is its own pip package that registers a Python entry point (e.g. `cerberus.backends.locomotion = jeplo:JeploBackend`), so a third party adds a model without forking Cerberus.
- **Train in the backend's env, run behind the interface.** The pins already conflict: JEPLO wants torch 2.7 and Lab 2.3.2.post1, LeggedManip\_Lab a newer RSL-RL, LeRobot no Isaac at all. Training stays in each backend's container; deployment loads an exported ONNX/TorchScript policy behind B1–B3. JEPLO's own deploy stack (ONNX + ZeroMQ) is the pattern to copy.
- **Backend card.** A YAML per backend: boundary, required sensors, rates, license, checkpoint URL, and its numbers on the shared evals.
- **License isolation.** JEPLO is GPLv3 and FoundationPose carries an NVIDIA research license, so both stay optional plugins and the Cerberus core stays permissive.
- **Shared evals.** The AllenAI VLA harness for arm policies; Cerberus scenes plus the Phase 0.5 adversary for every locomotion backend.

## Design principle: three command boundaries

Swappability comes from freezing the messages between layers, not from picking one algorithm. Each boundary below already exists in published systems, so every candidate later in this doc plugs into one of them without glue code beyond an adapter.

&#91;embedded content: command boundaries · 3 producers, 3 interfaces, 3 backend families\]

The dashed path is how UMI-on-Legs runs a tabletop Diffusion Policy on legs: an arm-level command fed into the whole-body controller instead of an IK solver.

| Boundary | Message (proposed `cerberus_msgs`) | Typical rate | Upstream producers | Downstream consumers |
| --- | --- | --- | --- | --- |
| **B1 Locomotion command** | `LocoCommand`: base twist (vx, vy, ωz) + optional body pitch/height | 5–50 Hz | Global/local planner, navigation policy (HiPAN-style), VLA base head | Any locomotion backend: blind, map-based, depth E2E, world-model, raw-LiDAR |
| **B2 Arm command** | `ArmCommand`: EE pose or EE trajectory chunk, or arm joint-position chunk + gripper | 5–30 Hz | Grasp planner + cuRobo, BC / Diffusion Policy / ACT, VLA | Arm IK / joint trajectory controller, or a whole-body controller |
| **B3 Body command** | `BodyCommand`: B1 ⊕ B2 in one vector (the IIT Go2 VLA used 11-D: twist, pitch, 6 arm joints, gripper) | 5–10 Hz (chunked) | Whole-body IL policy, loco-manip VLA | Unified whole-body controller (RoboDuet, UMI-on-Legs, TA-WBC style) |

Two rules make this hold:

1. **Backends declare their sensor needs and their boundary**, the same way `sensor_setup_handoff.md` already has algorithms declare required sensors. A config is valid only if every boundary has exactly one producer and one consumer.
2. **The safety filter sits on B1/B2/B3 and on joint targets**, so it is paradigm-agnostic. A fully end-to-end joint-token VLA would bypass B1–B3 entirely, which is one more reason it is out of scope (§ Whole-body).

## Locomotion stacks

Five paradigms are live, and all five consume a B1 velocity command and emit 12 leg joint targets at \~50 Hz, so they are drop-in replacements for each other. They differ in what sits between the sensor and the policy: nothing (blind), an explicit map, a learned image encoder, a learned predictive world model, or a point-cloud network.

| Paradigm | Candidate | Sensor input | Real robot | Code | Why it's a candidate |
| --- | --- | --- | --- | --- | --- |
| **A. Blind proprioceptive** | Isaac Lab `Isaac-Velocity-Rough-Unitree-Go2` PPO (Rudin et al. 2022) | IMU + joints | Go2 | Yes (Isaac Lab) | Phase 0 baseline; already the project default |
| A. Blind | CTS, concurrent teacher-student (Wang et al., RA-L 2024) | IMU + joints | Go1 | No standalone repo found; CTS training ships inside JEPLO's RSL-RL fork | Single-stage teacher-student; JEPLO reuses its training scheme |
| **B. Modular: map → policy** | Height-scan teacher → elevation map student (Miki et al., Sci. Robotics 2022) + GPU elevation mapping (Miki et al., IROS 2022) | LiDAR or depth via odometry + elevation map | ANYmal | Mapping code public; policy is the standard Isaac Lab height-scan task | Most interpretable; reuses the SLAM stack's odometry; the map is inspectable and safety-filterable |
| B. Modular | [AME-2](https://www.alphaxiv.org/abs/2601.08485) / [DELTA](https://www.alphaxiv.org/abs/2608.22033) map encoders | Elevation map | ANYmal-D, RAIBO2 | Not found | Best reported terrain success; already in `perception.md` |
| **C. End-to-end depth → joints** | Extreme Parkour (Cheng et al., ICRA 2024) | Front depth (D435) | A1 | [Yes](https://github.com/chengxuxin/extreme-parkour) (Isaac Gym) | Canonical teacher-student depth distillation baseline |
| **D. Latent world model (reconstructive)** | [WMP](https://arxiv.org/abs/2409.16784) (Lai et al., ICRA 2025) | Front depth | A1 | [Yes, Apache-2.0](https://gitee.com/ByteDance/WMP) (Isaac Gym legged\_gym, \~23 GB GPU) | RSSM world model learns terrain from depth history without a privileged teacher |
| D. Latent world model | [DAWN](https://dawn-parkour.github.io/) (IROS 2026) | Raw noisy depth (D435i) | Go1 | [Yes, MIT](https://github.com/DocyNoah/dawn-parkour) (Isaac Lab 2.3, Sim 5.1) | WMP + denoising target + contrastive alignment; no hand-tuned depth filters |
| **D′. Latent world model (JEPA, non-reconstructive)** | [JEPLO](https://arxiv.org/abs/2609.15770) (Yuan et al., Sep 2026) | Raw LiDAR (Livox Mid-360) projected to a 60×25 spherical depth image | **Go2** | [Yes, GPLv3](https://github.com/ASIG-X/JEPLO) (Isaac Lab 2.3.2.post1 + RSL-RL; pretrained checkpoint) | Mapping-free, no SLAM needed; beats WMP and PIE under occlusion |
| D″. World model as simulator | [RWM / RWM-O](https://arxiv.org/abs/2501.10100) (Li, Krause, Hutter, ETH) | Proprio | ANYmal-D | [Yes](https://github.com/leggedrobotics/rsl_rl_rwm) (RSL-RL fork) | Policy trained inside a learned dynamics model; directly relevant to Phase 4 |
| **E. Raw LiDAR end-to-end** | [Omni-Perception](https://arxiv.org/abs/2505.19214) (CoRL 2025) | Raw spatio-temporal point clouds | Go2 | [Partial](https://github.com/aCodeDog/OmniPerception): LiDAR simulator released; training code in progress | Omnidirectional collision avoidance inside the locomotion policy |
| E. Raw LiDAR | [PointWalker](https://arxiv.org/abs/2610.15548) (Oct 2026) | Raw points from 4 LiDARs | DEEP Robotics X30 | Not found | Single policy for stairs to balance beams from raw points |

**What the numbers say.** In JEPLO's MuJoCo benchmark (500 trials per cell, same CTS training for every method), its success rate on 0.2 m stairs was 59.2% vs 57.8% for a WMP-style DreamerV3 encoder and 54.8% for PIE. On a 0.5 m box it was 100% vs 56.2% (PIE) and 7.8% (WMP), and it degraded least under occlusion masks. Onboard it used 12.4% of one CPU core, 1.84 GB memory and 7.35% GPU on a Jetson AGX Orin, with the policy at 50 Hz and the JEPA module at 10 Hz.

**Where the user's "JEPLO-like" intuition lands.** JEPLO is not a fully end-to-end pixels-to-torques model: it is paradigm D′, a predictive latent encoder feeding a standard PPO actor. That is good news for modularity, because its encoder can be swapped for an elevation-map encoder (B) or a reconstructive world model (D) without touching the actor's B1 interface.

**Recommended locomotion slots:** A (baseline, have it), B (interpretable, pairs with SLAM), D′ JEPLO (LiDAR rig, strongest code-available Go2 option), D WMP/DAWN (camera-only rig). E is a stretch slot for collision-avoidance-in-the-policy experiments.

## Sensor configurations

Sensor rig and locomotion paradigm are separate axes, so each rig needs at least two candidate backends to make the `perception.md` rig ablation a fair comparison rather than a comparison of whatever algorithm happened to fit.

| Rig | Sensors | Locomotion backends that fit | Manipulation inputs | Notes |
| --- | --- | --- | --- | --- |
| **R0 Blind** | IMU + joint encoders | A (Isaac Lab PPO, CTS), RWM | n/a | Control floor; also the fallback when perception fails |
| **R1 LiDAR + front stereo** (primary) | Spinning LiDAR, front global-shutter stereo | B (elevation map), D′ JEPLO, E Omni-Perception | Front stereo + wrist RGB-D | JEPLO crops the LiDAR to a forward 120° × 50° sector and accumulates 5 scans; mounting position matters |
| **R2 Camera-only** (ablation) | Front + rear stereo | C Extreme Parkour, D WMP / DAWN, B via stereo depth | Front stereo + wrist RGB-D | DELTA's real rig (front + rear depth) is the reference |
| **R3 LiDAR-only** | Spinning LiDAR | D′ JEPLO, E | Needs wrist camera anyway | Isolates the LiDAR contribution; manipulation still needs a camera |
| **R4 Manipulation add-on** | Wrist-adjacent RGB-D + head camera | n/a | All IL and VLA candidates | The IIT Go2 VLA used a wrist D435 + head D405; most VLAs expect 1–2 RGB views |

**Caveat on the LiDAR model.** JEPLO and Omni-Perception were validated with a Livox Mid-360, not the Go2's built-in Unitree L1. Isaac Lab ray-cast LiDAR can emulate either pattern, but the sim pattern must match whichever unit is assumed, and the L1's sparser scan is untested for JEPLO.

## Manipulation stacks

Three paradigms cover the arm, and all three can emit a B2 arm command, so the Phase 2 comparison can include a classical baseline, not only learned policies. VLAs are now cheap enough to fine-tune with LoRA on one large GPU, which makes a fourth comparison point realistic rather than aspirational.

### M1. Modular: perceive → grasp → plan

| Stage | Candidate | Code | Role |
| --- | --- | --- | --- |
| 6D pose of novel objects | [FoundationPose](https://arxiv.org/abs/2312.08344) (NVIDIA, CVPR 2024) | Yes, research license | Object pose for grasp targets (`object_perception`) |
| 6-DOF grasp generation | [GraspGen](https://arxiv.org/html/2507.13097v1) (NVlabs, diffusion on SE(3)) | Yes; Isaac Sim USD export | Parallel-jaw grasps; new grippers via a config wizard in [GraspGenX](https://github.com/NVlabs/GraspGenX) |
| Motion planning | [cuRobo](https://github.com/NVlabs/curobo) (NVlabs) | Yes, Apache-2.0 since v2 | GPU collision-free trajectories; GraspGenX already wires grasp → cuRobo end to end |

No learning in the loop, so it is the honest "what does learning buy you" baseline, and it doubles as an automatic demo generator for Isaac Lab Mimic.

### M2. Imitation learning policies

| Candidate | Inputs | Notes |
| --- | --- | --- |
| BC (robomimic) | RGB(-D) + proprio | Baseline in the existing Phase 2 plan |
| Diffusion Policy (Chi et al.) | RGB + proprio | Strong comparison point; also the policy UMI-on-Legs runs on Go2 + ARX5 |
| ACT (Zhao et al.) | Multi-view RGB | Action chunking; cheap to train |
| 3D Diffusion Policy (DP3) | Point cloud | Natural fit if the rig has LiDAR or wrist depth |

### M3. Vision-language-action models

| Model | Params | Weights / license | LIBERO (reported) | Fine-tune memory | Why include it |
| --- | --- | --- | --- | --- | --- |
| **π0.5** (Physical Intelligence) | 3.6B | openpi, Apache-2.0 | 96.85% avg | >22.5 GB LoRA, >70 GB full | Common research baseline; in LeRobot |
| **GR00T N1.7** (NVIDIA, Feb 2026) | 3B | Isaac-GR00T repo | 94.4–98.5% per suite | ≥40 GB recommended | Native to the Isaac ecosystem; runs on Jetson Orin/Thor |
| **X-VLA** | 0.9B | Apache-2.0 | 98.1% | Not stated | Small and strong; cheapest serious option |
| **SmolVLA** (Hugging Face) | \~0.45B | LeRobot | Not compared here | Single consumer GPU | Budget option for the demo-count sweep |
| **MolmoAct2** (AI2, May 2026) | n/a | Open weights + recipe | 97.2% avg | One 96 GB GPU with rank-512 LoRA (IIT) | **Already deployed on a Go2 + arm** for loco-manipulation (Oct 2026) |
| VLA-JEPA | n/a | LeRobot | 96% (harness reproduction) | n/a | VLA with a latent JEPA world model; ties to the world-model thesis |

LIBERO numbers are self-reported by each team except the harness reproductions; the [AllenAI vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness) reproduced π0.5 at 100%, GR00T N1.7 at 99%, MolmoAct2 at 97% and VLA-JEPA at 96% on LIBERO through its LeRobot bridge. Use that harness plus LeRobot as the common serving layer so every VLA appears as one `ArmCommand` producer.

**Recommended manipulation slots:** M1 (classical baseline + Mimic seed), BC and Diffusion Policy (core IL comparison), π0.5 or X-VLA (VLA point; X-VLA if compute is tight), with MolmoAct2 reserved for the whole-body tier because it has a legged precedent.

## Whole-body loco-manipulation

The quadruped-plus-arm system has four tiers of integration, ordered from most to least modular. Tiers 1–3 keep a learned low-level controller and talk to it through B1–B3; only tier 4 removes that boundary. Tier 2 is the real upgrade over the current plan, because coordinated legs reach further and track more precisely than a frozen gait plus arm IK.

| Tier | What the upper layer outputs | Candidate | Hardware | Code | Evidence |
| --- | --- | --- | --- | --- | --- |
| **T1 Decoupled** | B1 to frozen legs policy + B2 to arm IK | Current Cerberus plan; LeggedManip\_Lab `GO2-ARX5` task | Go2 + ARX5 | [Yes](https://github.com/zzzJie-Robot/LeggedManip_Lab) (Isaac Lab; needs newer RSL-RL than the v2.3.2 tag) | Simplest; TA-WBC's baseline "blind locomotion + IK" reached 0.710 m³ workspace and 0.076 m EE error on flat ground |
| **T2 Unified RL whole-body controller** | Low-level policy tracks base twist + EE pose with all 18 joints | [UMI-on-Legs](https://github.com/real-stanford/umi-on-legs) (Ha et al., CoRL 2024) | **Go2 + ARX5** | Yes, MIT (Isaac Gym; Go2 + ARX5 configs and ARX5 SDK) | Manipulation-centric WBC tracks EE trajectories from a Diffusion Policy; >70% success on all tasks |
| T2 | [RoboDuet](https://arxiv.org/html/2403.17367v4) (Pan et al.) | Two cooperating policies (loco + arm) | Go1/Go2 + arms, cross-embodiment | [Yes](https://github.com/locomanip-duet/RoboDuet) (Isaac Gym; Go2 + ARX5 deploy repo) | +42.5% average success over baseline; zero-shot transfer across quadrupeds |
| T2 | [VBC](https://web3.arxiv.org/pdf/2403.16967v3) (Liu et al., CVPR 2024) | Low-level WBC + visual high-level EE proposer | B1 + Z1 | Project page | Hierarchical visual whole-body grasping |
| T2 | [TA-WBC](https://arxiv.org/abs/2605.31343) (HKUST-GZ, May 2026) | Perceptive unified WBC with foot-centric terrain encoder | B2 + Z1 | Not found | 0.884 m³ / 0.025 m EE error on flat; stairs at 0.8 m/s while manipulating |
| T2 | [Multi-critic twist tracking](https://www.alphaxiv.org/abs/2507.08656) (ETH, 2025) | One policy, separate critics for loco and EE | ANYmal + arm | Not verified | Clean answer to loco-vs-arm reward conflict |
| **T3 Learned high-level → body command** | B3 body-command chunks into a T2 controller | [IIT Go2 VLA](https://arxiv.org/abs/2610.32271) (Minelli et al., Oct 2026) on MolmoAct2 | Go2 + AgileX Piper | Paper only; builds on [MolmoAct2](https://github.com/allenai/molmoact2), which is open | 11-D command at 5 Hz, 10-step chunks; 131→233 demos cut EE error 68.7→48.2 mm (−30%) |
| T3 | UMI-on-Legs high level | EE trajectory from Diffusion Policy | Go2 + ARX5 | Yes | IL-to-WBC interface already proven on our exact hardware |
| T3 | [FWBC-VLA](https://arxiv.org/abs/2609.03889) (ZJU / Shanghai AI Lab, Sep 2026) | VLA actions + force tokens + corrective whole-body actions | Wheeled-legged M20S | Not found | Sensorless force estimate injected into the VLA action expert; 5,000+ episode dataset |
| **T4 Fully end-to-end** | Joint-level action tokens from one VLA | [QUART](https://arxiv.org/pdf/2312.14457) (Ding et al., ECCV 2024) | Quadruped | Not verified | 12-D action tokens; 348K sim + 3K real episodes |

**Why T4 is a non-goal.** The IIT team reports their VLA's reasoning variant was too slow for closed-loop quadruped control and ran it with depth reasoning disabled at 5 Hz, delegating balance to an RL locomotion policy. `manipulator.md` already notes MobileVLA-R1 takes \~10–15 s per decision. A joint-level VLA would also bypass the safety filter's B1–B3 hooks.

**Recommended whole-body slots:** T1 (have it), T2 via UMI-on-Legs ported to Isaac Lab (same hardware, code public), T3 with Diffusion Policy first and a VLA (π0.5 or MolmoAct2) second, using the IIT 11-D command as the `BodyCommand` definition. This also resolves the open "arm role during locomotion training" item: T2 replaces the stowed-arm policy rather than sitting beside it.

## Configuration matrix and ablations

Six named configs cover the space with no new training code beyond what each backend already ships; each differs from its neighbour in one slot, so every pair is a clean ablation.

| Config | Rig | Locomotion (B1 consumer) | Arm / body | Producer of commands | Tests |
| --- | --- | --- | --- | --- | --- |
| **C0 Baseline** | R0 | A Isaac Lab PPO | T1 arm IK | Scripted orchestrator + M1 | Floor for every comparison |
| **C1 Modular perceptive** | R1 | B elevation map + height-scan policy | T1 | Planner + M1 / BC / DP | Classical-perception reference |
| **C2 Latent LiDAR** | R1 / R3 | D′ JEPLO | T1 | Same as C1 | Map vs mapping-free (C1 vs C2), one slot changed |
| **C3 Camera-only** | R2 | D WMP or DAWN | T1 | Same as C1 | Rig ablation (C2 vs C3) |
| **C4 Whole-body** | R1 + R4 | T2 UMI-on-Legs-style WBC | T2 | Diffusion Policy EE trajectories | Decoupled vs whole-body (C1 vs C4) |
| **C5 VLA whole-body** | R1 + R4 | T2 WBC | T2 | π0.5 / MolmoAct2 emitting B3 | IL vs VLA at the same boundary (C4 vs C5) |

**Ablations this enables, beyond those already in `perception.md`:**

1. **Map vs latent world model vs JEPA** (B vs D vs D′) on the same R1 rig, including JEPLO-style occlusion masks: answers whether the explicit map earns its SLAM dependency.
2. **Reconstructive vs non-reconstructive world model** (WMP/DAWN vs JEPLO): the cleanest sub-question for the world-model PhD thread.
3. **Classical vs IL vs VLA manipulation** (M1 vs DP vs π0.5) on the demo-count sweep, with M1 as the zero-demo point.
4. **Decoupled vs whole-body** (T1 vs T2): reachable workspace and EE error, the two metrics TA-WBC reports.
5. **Adversarial gate on each locomotion backend**: Phase 0.5's attacker is backend-agnostic because it attacks B1 and the push channel, so every locomotion slot gets the same robustness report.

**Compute note.** JEPLO trained in \~12 h on one RTX 6000 Pro; WMP needs \~23 GB GPU for 10k+ iterations; π0.5 LoRA needs >22.5 GB and the IIT MolmoAct2 run used one 96 GB GPU. All fit the single-rented-GPU budget if run one at a time.

## Repo implications and open questions

The tree needs one new component and a backend registry per policy slot; nothing existing moves.

- **`common/cerberus_msgs/`**: add `LocoCommand`, `ArmCommand`, `BodyCommand` (B1–B3).
- **`common/cerberus_policies/`** (new, mirrors `cerberus_worlds`): `PolicyBackend` base with `boundary`, `required_sensors`, `load()`, `act()`; backends chosen by config, e.g. `LOCO_BACKEND=jeplo`.
- **`policy/whole_body/`** (new, under `policy/`, since it composes locomotion and manipulation hierarchically per rule 5): T2 controller training + its own `eval/` and `adversarial_robustness/` (Phase 0.5 must re-run on `go2_arx5` anyway).
- **`policy/manipulation/algorithms/`**: add `classical/` (M1) and `vla/` (LeRobot-served) next to BC / DP / ACT.
- **`perception/terrain_perception/`**: hosts only map-based (B) encoders; latent encoders (D, D′, E) live with their locomotion backend because they are trained jointly with the actor.

**Open questions**

- [ ] JEPLO's code pins Isaac Lab 2.3.2.post1, our version; does its policy hold with the Go2's built-in L1 LiDAR instead of a Mid-360?
- [ ] Can UMI-on-Legs' IsaacGym WBC be ported to Isaac Lab cleanly, or is LeggedManip\_Lab's `GO2-ARX5` task a better T2 starting point?
- [ ] VBC, QUART and multi-critic code still unverified; CTS and PIE have no standalone public code.
- [ ] Which VLA to fine-tune first: π0.5 (ecosystem) or X-VLA (cost)?
- [ ] Should C5 also test the VLA emitting B2 only (arm) while the planner keeps B1, to isolate the VLA's base-placement skill, which the IIT paper found was its main failure mode?

## Paper registry and sources

Every paper cited in this doc, with its code status as checked on Nov 6, 2026. "None found" means no repo was linked from the paper, project page or a search.

| Paper | arXiv | alphaXiv | Code |
| --- | --- | --- | --- |
| Rudin et al., Learning to Walk in Minutes (2022) | [2109.11978](https://arxiv.org/abs/2109.11978) | [link](https://www.alphaxiv.org/abs/2109.11978) | [Isaac Lab](https://github.com/isaac-sim/IsaacLab) |
| CTS (Wang et al., 2024) | [2405.10830](https://arxiv.org/abs/2405.10830) | [link](https://www.alphaxiv.org/abs/2405.10830) | None found standalone; inside [JEPLO](https://github.com/ASIG-X/JEPLO) |
| Miki et al., Perceptive locomotion in the wild (2022) | [2201.08117](https://arxiv.org/abs/2201.08117) | [link](https://www.alphaxiv.org/abs/2201.08117) | Policy: none; recipe = Isaac Lab height-scan task |
| Miki et al., Elevation mapping on GPU (2022) | [2204.12876](https://arxiv.org/abs/2204.12876) | [link](https://www.alphaxiv.org/abs/2204.12876) | [elevation\_mapping\_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) |
| AME-2 (2026) | [2601.08485](https://arxiv.org/abs/2601.08485) | [link](https://www.alphaxiv.org/abs/2601.08485) | None found |
| DELTA (2026) | [2608.22033](https://arxiv.org/abs/2608.22033) | [link](https://www.alphaxiv.org/abs/2608.22033) | None found |
| Extreme Parkour (2024) | [2309.14341](https://arxiv.org/abs/2309.14341) | [link](https://www.alphaxiv.org/abs/2309.14341) | [chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour) |
| PIE (2024) | [2408.13740](https://arxiv.org/abs/2408.13740) | [link](https://www.alphaxiv.org/abs/2408.13740) | None found |
| WMP (2024) | [2409.16784](https://arxiv.org/abs/2409.16784) | [link](https://www.alphaxiv.org/abs/2409.16784) | [ByteDance/WMP](https://gitee.com/ByteDance/WMP) |
| DAWN (IROS 2026) | [2609.29092](https://arxiv.org/abs/2609.29092) | [link](https://www.alphaxiv.org/abs/2609.29092) | [DocyNoah/dawn-parkour](https://github.com/DocyNoah/dawn-parkour) |
| JEPLO (2026) | [2609.15770](https://arxiv.org/abs/2609.15770) | [link](https://www.alphaxiv.org/abs/2609.15770) | [ASIG-X/JEPLO](https://github.com/ASIG-X/JEPLO) |
| Robotic World Model (2025) | [2501.10100](https://arxiv.org/abs/2501.10100) | [link](https://www.alphaxiv.org/abs/2501.10100) | [rsl\_rl\_rwm](https://github.com/leggedrobotics/rsl_rl_rwm) |
| RWM-O (2025) | [2504.16680](https://arxiv.org/abs/2504.16680) | [link](https://www.alphaxiv.org/abs/2504.16680) | [rsl\_rl\_rwm](https://github.com/leggedrobotics/rsl_rl_rwm) |
| Omni-Perception (CoRL 2025) | [2505.19214](https://arxiv.org/abs/2505.19214) | [link](https://www.alphaxiv.org/abs/2505.19214) | [Partial](https://github.com/aCodeDog/OmniPerception): LiDAR sim only |
| PointWalker (2026) | [2610.15548](https://arxiv.org/abs/2610.15548) | [link](https://www.alphaxiv.org/abs/2610.15548) | None found |
| FoundationPose (2024) | [2312.08344](https://arxiv.org/abs/2312.08344) | [link](https://www.alphaxiv.org/abs/2312.08344) | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) |
| GraspGen (2025) | [2507.13097](https://arxiv.org/abs/2507.13097) | [link](https://www.alphaxiv.org/abs/2507.13097) | [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen), [GraspGenX](https://github.com/NVlabs/GraspGenX) |
| cuRobo (2023) | [2310.17274](https://arxiv.org/abs/2310.17274) | [link](https://www.alphaxiv.org/abs/2310.17274) | [NVlabs/curobo](https://github.com/NVlabs/curobo) |
| robomimic (2021) | [2108.03298](https://arxiv.org/abs/2108.03298) | [link](https://www.alphaxiv.org/abs/2108.03298) | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) |
| Diffusion Policy (2023) | [2303.04137](https://arxiv.org/abs/2303.04137) | [link](https://www.alphaxiv.org/abs/2303.04137) | [real-stanford/diffusion\_policy](https://github.com/real-stanford/diffusion_policy), LeRobot |
| ACT / ALOHA (2023) | [2304.13705](https://arxiv.org/abs/2304.13705) | [link](https://www.alphaxiv.org/abs/2304.13705) | [tonyzhaozh/act](https://github.com/tonyzhaozh/act), LeRobot |
| 3D Diffusion Policy (2024) | [2403.03954](https://arxiv.org/abs/2403.03954) | [link](https://www.alphaxiv.org/abs/2403.03954) | [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) |
| π0.5 (2025) | [2504.16054](https://arxiv.org/abs/2504.16054) | [link](https://www.alphaxiv.org/abs/2504.16054) | [openpi](https://github.com/Physical-Intelligence/openpi), LeRobot |
| GR00T N1 (2025) | [2503.14734](https://arxiv.org/abs/2503.14734) | [link](https://www.alphaxiv.org/abs/2503.14734) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| X-VLA (2025) | [2510.10274](https://arxiv.org/abs/2510.10274) | [link](https://www.alphaxiv.org/abs/2510.10274) | [2toinf/X-VLA](https://github.com/2toinf/X-VLA) |
| SmolVLA (2025) | [2506.01844](https://arxiv.org/abs/2506.01844) | [link](https://www.alphaxiv.org/abs/2506.01844) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| MolmoAct2 (2026) | [2605.02881](https://arxiv.org/abs/2605.02881) | [link](https://www.alphaxiv.org/abs/2605.02881) | [allenai/molmoact2](https://github.com/allenai/molmoact2) |
| VLA-JEPA (2026) | [2602.10098](https://arxiv.org/abs/2602.10098) | [link](https://www.alphaxiv.org/abs/2602.10098) | Served via LeRobot in the AllenAI harness; repo not checked |
| UMI-on-Legs (CoRL 2024) | [2407.10353](https://arxiv.org/abs/2407.10353) | [link](https://www.alphaxiv.org/abs/2407.10353) | [real-stanford/umi-on-legs](https://github.com/real-stanford/umi-on-legs) |
| RoboDuet (RA-L 2025) | [2403.17367](https://arxiv.org/abs/2403.17367) | [link](https://www.alphaxiv.org/abs/2403.17367) | [locomanip-duet/RoboDuet](https://github.com/locomanip-duet/RoboDuet) |
| VBC (2024) | [2403.16967](https://arxiv.org/abs/2403.16967) | [link](https://www.alphaxiv.org/abs/2403.16967) | Not verified |
| TA-WBC (2026) | [2605.31343](https://arxiv.org/abs/2605.31343) | [link](https://www.alphaxiv.org/abs/2605.31343) | None found |
| Multi-critic EE twist tracking (2025) | [2507.08656](https://arxiv.org/abs/2507.08656) | [link](https://www.alphaxiv.org/abs/2507.08656) | Not verified |
| From Tabletop to Walking Bot (IIT, 2026) | [2610.32271](https://arxiv.org/abs/2610.32271) | [link](https://www.alphaxiv.org/abs/2610.32271) | None (builds on MolmoAct2) |
| FWBC-VLA (2026) | [2609.03889](https://arxiv.org/abs/2609.03889) | [link](https://www.alphaxiv.org/abs/2609.03889) | None found |
| QUAR-VLA / QUART (ECCV 2024) | [2312.14457](https://arxiv.org/abs/2312.14457) | [link](https://www.alphaxiv.org/abs/2312.14457) | Not verified |

Other pages used: [LeggedManip\_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab), [AllenAI vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness), [LeRobot v0.4.0 release](https://huggingface.co/blog/lerobot-release-v040), [open-weight VLA comparison (presenc.ai)](https://presenc.ai/research/best-open-weight-robotics-vla-models-2026), [DAWN project page](https://dawn-parkour.github.io/), [RoboDuet project page](https://locomanip-duet.github.io/).
