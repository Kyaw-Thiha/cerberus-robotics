# Cerberus Robotics

Cerberus is an end-to-end, learning-based robotics stack for a quadruped loco-manipulator, built and evaluated entirely in
simulation on Isaac Sim / Isaac Lab. It is a **platform**: locomotion, manipulation, whole-body and sensor-rig paradigms plug
in as swappable backends behind three command boundaries, and every backend carries an evidence-backed support level.

**Caveat: simulation only.** There is no real hardware. "Real-world demo" here means a well-instrumented simulation demo, not
a claim of physical deployment.

## Platform

- **Robot:** Unitree Go2 (legs) + ARX5 / X5A (6-DoF arm + motorised gripper), fused into a 20-DoF embodiment
  (`common/cerberus_description/`, see `docs/embodiment_asset_composition.md`).
- **Middleware:** ROS2 Jazzy nodes on a pixi + colcon workspace (the repo root is the colcon workspace).
- **Sim stack:** Isaac Sim 6.2 / Isaac Lab 3.0 GA on `docker/remote_v6` (legacy: Sim 5.1 / Lab 2.3.2 on `docker/remote`).

## Missions and scenes

Primary mission: household activities (navigate, pick, place; cabinet/drawer opening as a secondary primitive). Retail,
restaurant and office come from GRScenes' commercial scenes. Scenes sit behind `common/cerberus_worlds/` (`docs/worlds.md`):

- **GRScenes-100** (69 home + 30 commercial, plain USD) through the `plain_usd` backend, for navigation.
- **Lightwheel RoboCasa/LIBERO kitchens** through Isaac Lab Arena (`arena` backend), for manipulation and Mimic demos.

## Phase status (2026-11-06)

| Phase | What | Status |
|---|---|---|
| 0 | Go2 locomotion RL (PPO, push curriculum) | **Trained** (Flat + Rough checkpoints, push-recovery eval); not gated or frozen |
| 0.5 | Adversarial robustness gate | **Not started** (spec only: `plans/handoff_phase0.5.md`) |
| 1 | Navigation: SLAM, planner, composed policy | Partial: 6 SLAM backend adapters and a sim ground-truth bridge; planner/perception are registries only |
| 2 | Manipulation (IL, Mimic) | Partial: Go2+ARX5 Mimic env; fixed-base W2a passes, Lightwheel-kitchen W2b not passed |
| 3 | Safety filter (CBF/shielding) | Not started |
| 4 | Continual learning + world model | Not started |

Interface layer (`LocoCommand` / `ArmCommand` / `BodyCommand` messages, `PolicyBackend` in `common/cerberus_policies/`): `docs/interfaces.md`.
Full audit: `docs/status_report_2026-11-06.md`.

## Borrowed vs novel

Borrowed, not invented: the locomotion recipe (Isaac Lab PPO, Rudin et al. 2022), the Go2+ARX5 combination (LeggedManip_Lab,
ODYSSEY precedent), the scene libraries (GRScenes, Lightwheel/RoboCasa via Arena), and the manipulation and perception
algorithms. The contribution is the integration (RL locomotion, IL manipulation, a safety filter and continual learning on one
platform in a household setting, as far as we can tell not published elsewhere) plus evaluation artifacts: a push-recovery
robustness curve, an adversarial vulnerability report (planned), and a safety-filter ablation (planned).

## Supported backends

Each backend has a card and a support level; we do not train everything.

| Level | Meaning | Evidence required |
|---|---|---|
| L0 | Documented | Backend card only, no code |
| L1 | Integrated | Adapter exists and passes contract tests |
| L2 | Evaluated | Run on our scenes or shared evals (released weights or no training); numbers recorded |
| L3 | Trained by us | Trained and evaluated by us; result recorded in a report |

Backends are **modular** (consume a map or other module output) or **fused** (raw sensor streams, encoder trained inside the
policy, e.g. JEPLO, DAWN, Omni-Perception); both plug in at the same boundary.

Current backends, levels and targets: [`docs/backend_support.md`](docs/backend_support.md) (generated from
`common/cerberus_policies/cards/`; interfaces in [`docs/interfaces.md`](docs/interfaces.md)).

## Key docs

- `docs/cerberus_project_context.md`: motivation, mission, phases, open decisions
- `docs/project_structure.md`: repo layout and conventions
- `docs/interfaces.md`, `docs/backend_support.md`: the three command boundaries and per-backend support levels
- `docs/modular_stack_options.md`: locomotion / manipulation / whole-body options and the three boundaries
- `docs/worlds.md`, `docs/arena_integration_log.md`: scene sources and the Arena gate log
- `docs/isaac_lab_workflow.md`, `docs/runpod.md`, `docs/artifact_storage.md`: remote GPU workflow

## Quick start

```bash
pixi run lint     # ruff
pixi run test     # pytest (no Isaac needed; Isaac-only tests skip locally)
pixi run build    # colcon build --symlink-install
```
