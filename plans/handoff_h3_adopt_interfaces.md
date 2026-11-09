# Handoff H3: Adopt the Interfaces (ONNX backend, B1 slew limiter, ROS node, docs truth-up)

**Local only: no pod, no GPU, no Isaac imports.** Work **directly on `main`** (see Git rules). One commit per work package, each passing `pixi run pytest -q`, `pixi run lint`, and the backend-table freshness check before you move on. Push at the end of each package.

## Orientation for a new agent (read first, in order)

You are joining mid-project. Cerberus is a simulated Unitree Go2 (+ ARX5 arm later) learning stack in Isaac Lab. Read: `README.md`, `docs/project_structure.md`, `docs/interfaces.md`, `docs/backend_support.md`, `docs/h1_report.md`, `docs/h2_report.md`, `docs/reports/phase0_ga_parity.md`, `docs/reports/phase05_adversarial_report.md`, `docs/modular_stack_options.md` (the backend menu; it predates the Lab 3.0 GA stack), and `common/cerberus_policies/` (`base.py`, `registry.py`, `backend_card.py`, `validate_config.py`, `cards/`). Then `policy/locomotion_policy/` (the ROS inference node) and `common/cerberus_compat/quat.py`.

Key facts from earlier work (verify against the files, do not trust this summary blindly):
- Active stack: Isaac Sim 6.2 / Isaac Lab 3.0 GA / rsl-rl 5.5.1, PhysX pinned. Local machine cannot run Isaac.
- Phase 0 is done on GA. The candidate `policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.{pt,onnx,onnx.data,meta.json}` (gitignored, may or may not be on this machine) **failed** the Phase 0.5 adversarial gate (sidecar says `gate: FAIL`). Robustification was deliberately not done. `go2_locomotion_final.pt` does not exist and must not be created here.
- The rough policy is **not blind**: its observation is 235-D = 48 proprioceptive/command terms + 187 height-scan cells from a simulator ray-caster (ground-truth terrain, small uniform noise). Only the flat policy is blind (48-D). Docs and cards must say so.
- ONNX export quirks: fixed batch of 1, weights in an external `<name>.onnx.data` file that must ship next to the graph.
- Attack finding: the adversary breaks the policy mainly through the **velocity command** (pinned at its clamp, sign flipping about 3.4 times per second), while force contributes only about 3 N. Training resamples commands every 10 s. A command slew-rate limiter at the B1 boundary is the hypothesised mitigation; its effect is **untested**.

## Git rules

- Local handoffs go straight to `main`. Start every session with `git fetch && git pull --ff-only`; if it cannot fast-forward, **stop and report**.
- Push after each work package. Never force-push. No history rewriting.
- Never read or print `.env`. No network calls except git remotes and `pixi` dependency resolution for work package A.
- If anything here contradicts the repo, report it and stop at the next safe point; do not silently resolve it.

## Work package A: Truth-up (docs, cards, dependency)

- A1. Declare `pyyaml` explicitly in `pixi.toml`, re-lock (`pixi.lock` updates), confirm lint, tests and CI still pass.
- A2. Rename the `blind_ppo` backend card to `heightscan_ppo`. Content: slot `loco`, boundary B1, architecture `modular` (perception-conditioned by a simulator-provided height map), paradigm text "rough-terrain PPO conditioned on a ground-truth height scan (privileged in sim)", required sensors = the scan source and IMU/joint encoders in the sensor-id naming already used. Support level and evidence per the rules in `backend_card.py`; this one has real training and eval evidence (`docs/reports/phase0_ga_parity.md`, `docs/reports/phase05_adversarial_report.md`), so set the level the validator allows, record the gate result (**FAIL**, 0.568x) in `results`, and set the target level honestly. Do **not** claim a level the evidence rules reject.
- A3. Add a `blind_ppo_flat` card at L0 (the flat policy is truly blind, 48-D; no GA-validated checkpoint yet). Facts only from existing docs; unknown fields stay `unknown`.
- A4. Clarify the support-level rules in `docs/interfaces.md` and in `backend_card.py` validation: **L2 and L3 also require the L1 evidence** (an adapter with a passing contract test). Add a failing-card test for it. Update any card that now violates it (set its level honestly or add its evidence).
- A5. Fix `docs/modular_stack_options.md`: paradigm A "Blind PPO" row should say the rough task is height-scan-conditioned (paradigm B with a privileged map) and that the blind variant is the flat task. One short note; do not rewrite the doc.
- A6. Update `README.md` phase table and status: Phase 0 trained on GA with parity numbers (September 79.3% mean push recovery, GA retrain 74.4%); Phase 0.5 run, **gate FAIL** at 0.568x the trained push range with the one-line cause (command-channel attack); not robustified by decision. Add the honest-claims line: "locomotion uses a simulator-provided, noisy height scan; an estimated-map variant is a planned ablation."
- A7. Add a short "Git workflow" section to `README.md` or `docs/isaac_lab_workflow.md`: local handoffs on `main`, remote/pod handoffs on a short-lived branch from a fresh `origin/main` clone, `pull --ff-only` at session start, push per package, never force-push.
- Regenerate `docs/backend_support.md` (`pixi run backend-table`).
- **Gate A:** pytest, lint, freshness check green; `grep -rn "blind_ppo"` finds only intentional references; README no longer says Phase 0.5 is unbuilt.

## Work package B: B1 command slew-rate limiter (pure Python)

- B1. Add `common/cerberus_policies/command_limiter.py`: `CommandRateLimiter` configured with per-axis `max_rate` (vx, vy: m/s^2; wz: rad/s^2) and per-axis `max_abs` clamp (training command ranges: about 1 m/s linear, 1 rad/s angular; read them from the H2 contract table or the saved env params, do not guess). Methods: `reset()`, `apply(command: dict, dt: float) -> dict`. Operates on the same plain command dict the backends emit (`LocoCommand` fields). Optional body pitch/height fields pass through (clamped only).
- B2. Look at the existing `RateLimitedChannel` in `policy/locomotion/adversarial_robustness/` first. It is attack-side (it bounds the adversary). Reuse logic if the semantics match, but **do not modify any adversarial-robustness behaviour**; the gate artifacts must stay reproducible. Note the relationship in a comment.
- B3. Backend card schema gets a new optional field `max_command_rate` (per-axis, with units) and `command_limiter_applied` (bool). `unknown` allowed at L0 only, like the other fields. Update the schema doc and the validator.
- B4. Defaults are **untested placeholders** (suggest 1.0 m/s^2 and 2.0 rad/s^2), loaded from config, not hardcoded. Say so in the docstring and in `docs/interfaces.md`. Do not claim they fix the gate.
- B5. Tests: step input ramps at the configured rate; full sign reversal takes the expected time; per-axis independence; clamp; `dt` handling; `reset()`; passthrough of optional fields; idempotence when already within limits.
- **Gate B:** tests green; limiter importable with no torch/ROS/Isaac.

## Work package C: ONNX-backed policy backend + observation contract

- C1. `common/cerberus_policies/contracts/heightscan_ppo_v1.yaml`: the exact observation contract for the rough policy: term order, dims (3+3+3+3+12+12+12+187 = 235, verify against `docs/h2_report.md` / the saved params, **use the real order**), action dim 12, action scale 0.25, default joint offsets by name, decimation 4 at dt 0.005, policy rate 50 Hz, command ranges, height-scan grid (17 x 11, 0.1 m, 1.6 x 1.0 m, yaw-aligned), height-scan clip, and joint order **by name** as used in training. Anything you cannot confirm from local files (for example the exact sign/offset convention of the height-scan observation term) must be marked `unverified` and listed under Deferred to remote. Do not guess.
- C2. `common/cerberus_policies/backends/heightscan_ppo_onnx.py`: a `PolicyBackend` (B1 consumer). `load()` takes the `.onnx` path, **fails with a clear error if the sibling `.onnx.data` file is missing**, and creates an onnxruntime CPU session. `act(observation)` accepts either the named observation terms (assembled in contract order) or a pre-assembled 235-vector, validates dims and dtype float32, runs **batch of 1 row by row**, and returns the 12 raw actions. A helper converts raw actions to joint-position targets by name: `target = default_joint_pos + scale * action`.
- C3. Quaternion and gravity: add `projected_gravity_from_quat(q_xyzw)` to `common/cerberus_compat/quat.py` if not present. **Test vector measured on the GA stack:** quaternion `[x, y, z, w] = [0.0645, -0.0453, 0.0007, 0.9969]` must give projected gravity approximately `[-0.090, -0.129, -0.988]` (tolerance 2e-3). Note: ROS `geometry_msgs/Quaternion` and `sensor_msgs/Imu.orientation` are already x,y,z,w, which is the same as Isaac Lab 3.0, so **no reordering should be needed between ROS and this policy**. The H2 report said the node "must convert to wxyz"; that was inferred. Check what the existing node code actually assumes. If it assumes w-first anywhere, fix it and add a regression test; report what you found either way.
- C4. Tests: observation assembly order and dims; wrong-dim rejection; missing `.onnx.data` error; action-to-target mapping; quaternion/gravity vector above; determinism (same input, same output); and **if the real `go2_locomotion_ga_v1_candidate.onnx` and its `.onnx.data` exist locally**, a smoke test running 100 random-but-plausible observations through it (finite outputs, shape (12,)). Skip with a clear reason when the files are absent. Never commit checkpoints.
- C5. Add and validate the card changes: `heightscan_ppo` now has an adapter and a contract test; update `evidence` accordingly. Register the backend in the registry under the `loco` slot and add it to the all-cards and no-direct-import tests.
- **Gate C:** tests green; importing the backend module does not import torch or ROS at module top level (onnxruntime lazily); card validates at the level you set.

## Work package D: ROS node adopts `LocoCommand` (final, gated)

- D1. Add `HeightScan.msg` to `cerberus_msgs`: `std_msgs/Header header`, `uint32 rows`, `uint32 cols`, `float32 resolution`, `float32[] data` (relative heights in the same convention and clip the policy was trained on; if that convention is `unverified` in the contract, say so in the msg comment). Rebuild `cerberus_msgs` to a `/tmp` prefix.
- D2. In `policy/locomotion_policy/`: split the node into a pure-Python core (observation assembly, limiter, backend call, action-to-target mapping) and a thin rclpy wrapper. The core has local tests; the wrapper only wires topics and parameters.
- D3. Node inputs: `LocoCommand` (new primary command input), IMU, joint states, and `HeightScan`. Keep the existing velocity-command input only as a deprecated adapter behind a parameter (default off) that converts it into a `LocoCommand`. Joint ordering **by name** from a launch-time YAML or the contract, never from array position. Apply `CommandRateLimiter` between the command input and the backend, with the limiter parameters coming from launch config. Outputs joint targets keyed by name.
- D4. Resolve the hardware IMU TODOs at `inference_node.py:153-155` only as far as the new topic wiring allows; leave real-hardware paths as explicit, documented TODOs.
- D5. Update `system/cerberus_bringup` launch/config to pass the new parameters, and update the sensor-suite validation if a new sensor id (height-scan source) is needed.
- D6. If an rclpy smoke test is feasible in this pixi/ROS Jazzy environment (init, publish one `LocoCommand`, IMU, joint states and `HeightScan`, receive one joint-target message with finite values), add it and mark it skipped when rclpy is unavailable. If it is not feasible, say why and list it under Deferred.
- **Gate D:** `colcon build` of the changed packages to `/tmp` succeeds; core tests green; node wrapper imports under ROS; all prior checks green.

## Work package E: Docs and handoff close-out

- Update `docs/interfaces.md` (B1 rate limiting, `max_command_rate`, the `heightscan_ppo_v1` contract, ONNX specifics, the new message), `docs/project_structure.md`, and regenerate `docs/backend_support.md`.
- Write `docs/h3_report.md` in the same format as the earlier reports (commits per package, gate outputs, design choices where this handoff was silent, deferred items, contradictions found). Include a **Deferred to remote** list; at minimum: height-scan convention verification against the live Lab term, an eval of the policy behind the limiter (attack the policy+limiter system), and a map-corruption eval of the height scan.
- Push and confirm CI is green on `main`.

## Out of scope (do not start)

Any pod or GPU work, robustifying or retraining any policy, creating `go2_locomotion_final.pt`, changing adversarial-robustness behaviour, SLAM or planner changes, Arena or scene work, JEPLO or other new locomotion backends, manipulation or VLA work, and tuning the limiter defaults.

## Session log (this execution)

- 2026-11-09: `git fetch` failed (`Permission denied (publickey)`; no SSH agent identities). Work proceeded locally on `main` from `5d18f9f`; pushes attempted per package and reported if they fail.
- Package A: pyyaml declared, cards renamed/added, cumulative level rule, docs/README truth-up, table regenerated.
- 2026-11-09 close-out: packages A–D committed locally (79ae1a9, 2da0328, e8c4eb4, e814722) plus this
  docs commit. All gates green locally; `git push` still fails (no SSH identity), so CI is unconfirmed.
  See `docs/h3_report.md` §5 for the deferred-to-remote list.
