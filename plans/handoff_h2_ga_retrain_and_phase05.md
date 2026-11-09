# Handoff H2: GA Retrain + Parity, Phase 0.5 Adversarial Gate, Scene Eval

Save as `plans/handoff_h2_ga_retrain_and_phase05.md`. Runs on a **remote GPU pod** using `docker/remote_v6` (Isaac Sim 6.2 / Isaac Lab 3.0 GA). Branch from `main` **after H1 is merged**: `h2-phase0-ga-and-phase05`. One commit per work package. Platform: **legs-only Unitree Go2** (no arm; the arm is Phase 2, and the gate must be re-run on `go2_arx5` later).

> Execution note (2026-11-06): H1 was NOT merged into `main` when H2 started (`main` = 6b2ed37, H1 branch
> `h1-docs-interfaces` = 259c6a9), and no RunPod credentials were available on the workstation. The branch was
> therefore cut from `h1-docs-interfaces`, and only pod-independent work was done. See `docs/h2_report.md`.

## Why this exists

Phase 0 checkpoints (flat and rough) were trained in September, before the repo moved to Lab 3.0 GA. A checkpoint stores weights only, not the physics engine, so it may or may not behave the same on the new stack. Known risks: quaternion convention change (WXYZ to XYZW) in custom code, rsl_rl 3.1.2 to 5.x checkpoint layout (separate actor/critic state dicts), and any Go2 asset or actuator changes between Lab versions. Decision (made with the project owner): **retrain the rough Go2 task on the GA stack now**, and also run the September checkpoint through the same eval for a parity reference. The GA-trained checkpoint becomes the candidate for the Phase 0.5 adversarial gate, which must pass before the checkpoint is frozen for Phase 1.

Deliverables, all first-class artifacts:
1. `docs/reports/phase0_ga_parity.md`: September vs GA numbers, side by side.
2. A GA-trained candidate checkpoint with a metadata sidecar and ONNX export.
3. `docs/reports/phase05_adversarial_report.md`: standard-test results vs adversarial-attack results, plus before/after if robustification ran.
4. `go2_locomotion_final.pt` **only if the gate passes**.
5. A short scene-eval report (non-blocking).

## Read first (in this order)

`plans/handoff_phase0.5.md` (the Phase 0.5 spec; follow it), `docs/cerberus_project_context.md` (Phase 0 and 0.5 sections), `docs/project_structure.md` (adversarial_robustness layout, checkpoint naming), `docs/isaac_lab_workflow.md`, `docs/runpod.md`, `docs/artifact_storage.md`, `policy/locomotion/runs/RUN_001.md`, `RUN_002.md`, and `policy/locomotion/core/platforms/go2/{platform.py,env_cfg.py}` plus `train.py`, `play.py`, `eval/push_recovery_eval.py`.

## Ground rules

1. **Do not modify the Go2 rewards, PPO hyperparameters or network architecture** in the retrain. It must be the same recipe as RUN_002, run on the new stack. Any needed GA-compat change must be a minimal, documented fix, not a tuning change.
2. **Physics backend must be explicit PhysX** (`isaacsim_physx`). Record the resolved physics, asset and action/observation config in the checkpoint sidecar. Stock defaults to Newton on Lab 3.0; do not let that happen silently.
3. **Never overwrite an existing checkpoint.** New files only. Do not rename the September checkpoints.
4. **Sync before you stop.** At the end of each work package run `scripts/sync_to_r2.py` for checkpoints, logs, eval CSVs and videos, then confirm the manifest. Instances are disposable; unsynced work is lost.
5. **Use a headless smoke test before every long run** (64 envs, about 200 steps, no NaN, reward curve exists, no shape errors).
6. Use tmux sessions per context (`dev`, `train`). Do not run training in the dev session.
7. **Cost control.** Every long run has a wall-clock cap, defined relative to what the repo already knows (see each package). If a cap is exceeded, stop, sync, and report instead of continuing. Do not launch parallel seeds or sweeps beyond what is listed.
8. Never print `.env` contents. Credentials come from the environment on the pod.
9. If something in this handoff conflicts with what you find, report the conflict and stop at the next safe point; do not silently resolve it.

## Work package P0: Pod bring-up and stack report

- Bring up the pod per `docs/runpod.md` using `docker/remote_v6`. Clone the repo, check out `h2-phase0-ga-and-phase05`.
- Write `outputs/env_report.json`: Isaac Sim, Isaac Lab (commit), rsl_rl, torch, CUDA, GPU model and VRAM, Python, `git rev-parse HEAD`. Commit a copy as `docs/reports/env_report_<date>.json`.
- Run the existing headless smoke for the Go2 rough task. Confirm the resolved physics backend is PhysX.
- Add a tiny check (script or test) asserting the quaternion convention on the GA stack: a standing Go2's `projected_gravity_b` observation is approximately `[0, 0, -1]`, and note which convention `robot.data.root_quat_w` uses. This becomes input for the ROS node work in H3.
- **Gate P0:** smoke passes, physics is PhysX, env report committed.

## Work package P1: Parity eval of the September checkpoint on GA

- Goal: does `go2_locomotion_v1_candidate` (rough, 2026-09-08) behave the same on the GA stack?
- Loading: first try loading the `.pt` through the GA rsl_rl runner. If it raises (expected risk: `KeyError` on `actor_state_dict`/`critic_state_dict`), fall back to the `.onnx` export (`go2_locomotion_v1_candidate.onnx`) through onnxruntime wrapped as a policy callable. Record which path worked. Do **not** write a long-lived converter; if a tiny one-off state-dict conversion is trivial, save it under `policy/locomotion/eval/` with a note, otherwise skip.
- Contract check before evaluating: print and compare against the September run config (in `runs/` params or RUN_001/RUN_002): observation dim and order, action dim and scale, joint ordering, decimation, physics dt, command ranges, normalizer presence. Any mismatch is reported before eval numbers.
- Eval: run `policy/locomotion/eval/push_recovery_eval.py` with the **same grid and trial counts as** `runs/2026-09-08_19-28-32_analysis/eval/eval_rough_full_sweep.csv` (100 trials per cell). Output a CSV next to it.
- Parity criteria (proposed; record the numbers either way): per-cell success within 10 percentage points of the September CSV, and mean across cells within 5 points. Reference cells: 94% at 0 N / level 0, 89% at 120 N / level 0, 61% at 120 N / level 8.
- Output: `docs/reports/phase0_ga_parity.md` with the contract check, the side-by-side table, a plot, and a verdict (PARITY HOLDS / PARITY BROKEN / COULD NOT LOAD).
- **Gate P1:** the report exists. A broken or unloadable result does **not** stop the handoff; the retrain proceeds regardless.

## Work package P2: GA retrain of the Go2 rough task

- Replicate the RUN_002 recipe exactly (iterations, num_envs, seed policy, task `Isaac-Velocity-Rough-Unitree-Go2-Cerberus-v0`). Read RUN_002 for the real numbers and use them. **Wall-clock cap: 2x RUN_002's recorded training time.** One seed only.
- Smoke first, then the full run in the `train` tmux session. Save tensorboard logs, and a rollout video at intervals.
- Checkpoint naming (do not collide with September files): `go2_locomotion_ga_v1_candidate.pt`, plus `go2_locomotion_ga_v1_candidate.onnx` (use the GA rsl_rl export path) and a sidecar `go2_locomotion_ga_v1_candidate.meta.json` containing: git hash, env report, task id, resolved physics backend, iterations, seed, num_envs, config hash, wall-clock, final reward and terrain-level stats. Add a short note to `docs/project_structure.md` explaining the `_ga_` tag in the checkpoint naming scheme.
- Evaluate with the same push-recovery sweep as P1; add the GA row to `docs/reports/phase0_ga_parity.md` and a comparison plot of September vs GA.
- Decision rule: the GA checkpoint is the Phase 0.5 candidate regardless of the P1 verdict. **If the GA candidate is more than 10 points below the September mean, or fails to converge within the cap, stop here, sync, and report; do not start P3.** The human decides what to do next.
- **Gate P2:** candidate checkpoint, ONNX, sidecar and eval CSV exist; report updated; synced to R2.

## Work package P3: Phase 0.5 adversarial robustness gate

Follow `plans/handoff_phase0.5.md` for the method (small MLP adversary trained with PPO against the frozen candidate; perturbation-force + command-space attack; observation attack is a deferred stretch goal) and for the reward formula, architecture and hyperparameter ranges. Layout per `docs/project_structure.md`:

```
policy/locomotion/adversarial_robustness/
├── train_adversary.py
├── attack_env.py
└── eval/
```

Design requirements beyond the spec:
- **Backend-agnostic by construction.** `attack_env.py` takes the frozen policy as a callable (`obs -> actions`) and injects perturbations through two hooks: a base push force channel and a velocity-command channel. Do not import the locomotion training task's runner into the wrapper. This lets later locomotion backends (blind, map-based, world-model) reuse it unmodified.
- Keep the pure-Python parts (perturbation bounds and clamping, the minimal-magnitude metric, gate decision logic) in a separate module with local pytest tests that need no Isaac.
- The "realistic" perturbation bounds must come from the Hwangbo push curriculum in `env_cfg.py` and the command ranges the policy was trained with; read them from config, do not hardcode.

Steps and gates:
1. **P3a: wrapper smoke.** Wrapper env runs the frozen candidate with the adversary emitting zero perturbation; reproduces the standard-test success rate within noise. 64-env smoke, no NaN.
2. **P3b: standard-test baseline.** Reuse the P2 sweep numbers as the "standard" column of the report.
3. **P3c: adversary training.** Cap iterations per `handoff_phase0.5.md`; wall-clock cap: the larger of 3 hours or what the handoff states. Log adversary reward, fall rate, and minimal perturbation found.
4. **P3d: gate evaluation.** Compute the minimum combined force + command magnitude that induces a fall. Gate: that magnitude must **exceed** the push range the candidate was trained and evaluated against. Emit `eval/gate_verdict.json` with the numbers and PASS or FAIL. Also run the attack **per modality** (force-only, command-only, combined) so the report can show single-modality attacks failing where the combination succeeds, as the source paper predicts.
5. **P3e: robustify loop (only if FAIL).** Fine-tune the candidate against the discovered adversarial scenarios, re-attack, up to **3 rounds**. Name checkpoints `go2_locomotion_v1_robustified.pt` (then `_v2_candidate`, etc., per `project_structure.md`). rsl_rl 5.x resume: if resuming from a checkpoint needs the new actor/critic state layout, use checkpoints produced on GA only; do not resume from September files. After 3 failed rounds, stop and report.
6. **P3f: promotion.** On PASS only: write `go2_locomotion_final.pt` (+ `.onnx` + `.meta.json` carrying the gate verdict and which candidate it came from). On FAIL after the loop: **do not** create `go2_locomotion_final.pt`.

- Write `docs/reports/phase05_adversarial_report.md`: standard-test vs adversarial side by side (table + plot), per-modality results, the gate verdict, and the before/after comparison if robustification ran. State plainly which findings are expected from the source paper and which are ours.
- **Gate P3:** adversarial report exists, `gate_verdict.json` exists, and either `go2_locomotion_final.pt` exists (PASS) or a clear FAIL report with the robustify history exists. All synced to R2.

## Work package P4: Scene eval (non-blocking)

Purpose: sanity-check the frozen policy outside procedural terrain, and exercise `plain_usd` with a walking robot.
- Candidate for the policy: `go2_locomotion_final.pt` if it exists, otherwise the GA candidate (say which).
- Fetch 2 or 3 GRScenes homes with `scripts/fetch_grscenes_from_r2.py`, run the `plain_usd` preprocess for navigation, and load via `cerberus_worlds`. Per the earlier W1 spike, a ground-plane control works and GRScenes via `plain_usd` has passed; do not use the Arena path here.
- Drive a **scripted velocity path** (forward, turns, a loop through doorways) with no perception. Record per scene: fall rate over N trials, fraction of path completed, any physics blow-ups.
- Write `docs/reports/scene_eval_phase0.md`. Do not tune anything based on the result in this handoff.
- If scene fetch or preprocess fails, record exactly where and stop this package. **A P4 failure does not block P0-P3 deliverables.**

## Final report (print in your last message and save as `docs/h2_report.md`)

1. Commits per package, one line each.
2. Gate results P0-P4 with key numbers (parity verdict, GA vs September success rates at the three reference cells, gate verdict and minimum perturbation vs trained range).
3. GPU wall-clock actually used per package versus caps.
4. Everything synced to R2: bucket paths and manifest.
5. Findings to feed H3 (quaternion convention result from P0, any observation or ordering mismatches found in P1, the rsl_rl resume behaviour).
6. Anything you could not do, and why. Anything that contradicts this handoff.

## Out of scope (do not start)

Arm or `go2_arx5` training, perception or SLAM, any JEPLO or other new locomotion backend, `PolicyBackend` adoption in the ROS node (that is H3), observation-space attacks, Arena W1 rerun, Lightwheel scenes, VLA or manipulation work, hyperparameter sweeps.
