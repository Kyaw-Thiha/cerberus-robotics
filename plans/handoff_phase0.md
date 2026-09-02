# Cerberus — Phase 0 handoff: Locomotion (RL)

Read `docs/cerberus_project_context.md` and `docs/project_structure.md` first for
full context. This doc is the actionable spec for this phase only.

## Objective

Prove the whole remote-compute + Isaac Lab pipeline runs end to end, and
produce a trained Go2 locomotion policy plus a quantified push-recovery
robustness curve. This phase is infrastructure validation, not the project's
novel contribution — keep it close to the standard published recipe rather
than adding technique. (The project's actual robustness contribution comes
next, in the separate Phase 0.5 handoff, which this phase's output feeds
into.)

## Scope

### 1. Remote environment
- Write `docker/Dockerfile.remote`: Isaac Sim + Isaac Lab base image, pinned
  `rsl_rl` and other extras. Reusable across providers (Brev, Lambda, RunPod,
  vast.ai) — don't hardcode anything provider-specific into the image.
- Nothing Isaac Sim/Isaac Lab related runs locally under any circumstances —
  see `docs/isaac_lab_workflow.md` for why (hardware floor, not a driver
  issue). All execution happens via SSH/mosh + tmux on the remote box.
- Verify with a headless smoke test (few hundred steps, small env count,
  finishes in seconds) before doing anything else — this is the actual test
  of whether the environment is correctly configured.

### 2. Baseline task
- Pull `isaaclab-go2-locomotion` and read its config end to end (reward
  terms, PPO hyperparameters, network architecture, curriculum trigger
  logic) before writing any training code.
- Use Isaac Lab's built-in `Isaac-Velocity-Flat-Unitree-Go2` task, unmodified,
  as the literal starting point. Getting this running is itself the smoke
  test that the remote env is correctly configured — don't move to the next
  step until this trains a sane reward curve with no NaNs.
- Algorithm: PPO via `rsl_rl`, following Rudin et al. 2022's massively
  parallel recipe (thousands of parallel environments, small `n_steps` per
  robot, large batch size — see the paper's Section 2.2 for the specific
  hyperparameter reasoning if anything is unclear; `isaaclab-go2-locomotion`
  is the primary practical reference, `legged_gym`, Rudin et al.'s own
  open-sourced code, is the fallback if something is undocumented in the
  above two).

### 3. Terrain curriculum
- Move from `Flat` to `Rough` task variant using Isaac Lab's existing
  curriculum machinery. This is configuration, not something to build from
  scratch.
- Confirm the policy reaches the curriculum's highest difficulty level
  across all terrain types before calling this step done (per Rudin et al.,
  robots looping back to random lower levels once they've solved the
  hardest is the intended signal that training has converged).

### 4. Push recovery (standard, Rudin-recipe)
- Training-time: pushes every 10s, accelerating the base up to ±1 m/s in
  x/y (Rudin et al.'s published values — adjust only if Go2-specific tuning
  in `isaaclab-go2-locomotion` suggests otherwise, and note the deviation if
  so).
- Eval-time: don't just test at the trained push magnitude. Sweep a range of
  magnitudes, including some above the trained range, at each terrain
  difficulty level. Define recovery as: base roll/pitch returns within
  bounds and commanded-velocity tracking resumes within a short fixed
  window (a few seconds), without a base collision or fall-triggered reset.
  Use the same fall/reset criteria already used as training termination
  conditions — don't invent a separate definition here.
- N trials per (magnitude, terrain-difficulty) cell — 100 is a reasonable
  default (matches published practice in this space); adjust down only if
  compute budget genuinely requires it, and note if so.
- Output: a success-rate curve (one line per terrain difficulty, x-axis =
  push magnitude), not a single on/off number.

### 5. Repo scaffolding
Per `docs/project_structure.md`:
```
policy/locomotion/
├── train.py / play.py
├── eval/                  # this phase's push-recovery curve + plotting script
├── checkpoints/            # go2_locomotion_v1_candidate.pt lands here
└── (adversarial_robustness/ — NOT this phase's responsibility, see Phase 0.5 handoff)
```
Also touch, if not already present: `pixi.toml`, `docker/Dockerfile.remote`,
`common/cerberus_description/` (Go2 URDF/USD).

### 6. `locomotion_policy` ROS2 inference node
- Thin ROS2 node wrapping the frozen checkpoint (once it exists — see "Done
  when" below for what "frozen" means at this stage).
- **Important**: at the end of Phase 0, the checkpoint is a *candidate*, not
  final. Name it `go2_locomotion_v1_candidate.pt`, not `_final.pt`. The
  ROS2 inference node can be built and tested against the candidate, but
  don't treat the candidate as the checkpoint Phase 1 will build on — that
  only happens after it passes the Phase 0.5 gate (separate handoff).

## Explicit non-goals for this phase
- No adversarial robustness testing — that's Phase 0.5, a separate handoff,
  and depends on this phase's candidate checkpoint as input.
- No cross-morphology / generalization work (different robots, unseen
  bodies) — out of scope for the whole project at this stage, not just this
  phase.
- No vertical/dynamic-terrain perturbation — considered and deferred (tooling
  mismatch with Isaac Lab's static height-field terrain system; see project
  context doc).
- Don't hand-tune the reward function beyond what `isaaclab-go2-locomotion`
  or Rudin et al. already establish, unless something is actually broken.
  This phase's value is in getting the standard recipe running correctly and
  producing a real eval artifact, not in reward engineering.

## Done when
- [ ] Remote env reproducible via `Dockerfile.remote`, headless smoke test
      passes in seconds.
- [ ] Flat-terrain PPO policy trained, visibly walking (video).
- [ ] Rough-terrain curriculum policy trained, reaches highest difficulty
      level.
- [ ] Push perturbations integrated into training per the Rudin recipe.
- [ ] Push-recovery eval sweep implemented and run (magnitude × terrain
      difficulty grid, N=100/cell or documented deviation).
- [ ] Robustness curve plot produced and saved under `policy/locomotion/eval/`.
- [ ] Checkpoint saved as `policy/locomotion/checkpoints/go2_locomotion_v1_candidate.pt`
      and synced to object storage (per `docs/isaac_lab_workflow.md`'s
      disposable-instance guidance).
- [ ] `locomotion_policy` ROS2 inference node running against the candidate
      checkpoint.

**Handing off to Phase 0.5 next** — do not proceed to Phase 1 (freezing this
checkpoint under a high-level navigation policy) until the separate Phase 0.5
handoff has run against this candidate and produced a `_final.pt`.
