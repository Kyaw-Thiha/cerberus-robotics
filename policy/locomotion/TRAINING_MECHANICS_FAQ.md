# Training/Eval/Video Mechanics — FAQ

Conceptual reference for how a Phase 0 locomotion run actually works under the
hood: what "training" is doing while it runs, what the reward is actually
optimizing for, and how eval/video capture differ from training. Written so a
future run's numbers (iteration counts, wall-clock time, reward values) can be
sanity-checked against a normal baseline instead of re-deriving all of this
from scratch. Concrete numbers below are taken from the
`unitree_go2_rough/2026-09-05_02-30-44` run (see `RUN_001.md`) as a worked
example — re-check `params/env.yaml` / `params/agent.yaml` for any given run
if these ever change.

## How long does training take, and what are we "waiting" for?

Not one robot walking around while we wait — **4096 copies of the robot
simulated in parallel on the GPU**, all at once (`num_envs: 4096`). Each
training "iteration" only advances every robot's simulated clock by
`num_steps_per_env` (24) control-steps x `dt` (0.005s) x `decimation` (4) =
0.02s per control-step, so **0.48 simulated seconds per iteration** — then the
network gets one gradient update from all that combined experience. Since all
4096 envs advance that same 0.48s in parallel, one iteration produces
4096 x 0.48s ≈ 33 minutes of cumulative robot-experience, and over 1500
iterations that's a cumulative ~3200+ hours of simulated walking experience —
compressed into this run's actual wall-clock time of **~90 minutes**
(5382.38s / 1500 ≈ 3.59s per iteration). Massively-parallel, headless
(no rendering) GPU physics is what makes that ratio possible; Flat trains
faster per iteration still (~1.2s vs Rough's ~3.5-4s), since Rough also has to
step the terrain generator and handle more complex contacts.

Checkpoints save every `save_interval` iterations (50) — a 1500-iteration run
leaves ~30 periodic snapshots plus the final one, plus the best-bucket
checkpoints (`model_best_easy/medium/hard.pt`, see `best_checkpoint_runner.py`).

A single episode (one robot's "life" before reset) lasts up to
`episode_length_s` (20.0) simulated seconds — 1000 control-steps — ending
early if that robot falls, or timing out at 20s otherwise. Both happen
constantly, independently, across all 4096 parallel envs the entire time
training runs; that churn is normal, not a bug.

## What is the reward actually for — is the goal "go somewhere"?

No destination point at all — this is a **velocity-tracking** task. Each
robot is given a random target forward/side velocity + turning rate
(resampled roughly every 10 sim-seconds), and reward is mostly:

- `track_lin_vel_xy_exp` / `track_ang_vel_z_exp` — the large positive terms:
  match the commanded velocity / turn-rate.
- A handful of small negative terms penalizing "cheating" ways of doing
  that: bouncing vertically (`lin_vel_z_l2`), wasting torque
  (`dof_torques_l2`), jerky actions (`action_rate_l2`, `dof_acc_l2`),
  unnatural foot timing (`feet_air_time`), hitting joint limits
  (`dof_pos_limits`).

So the goal is "walk/run/turn at whatever speed and direction you're told,
smoothly and without falling" — not navigation. Push disturbances and terrain
difficulty are **not** reward terms; they're separate stressors layered on
top via curricula (see `curriculum.py`, `push_disturbance_cfg.py`), and the
policy simply has to keep tracking velocity despite them.

## Is eval the same setup as training?

Different in one important way: **no learning happens during eval** — it's a
pure forward pass, no gradient computation. Each of up to
`len(magnitudes) * len(terrain_levels) * trials_per_cell` parallel envs gets
exactly **one fixed trial** (one push magnitude + one terrain difficulty level
assigned for its whole episode), runs for about `push_time_s` (3s) +
`window_s` (3s) ≈ 6 simulated seconds, and gets graded pass/fail on whether it
recovered (no fall, roll/pitch within bound, tracking error low for the tail
`sustain_s`). Since there's no backward pass, eval is much faster wall-clock
per env than training — the 2026-09-05 run's sub-terrain sweep (1000 envs: 2
magnitudes x 5 levels x 100 trials/cell) took only **~1 minute** end to end
(including Isaac Sim's own app-launch overhead).

## And video capture?

Also inference-only, same as eval, but slower per real-world-second because it
actually **renders RGB frames** — training and eval both skip rendering
entirely to stay fast (`--headless`, no `--enable_cameras`). Each clip is
short in simulated time (`play_terrain_showcase.py`'s default
`--steps-per-clip 200` x 0.02s = 4s per clip;
`push_recovery_showcase.py`'s clips run push_time_s + window_s ≈ 6s), but real
wall-clock time is dominated by Isaac Sim's one-time app boot (~1 min) plus
per-frame render cost, not the simulated duration. Worked example: 15
terrain-showcase clips took ~6.5 min wall-clock; 29 push-recovery-showcase
clips took ~10 min.

## Quick reference table

| | training | eval (`push_recovery_eval.py`) | video capture |
|---|---|---|---|
| Learning? | yes (PPO update every iteration) | no (forward pass only) | no (forward pass only) |
| Rendering? | no (headless) | no (headless) | yes (`--enable_cameras`) |
| Episode assignment | random commanded velocity, curriculum-driven push/terrain | one fixed (magnitude, terrain_level) per env for its one episode | one fixed condition per clip |
| Sim-time per unit of work | 0.48s/iteration (all envs in parallel) | ~6s per trial | ~4-6s per clip |
| Wall-clock, this run | ~90 min (1500 iterations) | ~1 min (1000 trials) | ~16.5 min (44 clips total) |
