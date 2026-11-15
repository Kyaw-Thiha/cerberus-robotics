# Handoff: Newton Feasibility Spike (kit-less Isaac Lab, no Isaac Sim)

This is a time-boxed feasibility spike, not a migration. It answers one question for the project
owner: **can Cerberus run its locomotion and navigation loop on Isaac Lab's kit-less Newton backend,
as a lighter, more portable physics path next to the PhysX / Isaac Sim path?** The result is a
go / no-go report with an effort estimate.

## Why

Kit (Isaac Sim) crashes at startup on many rented hosts (driver 595.x), the container image is about
35 GB, and recording is tied to RTX. Isaac Lab 3.0 supports a kit-less install with the Newton
physics backend (co-developed by Disney Research, Google DeepMind and NVIDIA, governed by the Linux
Foundation). If it works for our Go2 stack, other researchers can run Cerberus without Isaac Sim, on
more GPUs, with a small image. The PhysX path stays as the high-fidelity path.

## Working setup

- Work in your **own git worktree and branch** (`newton-spike`), not in the main working tree. Other
  agents are changing `scripts/` and `docs/runpod.md` on `main`. Commit locally; do not push (SSH
  auth is broken); deliver code to pods through a git bundle on R2; report the commit range.
- **File ownership:** you own `spikes/newton/` and `docs/reports/newton_spike.md`. **Do not modify
  existing PhysX tasks, `cerberus_compat`, existing eval scripts, or any file under `scripts/`.**
  Add new files only (for example new task IDs such as `...-Cerberus-Newton-v0` that build on the
  existing config without changing it), and import existing helpers read-only. If you need a change
  to an existing file, describe it in the report instead.
- Never print or read `.env` or env contents; use the repo's redaction helper. Pass only the
  restricted R2 variables to pods.
- Commit messages describe what changed; no handoff labels.

## Read first

Repo: `docs/reports/phase0_ga_parity.md`, `docs/h2_report.md`, `docs/h4_report.md`,
`policy/locomotion/core/platforms/go2/env_cfg.py`, `common/cerberus_compat/`,
`common/cerberus_policies/contracts/heightscan_ppo_v1.yaml`, `scripts/session_guard.py` and
`scripts/create_pod.py` (reuse read-only), `docs/runpod.md`. Isaac Lab 3.0 docs (search the docs site
if a URL has moved): the kit-less installation page, the Newton "Supported Features" page,
"Transfer Policies Between PhysX and Newton", "Recording video clips during training", the Ray
Caster page (on Newton the ray-caster ignores `mesh_prim_paths`; there is a Newton raycast sensor
config), and the Known Issues page (OpenUSD before 26.05 can crash parsing rigid bodies with many
colliders in kit-less Newton).

## Budget and pod rules

- Total pod budget: **2.5 USD and 2 hours**, enforced by the laptop-side guard (reuse
  `session_guard.py` through its CLI or import; do not edit it). One pod at a time, at most 2
  creations. Stop the pod at the end (the owner deletes it). Sync outputs to R2 before stopping.
- No Kit means the 595.x Kit crash does not apply. Driver rule for this spike: accept any driver at
  or above **580.65.06** (any branch, including 595.x); record the driver build of every host.
- Image: use a small public CUDA 13 base image (verify the tag exists and is public) and install
  Isaac Lab kit-less at boot with `uv` (`./isaaclab.sh -i newton,tasks,assets,rl[rsl_rl]` or the
  documented equivalent). Record image pull time and install time. Do not use the 35 GB Cerberus
  image. No SSH: use the job pattern (bootstrap env var, heartbeat to R2, self-stop with
  `runpodctl stop pod $RUNPOD_POD_ID`) by copying the pattern into `spikes/newton/` without editing
  the existing entrypoint.

## Questions to answer (stages, in order; stop a stage's work when its kill criterion hits)

1. **Boot and stock task.** Kit-less install works; the stock Go2 rough task runs with
   `physics=newton_mjwarp` for a short smoke training (about 50 iterations) with no NaN; 5
   consecutive process launches succeed (record starts ok/5). Kill criterion: install fails after 25
   minutes.
2. **Video on a headless pod.** `--video` records an mp4 through the Newton GL headless viewer.
   Record whether a graphics context is available in the container (EGL/OpenGL), and what was needed.
   Report how the clip looks.
3. **Cerberus Go2 task on Newton.** Add a Newton variant of the Cerberus rough Play task: same
   observation layout (235-D: 48 proprio/command terms + 187 height-scan cells), same action term,
   decimation and command ranges, terrain generator, and the height scan implemented with the Newton
   raycast sensor. Run it headless for 500 steps with 64 envs. Compare observation dimension and term
   order against the contract file; the joint order must match the contract by name.
4. **Policy transfer.** Evaluate the PhysX-trained policy (`go2_locomotion_ga_v1_candidate`, ONNX or
   checkpoint; check which one loads on this stack) on Newton, using the documented transfer flags
   for joint and body ordering. A policy that falls within a few dozen steps in every environment
   signals an ordering mismatch; check that first. Then run a forward-command rollout on rough
   terrain at levels 0 to 5 (64 envs, 1000 steps) and report fall rate and velocity-tracking error,
   next to the PhysX numbers from the existing push-recovery sweep CSV
   (`runs/2026-09-08_19-28-32_analysis/eval/eval_rough_full_sweep.csv` and the GA sweep). Do not tune
   anything. If transfer is poor, run a 30-minute smoke retrain on Newton (stock recipe) only if
   budget remains, and report the numbers either way.
5. **Scene load (time-boxed to 45 minutes).** On the pod only (scene assets are pod-only; set
   `CERBERUS_ON_POD=1`), fetch one preprocessed GRScenes home with the existing fetch script
   (read-only use), and try to load its static geometry in kit-less Newton with the robot standing at
   a valid spawn pose; run 200 steps. Record the OpenUSD version in use, whether the Known Issues
   crash appears, and what workaround the docs suggest. A failure here is a valid result; describe
   exactly where it fails.

## Deliverable: `docs/reports/newton_spike.md`

- Per stage: what ran, commands, numbers, logs paths on R2, and what failed and why.
- Host rows: driver build, GPU, price, image pull time, install time, launches ok/5.
- A **decision table**: Go / Conditional / No-go for adopting Newton as a second physics backend,
  with the evidence for each of: install and boot reliability, video recording, Cerberus task parity,
  policy transfer, scene loading.
- An **effort estimate** for adoption: what would need retraining on Newton, which tasks and sensors
  need Newton variants, what the evaluation protocol should be (PhysX vs Newton columns), and what
  the repository structure should look like (a physics backend selection that follows the same
  plug-in pattern as `cerberus_worlds` and `cerberus_policies`). Estimates only; do not build it.
- Limitations and anything you could not verify.

## Out of scope

Changing any PhysX task or existing script, retraining the main policies, Phase 0.5 on Newton,
perception/SLAM/camera rendering beyond the video check, a SkyPilot adapter, provider changes, and
anything on Vast.
