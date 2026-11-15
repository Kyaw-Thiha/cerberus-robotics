# Newton spike — kit-less Isaac Lab on the Newton backend

Time-boxed feasibility spike (see `plans/handoff_newton_spike.md`): can the Cerberus Go2 locomotion
stack run on Isaac Lab's kit-less Newton backend, next to the PhysX / Isaac Sim path? The result is
`docs/reports/newton_spike.md`; nothing here is imported by the PhysX training or eval code.

## What runs where

| file | role |
|---|---|
| `tasks.py` | the Newton variant of the Cerberus rough Play task (`...-Cerberus-Newton-Play-v0`), built from the PhysX cfg by swapping physics + height scanner and applying the transfer flags |
| `parity_check.py` | stage 3: 64 envs, 500 steps, observation/action/joint-order parity against the contract file |
| `policy_loader.py` | loads the PhysX-trained policy from `.pt` (actor MLP rebuilt from the state dict) or `.onnx` |
| `transfer_eval.py` | stage 4: forward-command rollout, terrain levels 0–5, fall rate + velocity-tracking error |
| `scene_load.py` | stage 5: `--probe-only` parses a GRScenes USD with the kit-less OpenUSD; otherwise builds the home as terrain and steps 200 times |
| `gl_probe.py` | EGL/OpenGL/pyglet availability in the container (what `--video` through `newton_gl` needs) |
| `contract.py` | reads `common/cerberus_policies/contracts/heightscan_ppo_v1.yaml` |
| `newton_job.sh` | the pod-side job: deps → pinned Isaac Lab kit-less install → the five stages → R2 sync → self-stop |
| `submit_run.py` | laptop side: git bundle + checkpoint → R2, create the job pod, start the watcher and the guard; `--resume-pod/--job-id/--set` restart a stopped pod with new code or env |
| `pod_guard.py` | RunPod-side cost guard: the same decision function as `scripts/session_guard.py`, stopping through `runpod_provider.stop_pod` (that file's runpod branch uses the Vast key) |
| `collect_results.py` | pull a job's artifacts from R2 and print the numbers the report quotes |

## End-to-end

```bash
# 1. commit the branch first: the pod gets HEAD of `newton-spike`, not the worktree
git commit -m "..."

# 2. review what would be sent (no pod, no upload)
python3 spikes/newton/submit_run.py --dry-run

# 3. create the pod and start the laptop watcher + the detached 2.5 USD / 2 h guard
python3 spikes/newton/submit_run.py --start-watch
```

Progress: `rclone copyto cerberus_r2:<bucket>/jobs/<job_id>/status.json -` every 30 s; logs land
under `cerberus_r2:<bucket>/jobs/<job_id>/{outputs,logs}`. The pod stops itself; the owner deletes it.

## Environment the job builds

* image `nvidia/cuda:13.0.0-base-ubuntu24.04` (public, ~250 MB) — no SSH, job mode
  (`dockerEntrypoint ["/bin/bash","-c"]`, the bootstrap travels base64 in `CERBERUS_BOOTSTRAP_B64`);
* Isaac Lab pinned to tag `v3.0.0` (`e3f7d9ff`, the commit the Cerberus image's stack is built from),
  installed with `uv sync --frozen --inexact --extra video --extra importers` — Newton, rsl-rl and
  OpenUSD (`usd-exchange`) are base dependencies of that tag;
* the repository's own `.env` never reaches the pod: only the four R2 variables, the job identity and
  `RUNPOD_API_KEY` (needed for the self-stop) are passed.
