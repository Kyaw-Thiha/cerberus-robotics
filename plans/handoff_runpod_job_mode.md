# Handoff: Physics Session on RunPod in Job Mode (PhysX path)

Save as `plans/handoff_runpod_job_mode.md`. Goal: get the Phase 1 physics results (scene-1 stand test,
scripted smoke, navigation eval) from one bounded RunPod session, with no SSH, using the job-mode tooling
already built (job entrypoint, heartbeat, watcher, laptop guard, resilient launches). This is the PhysX /
Isaac Sim path.

## Context (settled by earlier sessions)

- Kit crashes at startup on RunPod hosts with driver 595.91.07 (roughly 1 in 4 launches start; hosts can
  degrade during a session). Both A40 and RTX 4090 hosts behaved the same, so the cause is the driver
  build, not the GPU model.
- Vast is parked: the doc-exact args hook works, but the 35 GB image pull stalls and Vast stops the
  instance mid-pull. Keep the Vast code, mark it "image too large, needs a slimmer image" in
  `docs/runpod.md`.
- The R2 credentials were rotated by the owner and the new token is scoped to the `cerberus-robotics` bucket.
- Another agent works in a **separate git worktree** on a Newton spike under `spikes/newton/`. Do not touch
  that directory. Stay in the main working tree and keep your changes to `scripts/`, `docs/runpod.md`,
  `plans/`, and navigation files if the physics run requires them.

## Rules

- Never print or read `.env` or env contents; use the redaction helper everywhere.
- Spend caps: 3 USD per session counted across all live pods, 90 minutes of job wall clock (starts at the
  first heartbeat), 3 hours absolute ceiling, at most 3 pod creations, one live pod at a time. Detached
  laptop guard and heartbeat watcher stay.
- Deliver code to the pod through the R2 bundle. Do not wait for a GitHub push (SSH auth on this machine is
  broken). Report the commit range that still needs pushing.
- Tests and lint green before and after. Commit messages describe what changed; no handoff or package labels.

## Steps

1. **RunPod job mode.** Create pods through the RunPod REST API with the image, no SSH port, and the start
   command overridden to run a short bootstrap that fetches `job_entrypoint.sh` from the bundle. Verify the
   REST field names for overriding the entrypoint/start command in the RunPod docs before coding; the long
   bootstrap travels in a base64 env variable. Pass only the allowed env: restricted R2 variables,
   `CERBERUS_CODE_BUNDLE`, `CERBERUS_JOB_ID`, `CERBERUS_DEADLINE_MINUTES`, `CERBERUS_IMAGE_TAG`. Try
   `allowedCudaVersions ["13.0"]` once (it fails fast when there is no stock), then create filter-free;
   secure cloud; GPU list A40, RTX 4090, A5000, L40S, A6000.
2. **Self-stop.** Inside the job, stop the pod with `runpodctl stop pod $RUNPOD_POD_ID` (the pattern in
   RunPod's docs), 3 retries; if the binary is missing, install it or use the REST stop with the exact
   headers and User-Agent that worked from the laptop. Run a live self-test at the start of the job (a
   harmless authenticated call on the same path) and report `self_stop_ok` in the heartbeat. If self-stop
   fails, write `STOP_FAILED` so the laptop guard acts. **Test the self-stop path live before any physics
   runs.**
3. **Laptop side.** Reuse `watch_job.py` and the detached guard. Startup grace 25 minutes until the container
   is running (the pull took about 12 minutes on RunPod before), then 4 minutes to the first heartbeat,
   stale after 5 minutes. Enforce the USD cap from the laptop in code (elapsed times hourly price), not as a
   printed estimate.
4. **Launch experiment.** After the driver gate, run 5 consecutive Kit launches through the resilient
   wrapper. 4 or more start: continue. Fewer than 4: record launches ok/total, crash signatures and hygiene
   logs, sync, stop the pod, and create the next host (max 3 creations, then stop and report).
5. **Physics plan** (scene 1 only), all through `run_resilient.sh`: stand test at 1 env, scripted smoke,
   navigation eval with metrics only (N=10 start-goal pairs), then replay one success and one failure for
   clips only if at least 20 minutes remain before the deadline. No policy or planner tuning. If the stand
   test fails its assertions (joint order, 235-D observation, NaN, sinking), record the diagnosis and
   `build_timings.json` and stop.
6. **Wrap-up.** Sync everything, verify the R2 manifest lists every output (assertions, timings, metrics,
   clips, replay.json, launch logs, branch bundle), and report: launches ok/total per host with
   driver/GPU/price, restarts needed, spend, GPU minutes, the kinematic-vs-physics gap if physics ran, and
   the push range. The job stops the pod; the owner deletes it.

## Out of scope

Vast runs, a second provider, scenes other than scene 1, scene-2 diagnosis, policy or planner tuning,
anything under `spikes/newton/`, and any change to the Newton spike's files.
