# Cerberus — RunPod Remote Execution Playbook

Practical, tactical reference for actually operating a RunPod pod session for
this project — accumulated from real sessions, not theoretical. See
`docs/isaac_lab_workflow.md` for the broader "why remote at all" / provider
strategy; this doc is the "how, concretely, day to day" companion to it.

## Pod lifecycle

Use the RunPod MCP tools (`create-pod`, `get-pod`, `list-pods`, `delete-pod`,
`stream-pod-logs`) rather than shelling out to `runpodctl` when they're
available — structured, no auth-string parsing.

- **Image**: `kevinkyawthiha/cerberus-remote:v3` (old stack; the new family is in "New image family" below) (built from
  `docker/remote/Dockerfile`: Isaac Sim + Isaac Lab, no scene dataset), GPU `NVIDIA GeForce RTX 4090`,
  `SECURE` cloud tier, 40GB container disk, port `22/tcp` exposed, SSH public
  key from the dedicated automation key (see "SSH access" below).
- **Cold start is unpredictable, not stuck**: pulling the ~18.7GB image (down
  from 20.3GB before OmniGibson was removed) takes anywhere from ~1 to ~5+ minutes depending on
  host/network. `get-pod`
  returning `runtime: null` for a while is normal. Check
  `stream-pod-logs(source="system", tail=5, maxWaitMs=3000)` for actual
  download progress before concluding anything is wrong — don't re-roll on a
  hunch. See `runpod_stuck_pod_diagnosis` precedent: this exact pattern was
  once mistaken for a stuck host and it wasn't.
- **`get-pod` gives the direct SSH mapping** once runtime is up
  (`runtime.ssh.direct` / the top-level `ssh.direct` field: host + port).
  Prefer this over the `ssh.io` proxy once available — fewer moving parts.
- **Availability varies** — `create-pod` can 400 with "no instances
  available" for a GPU type/region combo; retry without a restrictive
  `dataCenterIds` filter, or with a different explicit list, rather than
  giving up.
- **Terminating and recreating can land you back on the same physical host**
  if you don't constrain the datacenter differently — relevant if you're
  re-rolling specifically to escape a bad host (see driver section below).

## Pod hosts and the driver gate

Isaac Sim 6.x segfaults at startup (`libcarb.tasking`, exit 139) on some NVIDIA 595.x host drivers. The evidence:
595.71.05, 595.79 and 595.84 are reported bad; **595.58.03 is the validated 595 build**, and the **580 branch from
580.95.05** is validated by the Isaac Lab 3.0 docs. The host driver decides this, not the container image, so no image
tag can fix it. The allowlist lives in `scripts/pod_driver_allowlist.json` with this reasoning beside it; the gate
(`scripts/pod_driver_gate.sh`) runs first in every pod driver, records every observed (CUDA shown, driver) pair in
`outputs/launch_probe.csv`, and exits 2 on any other build. A pod that fails the gate holds no data, so the gate may
terminate it through the provider API immediately (the one approved exception to "never terminate a pod").

| date | driver | CUDA shown | GPU | datacenter | image | verdict |
|---|---|---|---|---|---|---|
| 2026-10-06 | 580.159.04 | unknown | RTX A5000 | unknown | v6-rel / v6-rel2 | OK: renders and steps |
| 2026-10-25 | 595.91.07 | unknown | RTX A5000 | unknown | v6-ga | render OK only with `CERBERUS_ALLOW_BAD_DRIVER_RENDER=1` |
| 2026-11-08 | 595.91.07 | unknown | RTX A5000 | unknown | v6-ga | Kit launch crashed roughly one run in two |
| 2026-11-09 | 595.91.07 | unknown | NVIDIA A40 | unknown | v6-ga | **SEGFAULT** at app start (exit 139) |
| 2026-11-10 | 595.91.07 | unknown | NVIDIA A40 | unknown | v6-ga | **SEGFAULT** at app start (exit 139) |
| 2026-11-10 | unknown (crash signature matches 595.x) | unknown | RTX 4090 | unknown | v6-ga | **SEGFAULT** at app start (exit 139) |
| 2026-11-10 | unknown | unknown | RTX 4090 | unknown | v6-ga | provisioning only: port 22 never published |

"unknown" means the value was not recorded in that session; the gate now records it automatically. Back-filled from
`docs/arena_integration_log.md`, `docs/reports/scene_eval_phase0.md` and the H4 session logs; no `env_report.json`
under `migration/` on R2 carried a pod driver version.

**Provisioning.** Runpod's CUDA filter is the only way to steer the host branch, and the mapping is an unverified
hypothesis until the gate's log confirms it. Through the API pass `"allowedCudaVersions": ["13.0"]` first (expected to
select the 580 branch); fall back to `["13.2"]` **only** with the gate enforcing exactly 595.58.03. In the console set
the **CUDA Versions** filter to the same value. `runpodctl pod create` (2.14.0) exposes only `--min-cuda-version`, a
floor rather than an allowlist, so it cannot express this choice; it also has no `--terminate-after`, hence the gate's
self-termination and the session clock.

## Building and pushing the image

`docker/` is organized as one nested directory per image, not a flat list of
`Dockerfile.<name>` files (restructured 2026-09-21 once SLAM backend images
joined `remote` as a second image family — see below):

```
docker/
├── remote/            # Isaac Lab (Dockerfile, requirements.txt, start.sh)
└── slam/               # one SLAM backend per subdirectory
    ├── shared/start_slam_backend.sh   # common entrypoint, one copy
    ├── fast_lio2/Dockerfile
    ├── bievr_lio/Dockerfile
    ├── fast_livo2/Dockerfile
    ├── orb_slam3/Dockerfile
    ├── multicam_slam/Dockerfile
    └── gs_livo/Dockerfile
```

No CI — build and push manually when a Dockerfile or its associated files
change. The `remote` image:

```bash
docker build -t kevinkyawthiha/cerberus-remote:v3 -f docker/remote/Dockerfile .
docker push kevinkyawthiha/cerberus-remote:v3
```

A SLAM backend image (run `python scripts/fetch_slam_backends.py <name>`
first — each backend's Dockerfile `COPY`s locally-vendored source, it does
not `git clone` inside the Dockerfile the way `remote` does):

```bash
docker build -t kevinkyawthiha/cerberus-slam-fast-lio2:v1 -f docker/slam/fast_lio2/Dockerfile .
docker push kevinkyawthiha/cerberus-slam-fast-lio2:v1
```

(substitute the backend name in both the tag and `-f` path for any of the
other five — `bievr_lio`, `fast_livo2`, `orb_slam3`, `multicam_slam`,
`gs_livo`).

Bumping the tag vs. overwriting `v1` is free either way (Docker Hub doesn't
charge for public-repo storage, and layers dedupe by content hash) — bump for
rollback safety, or overwrite for simplicity; update the tag referenced in
pod-create calls either way.

**Rebuild needed for:** Dockerfile changes, `requirements.txt` changes,
`start.sh` changes. **Not a rebuild
trigger:** any Cerberus code change (`policy/`, `scripts/`, etc.) — that's
synced per pod via `tar`/`git`, separate from the image.


## New image family: `docker/remote_v6` (2026-10-05)

Sim 6.1.0 base at first (`nvcr.io/nvidia/isaac-sim:6.1.0`; now 6.2.0, both pullable anonymously), Python 3.12, plus a pixi `sim-ros2` stage (re-added 2026-10-06: `rclpy` for the sim-to-ROS2 relay and `evo` for scoring).
Old `cerberus-remote:v3` (Sim 5.1.0 / Lab 2.3.2 / Py 3.11) is kept as the stack all locomotion results were produced on; it is no longer needed to build the fused USD (the URDF workflow builds it on Lab 2.3.2 or 3.0 EA, see `docs/embodiment_asset_composition.md`).

- `kevinkyawthiha/cerberus-remote-v6:v6-dev`, digest `sha256:81d8ff200a5952e417743cf9ad4da2b4a16c5ad47af26759546e3b87e9a13929`,
  35.4 GB. Isaac Lab commit `bb0c8e1b9` (Arena main's pin, pkg 16.4.0), Arena main `311d0f0fc`, rsl-rl-lib 5.4.1, torch 2.11.0+cu128,
  lightwheel-sdk 1.0.3, numpy 2.3.5.
- `kevinkyawthiha/cerberus-remote-v6:v6-ea-dev` (superseded), digest
  `sha256:4377062efde6d1b83798374db8efba5b8233ddbc1f4348442f4c656abefb99c3`, 35.0 GB. Isaac Lab `v3.0.0-EA`
  (`ae37b028e`, pkg 17.0.2).
- **`kevinkyawthiha/cerberus-remote-v6:v6-ga` (current, 2026-10-25)**, 34.4 GB, digest
  `sha256:da1dc4cf73cb1955f062921d35147bf8fc7519d8c7bca26dbde790a16367b19c`. Sim 6.2.0, Isaac Lab `v3.0.0` (`e3f7d9ff`, pkg 28.0.0), Arena `main` `276d9731`,
  rsl-rl-lib 5.5.1, torch 2.11.0+cu130, Python 3.12.14. Pod-verified, including scene render and video capture: see `docs/arena_integration_log.md`
  "Lab 3.0 GA upgrade". All pods drawn on 2026-10-25 had driver 595.91.07 (see the driver section there); set `CERBERUS_ALLOW_BAD_DRIVER_RENDER=1` to
  render on it and wrap every launch in `timeout`. Build: `TAG=v6-ga PUSH=1 docker/remote_v6/build.sh`.
  Off-pod-unverifiable changes are checked by `scripts/run_ga_smoke_pod.sh` (train smoke -> play `--video` on the
  checkpoint it just wrote -> Arena spike), which judges each step from its artifact and writes
  `data/reports/ga_smoke_<stamp>/`; `--upload` copies it to `r2:migration/ga_smoke_<stamp>/`.
- `kevinkyawthiha/cerberus-remote-v6:v6-rel2` (previous, 2026-10-11, rollback), 35.1 GB, digest
  `sha256:5ab14c76a0230f4ef533991ff5be3b7682c3d4195e91485eaabc8a4a72a81b1b`. Same Isaac Lab / Arena / rsl-rl / torch
  as `v6-rel` (`/etc/cerberus_versions.json` is byte-identical), plus a pinned, sha256-verified **rclone 1.75.1** and
  `lightwheel-sdk==1.0.3`: no per-pod rclone workaround any more. Verified after the push: `rclone version` v1.75.1,
  pixi `sim-ros2` imports (`rclpy`, `zmq`, `msgpack`) and `evo_ape` work. `v6-rel` (below) is kept as the rollback.
- `kevinkyawthiha/cerberus-remote-v6:v6-rel` (previous), 35.1 GB, digest
  `sha256:c3443f44003dff5389e29d3d386c292e7a7330db602902207de6ab1960833e5c`. Isaac Lab `release/3.0.0`
  (`dcabdcda`, pkg 27.0.1) + Arena `main` (`34bb35a85`), rsl-rl-lib 5.5.1, **torch 2.12.0+cu130**.
  Two operational consequences found on 2026-10-06: (1) the CUDA 13 build needs a host driver >= 580, so create
  these pods with `--min-cuda-version 13.0` (driver 580.159.04 on an RTX A5000 renders and steps fine); (2) the
  pixi `sim-ros2` stage is back in the image (rclpy, pyzmq, msgpack, `evo` 1.37.1), which is what the SLAM eval's
  relay and scoring need — `build.sh` now also copies `pixi.toml`/`pixi.lock` into the build context.
  Third consequence found on 2026-10-07: **this image's `rclone` is v1.58.1, too old for the R2 remote** — it
  rejects the `Cloudflare` S3 provider with `s3 provider "Cloudflare" not known`, which reads like a credentials
  problem, and it breaks *every* R2 write (scene fetch/mirror, and `scripts/sync_to_r2.py`, whose failure
  `run_slam_eval_remote.sh` treats as fatal and refuses to self-terminate). The Dockerfile now installs a pinned
  rclone 1.75.1 (built into `v6-rel2`), and `scripts/sync_to_r2.py` preflights the version (fail-fast with an explicit
  message). Only a pod created from `v6-rel` still needs `curl -fsSL https://rclone.org/install.sh | bash` once
  before anything that writes to R2 (~20 s). Gotcha found building `v6-rel2`: a Dockerfile `ARG` is an env var inside
  `RUN`, and rclone parses every `RCLONE_*` env var as a flag, so the args are named `CERB_RCLONE_*`.

Build and push with the script (env vars `TAG`, `ISAACLAB_REF`, `ARENA_REF`, `PUSH=1`), not a bare `docker build`:

```bash
TAG=v6-dev ISAACLAB_REF=bb0c8e1b9af381bf13064ec3303e17db79e4b6ef ARENA_REF=311d0f0fc6724eb953313f04765f76d947982c40 PUSH=1 docker/remote_v6/build.sh
TAG=v6-ea-dev ISAACLAB_REF=ae37b028ea415c91ea2bc32609efcd759ed2b974 PUSH=1 docker/remote_v6/build.sh
```

`build.sh` builds from a minimal temp context (the repo root has no `.dockerignore`); the exact args used per build are in `data/reports/overnight/docker_build_v6_*.log` (the lines above are reconstructed from NOTES.md, not copied from the logs). `build.sh` comments mention `docker login nvcr.io`, but the base image pulled anonymously.
Dockerfile fixes found: `ARG ISAACSIM_VERSION` must be redeclared after `FROM`; version-JSON distribution names need normalization.
At 35 GB the image needs container disk >= 80 GB (the old 40 GB default is too small).

## Pod-run findings (overnight run, 2026-10-05)

- **CPU pods cap container disk at 20-30 GB and ignore the volume**, so large extractions cannot run there. The GRScenes zip extraction
  ran on an A5000 GPU pod (160 GB disk, $0.27/h, about 3 h). 7z reads the split zip (`.z01`...) directly (170,029 members).
- **Driver vs renderer**: driver 595.91.07 + Sim 5.1 RTX rendering segfaults (`librtx.scenedb.plugin.so` at Hydra engine creation; `play --video`
  blocked by the guard); physics without cameras is fine. On Sim 6.1, drivers 570.195.03 and 580 passed the render smoke (Isaac RTX camera RGB).
  595 on Sim 6.1 was not tested.
- **Env for SDK/Isaac `isaaclab.sh`** over non-tty ssh: set `ISAACLAB_PATH` and `TERM` explicitly; do not rely on the login environment.
- **Never `cat /etc/cerberus_env.sh`.** `start.sh` writes all `RUNPOD_*` env there, including the Runpod API key. That happened once overnight
  and the key appeared in an agent transcript: it must be rotated (security note, no value recorded). Recommended fix, not yet applied: filter
  `RUNPOD_API_KEY` out of that file in `start.sh` (only `ISAACLAB_PATH` and non-secret vars are needed).
- **Lab 3.0 `--headless`**: `AppLauncher` no longer registers `--headless` / `--enable_cameras` (use `--viz`); Lab's `annotate_demos.py` reads
  `args_cli.headless` and breaks, so we use our in-process annotate port.
- **`play.py` without `--video` has no end condition**; it only stops via `timeout` (expected, not a failure).
- **Cost and cleanup discipline**: every session used a hard wall-clock cap, explicit `delete-pod`, then `list-pods` to verify deletion. Each GPU
  session cost under about $1.1. Pods named `dqn-*` belong to another project and were never touched. Balance fell 35.30 -> 16.00 overnight but
  other-project pods were running, so that delta is not attributable to Cerberus. One pod agent overran its cap; the orchestrator killed it and
  deleted the pod explicitly.

## Pod-ops lessons from the 2026-10-06 SLAM-readiness session

- **`pkill -f` / `pgrep -f` on a pattern that also appears in your own `ssh ... "<command>"` kills the ssh session** (the bash
  running your command matches it). Kill by numeric PID, or use the `[v]alidate_scenes` bracket trick.
- **Sync in small batches** (`tar czf - <paths> | ssh ... "tar xzf - -C /workspace/cerberus"`) with `-o ServerAliveInterval=10
  -o ConnectTimeout=20`; one big sync to a loaded shared host reset the connection and cost 28 minutes. Resync only the files
  that changed. Remember the sync excludes `policy/locomotion/checkpoints`: copy the one ONNX/`.pt` you need explicitly.
- **`runpodctl pod create --wait` can block 10+ min on the 35 GB image pull.** Create without `--wait` and poll ssh in a bounded
  loop (`for i in $(seq 1 40); do ssh ... echo up && break; sleep 20; done`).
- **Pull logs and artifacts BEFORE deleting a pod.** The crash dumps and child logs that would have diagnosed the admission
  failures were lost with the first pod. For admission runs pull `data/reports/scene_admission/`; for SLAM runs pull
  `slam/slam/eval/results/<run>/` (the bag is GBs: take `smoke_check.json`, `sim_loop_report.json`, `videos/` unless you need it).
- **Launch Isaac with cameras enabled** (`args.enable_cameras = True` on the Lab 3.0 launcher) on the v6 image family (`v6-rel2`/`v6-rel`): every launch without it
  died within ~1 s at startup in these sessions. Do not import `pxr` before the app launches (it breaks `omni.usd`); use a child
  interpreter (`scene_geometry.main`) for anything that needs `pxr` early.
- **Hugging Face rate-limits range reads** (HTTP 429). `scripts/remote_zip.py` paces them (`REMOTE_ZIP_RPS`, default 6/s).
- **Sim speed varies by host**: the same 21 s mission took ~15 min on one A5000 host and ~14 min on a loaded one (load average 58);
  give smoke runs a 35-45 min cap, not 10.

## SSH access

Use the dedicated, passphrase-free automation key
(`~/.ssh/cerberus_runpod_ed25519`), never the personal key — the personal key
is passphrase-protected and silently fails for non-interactive SSH. Pass its
public key as `sshPublicKey` at pod-create time.

## GPU driver compatibility — check before any rendering work

Isaac Sim 5.1.0's RTX renderer is driver-version-sensitive, and this is
**host-level infrastructure outside the container's control** — whichever
physical machine RunPod allocates brings its own driver.

- **595.x (R590 branch) is confirmed-bad** — reproducibly crashes the RTX
  renderer (segfault at Hydra engine creation). `policy/cli_common.py`'s
  `check_gpu_driver_for_rendering()` blocks this branch specifically before
  paying Isaac Sim's multi-minute boot.
- **580.65.06 is the officially validated version.**
- **Everything else (570.x seen twice in one session, other untested
  branches) is a genuine unknown** — not pre-emptively blocked by the driver
  check, and re-rolling isn't guaranteed to escape it. Don't guess from the
  version number either way. Instead, for any pod session where rendering
  happens at some point (video capture, `--video`), run a **cheap render
  smoke test first**: a minimal standalone script — `AppLauncher` with
  `enable_cameras=True`, a tiny 1-2 env task, no trained checkpoint needed,
  a handful of steps checking `env.unwrapped.render()` returns non-None. Takes
  under 3 minutes (mostly Isaac Sim's own boot) and gives a definitive answer
  before committing pod time to an expensive run that only renders at the
  very end — much cheaper than discovering a crash after paying for the
  training part first.

## Operational gotchas found on the W1.5a pod runs

- `$ISAACLAB_PATH/isaaclab.sh` fails instantly (`'ansi+tabs': unknown terminal type`) when run from a
  non-tty `ssh host "cmd"`. `export TERM=xterm` first.
- `SimulationApp.close()` terminates the process with exit code 0 (so a `finally: close()` hides tracebacks and
  failures) and has hung for 10+ minutes at shutdown on a pod. In one-shot scripts, write results to disk, flush
  and `os._exit(code)` instead (see `scripts/spawn_hold_go2_arx5.py`).
- `runpodctl pod create --env '{"PUBLIC_KEY":"<pub key>"}'` works with this image's `start.sh`; `--wait` returned
  after about 3 minutes including the image pull. `runpodctl` has no `--terminate-after`: delete the pod explicitly.
- Do not `pkill -f python3` over ssh: the pattern matches the ssh command itself and kills the session.
- Driver 595.91.07 (the branch marked bad above for RTX rendering) ran headless physics with no cameras fine;
  rendering on it was not tested.

## Syncing code to a pod

The image does **not** have `rsync` installed — use `tar` over SSH instead,
which needs zero remote setup:

```bash
tar czf - --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='policy/locomotion/checkpoints' --exclude='policy/locomotion/runs' \
    policy docs scripts .env | \
  ssh -i ~/.ssh/cerberus_runpod_ed25519 -p <port> root@<host> \
    "mkdir -p /workspace/cerberus && tar xzf - -C /workspace/cerberus"
```

`.env` is a dotfile — verify it actually synced with `ls -la`, not plain
`ls`, which silently hides it and can look like a sync failure when it
worked fine.

## Launching long-running work: always detached, always killable

Every remote invocation — training, eval, video capture, a whole multi-step
pipeline script — should be:

1. **Launched via `nohup ... & disown`** over a single SSH command. This is
   the entire mechanism that lets work survive the local terminal closing,
   the SSH connection dropping, or the calling Claude Code session ending
   entirely — the work has to live purely in the pod's own process tree, not
   depend on anything local staying connected.
2. **Wrapped in `timeout --signal=KILL --kill-after=<grace> <duration>`,
   never bare `timeout`.** A genuinely stuck Isaac Sim process (degenerate
   physics state, or a Python-level crash that doesn't cleanly tear down the
   Kit app) can ignore plain SIGTERM outright and spin indefinitely — this is
   confirmed, not theoretical, and has happened more than once. The SIGKILL
   wrapper is what self-recovers a hung step automatically; don't rely on
   active monitoring to catch it.
3. Note GNU `timeout`'s duration syntax wants a single unit suffix
   (`150m`), not combined units (`2h30m` is rejected as "invalid time
   interval").

For a multi-step sequence (train → video → eval → video → sync, say), write
**one shell script** that chains every step with its own R2 sync after each,
upload it once, launch it detached, then verify it's alive — don't try to
babysit each step with live tool calls if the human is about to close their
laptop or step away.

## Verifying before you say "safe to walk away"

Never tell a user it's safe to close their terminal or sleep their laptop
just because a launch command returned successfully — a launch that crashed
10 seconds in looks identical to "started fine" from the outside. Actually
check, right after launching:

- `ps aux | grep ...` — confirm the process tree exists **and** shows no
  controlling TTY (`?` in the TTY column) — that's what confirms it's
  genuinely detached, not just backgrounded in a session that's about to die.
- `nvidia-smi` — confirm real GPU utilization, not 0%.
- The actual log content — confirm real progress (an iteration count ticking
  up, a checkpoint being written), not just a process existing with no output.

## Monitoring a long remote job

A bare `ssh ... "tail -F remote.log" | grep ...` Monitor **can silently stop
delivering live updates for hours while the underlying job keeps working
correctly underneath** — this caused a real ~10-hour idle-pod cost incident
in this project's history (the training had actually finished on time; the
notification stream just went quiet). Always wrap remote log-tailing in a
reconnect loop:

```bash
while true; do
  ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -p <port> root@<host> "tail -n 0 -F remote.log" 2>&1 | your-filter
  echo "[monitor] disconnected, reconnecting in 5s..."
  sleep 5
done
```

Treat "disconnected, reconnecting" messages as normal background noise, not
alarming — as long as you independently re-verify the underlying job is
still progressing (direct log read, process check) rather than trusting the
notification stream alone as the sole source of truth.

**Default policy: pods terminate after their work + final sync succeeds**,
unless the very next step is queued immediately in the same session. If a
pod needs to stay up between steps (e.g. review results before deciding
whether to run eval), that's a deliberate `--no-terminate`-style choice, not
the default — and it should still get terminated explicitly once the human
confirms they're done reviewing, not left running "just in case."

## A `torch.inference_mode()` gotcha specific to multi-checkpoint inference scripts

If you write a script that loops over multiple checkpoints/conditions against
one persistent env, calling `env.reset()` between each: wrap the **entire**
loop (every reset, every step, across every checkpoint) in ONE persistent
`torch.inference_mode():` block from the start, not re-entered per
clip/iteration. A tensor touched inside `inference_mode()` stays permanently
marked as an "inference tensor" even after that block exits, and a later
in-place write to it from *outside* inference_mode (the next iteration's
`env.reset()`, if it's outside the block) throws `RuntimeError: Inplace
update to inference tensor outside InferenceMode is not allowed`. This only
bites scripts with multiple manual resets across separate inference_mode
scopes in one process — existing single-continuous-rollout scripts in this
repo never hit it because they never manually reset mid-script.

## R2 artifact sync

See `docs/artifact_storage.md` for the full bucket layout and rationale.
Practical note relevant to remote sessions specifically: `scripts/
run_remote_job.sh` handles periodic + final sync automatically for a
training invocation (survives a mid-run force-kill from RunPod's own balance
enforcement); anything outside that wrapper (eval sweeps, video capture)
should call `scripts/sync_to_r2.py sync-run` explicitly after each step in
your pipeline script, not just once at the very end.

## Scene assets are pod-only

Scene assets (GRScenes homes; tens of GB of shared `Materials`/`models`) are downloaded, preprocessed and read on the pod only; nothing scene-sized is ever fetched to the laptop, and the scene-touching scripts (`scripts/fetch_grscenes_*.py`, `scripts/validate_scenes.py`, `scripts/build_lidar_proxy.py`, `scripts/build_scan_proxy.py`, `scripts/build_occupancy.py`) refuse to run without `CERBERUS_ON_POD=1`, which the `*_pod.sh` runners set.

## Run artifacts: every eval records clips

Every pod evaluation in Phase 1 (scene stand, scripted smoke, navigation eval) records video and syncs it, not
just numbers. A metric table cannot show whether a "success" walked through a doorway or through a wall, and the
failure clips are the ones that explain a bad rate.

Rules:

- Record clips with `common/cerberus_eval/video_clips.ClipRecorder` (headless viewport capture; `--no-video`
  switches it off for pure-metrics reruns). Default 3 clips per run at half resolution.
- Keep the **first success** and the **first failure** when both occur, then fill up to the count
  (`video_clips.plan_clips`). Failure clips are mandatory when a run has any failure.
- Name clips `<scene>_<seed>_<outcome>.mp4` and write the matching `<scene>_<seed>_<outcome>.json` beside each
  clip, so a clip cannot be separated from its numbers.
- Sync after each step with `scripts/sync_to_r2.py sync-run ...` and print the manifest (see above). Do not wait
  for the end of the session: a balance cut-off has already cost this project a run's artifacts.
- The report quotes the clip filenames; a number without a clip is a number nobody can check.
