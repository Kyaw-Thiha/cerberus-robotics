# Kit launch crashes on this host class (report draft -- NOT filed)

Ready to file on `isaac-sim/IsaacSim`. Not filed: this repository's session rules say the report is prepared, not
posted. Evidence collected 2026-11-22 and 2026-11-23 on Hyperstack L40 VMs.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA L40, 46,068 MiB, PCI 0000:00:06, architecture 8.9 |
| driver | 595.58.03 (runfile, DKMS; also tested 580.196.07, 580.95.05 -- the camera runs only survived 595.58.03) |
| OS | Ubuntu 24.04.3 LTS, kernel 6.14.0-37-generic, 56 GiB RAM, 28 vCPU |
| install | NVIDIA container `nvcr.io`-style image built from `docker/remote_v6` (Isaac Sim 6.2, Isaac Lab 3.0 GA), run as `docker run --gpus all` with the host driver mounted |
| provider | Hyperstack, CANADA-1, VM ids 1176254, 1176656 (2026-11-22) and 1177066 (2026-11-23) |
| API | `isaaclab.app.AppLauncher`, headless, `enable_cameras` on for camera runs |

## Symptom

Most launches die before our code runs:

- **14-15 s**, `exit -9` (killed) with the crash reporter logging
  `A crash has occurred. If a debugger should be attached, ...` and writing
  `/isaac-sim/kit/data/Kit/IsaacLab/3.0/<uuid>.dmp.zip`;
- a minority reach the app and then fail at 36-80 s with `exit 1` and a Python traceback from our own code (that
  part is ours, not NVIDIA's);
- successful starts are rare: **4 STARTED in 163 attempts (2.5 %)** on 2026-11-23; 6/63 on 2026-11-22.

**Masked exit status.** Several runs that produced no artifacts nonetheless exited **0** at the first Python block
after a long sim block (a 200-step stand report, a pose read, a metrics write). The dump file and the crash-reporter
line are present for some of them, so at least part of the "clean exit" population is a crash whose status Isaac
Lab's `AppLauncher` reports as 0. Upstream Isaac Lab PR #6636 addresses exactly this (truthful exit codes); it was
not applied to this image during the session (it is an image-only change, never committed here).

## Minimal reproducer

`scripts/host/kit_min_probe.py`: `AppLauncher(headless, enable_cameras)`, five `app.update()` calls, print
`PROBE STARTED`, loop `app.update()` for N seconds, print `PROBE DONE`, close. No scene, no Isaac Lab env, no asset.

```bash
docker run --rm --gpus all --entrypoint bash <image> -lc \
  '/workspace/isaaclab/isaaclab.sh -p scripts/host/kit_min_probe.py --seconds 5'
```

## A/B matrix (partial: the run that would measure it was cut to protect a 90-minute training job)

| variant | launches | starts | completes | median s to crash |
|---|---|---|---|---|
| V0 baseline, 595.58.03 | 8 attempted | 0 (probe bug: `--device` default `None` made the app fail at 3 s, fixed mid-run) | 0 | ~3 s (bug), not comparable |
| V1 `taskset -c 0-7` | not measured | -- | -- | -- |
| V2 `PXR_WORK_THREAD_LIMIT=1 OMP_NUM_THREADS=1` | not measured | -- | -- | -- |
| V3 official 6.2.0 container | not attempted (no `NGC_API_KEY` in the environment) | -- | -- | -- |
| V4 cameras off | not measured | -- | -- | -- |

The tooling is in place (`scripts/host/kit_ab_probe.sh <variant> <launches>`); the numbers above are what the session
could honestly measure before choosing the training job over the matrix.

## Representative evidence

```
[carb.crashreporter-breakpad.plugin] [crash] 'graphicsApi' = 'vulkan'
[carb.crashreporter-breakpad.plugin] [crash] 'buildVersion' = '110.4.0'
[carb.crashreporter-breakpad.plugin] [crash] 'experience' = 'Isaac Sim'
[carb.crashreporter-breakpad.plugin] [crash] A crash has occurred.
NVRM: Xid (PCI:0000:00:06): 43, pid=61050, name=python3, channel 0x00000008   (dmesg, once)
```

After that Xid 43 the container could not see the GPU at all (`torch.cuda.init()` ->
`RuntimeError: No CUDA GPUs are available`) until the VM was rebooted, while `nvidia-smi` on the host worked: the
failure poisons the context, it is not a driver-invisible GPU.

## Mitigations that helped

1. **relaunch until a start marker** (`scripts/host/kit_retry.py`): candidate start marker for this build is our own
   `[scene_env] build timings` (or `Logging experiment in directory` for training) -- this Lab build does not print
   `Simulation App Startup Complete`; with the wrong marker every launch looked like a failure.
2. **judge work by artifacts, not exit codes** (`scripts/host/run_until_complete.py`): a launch that exits 0 without
   its outputs is counted as a crash and relaunched.
3. **fail fast on the crash signature** (`libcarb.tasking`, `make_fcontext`, `A crash has occurred`) instead of
   waiting out a five-minute startup window.
4. driver 595.58.03 (camera runs stayed alive on it; 580.x did not).
5. **reboot after an Xid 43**, and blacklist `nouveau` after installing the runfile driver.

## Related

NVIDIA forum thread on random SIGSEGV in `libcarb.tasking` on Isaac Sim 6.0.1 (the same signature):
<https://forums.developer.nvidia.com/t/random-sigsegv-in-libcarb-tasking-isaac-sim-6-0-1/>
