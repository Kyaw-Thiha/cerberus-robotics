# Newton feasibility spike — kit-less Isaac Lab on the Newton backend

Status: **in progress** (numbers filled from the job's R2 artifacts; sections marked *pending* are not yet
written). Branch `newton-spike`; code in `spikes/newton/`, nothing outside it changed except this report
and `plans/handoff_newton_spike.md`.

## The question

Can Cerberus run its Go2 locomotion loop on Isaac Lab's **kit-less** install with the **Newton** physics
backend — a path that does not need Isaac Sim (the 35 GB image, the RTX requirement, and the 595.x startup
crash), while the PhysX / Isaac Sim path stays the high-fidelity one?

## What was fixed before measuring

* Isaac Lab is pinned to tag **`v3.0.0` = `e3f7d9ff`**, the same commit the Cerberus `v6-ga` image pins, so
  every number here is measured on the stack the current policies were produced on, not on a newer one.
* At that tag `isaaclab.sh` is a 104-line deprecated wrapper: **the `-i newton,tasks,assets,rl[rsl_rl]`
  selector installer the handoff mentions no longer exists there**. The documented kit-less install at that
  commit is the one its own `docker/Dockerfile.kitless` uses:
  `uv sync --frozen --inexact --extra video --extra importers` (Newton, rsl-rl, Warp and OpenUSD via
  `usd-exchange` are base dependencies).
* Image: `nvidia/cuda:13.0.0-base-ubuntu24.04` (public Docker Hub tag, verified via the registry manifest),
  ~0.25 GB, versus the 35 GB Cerberus image. No SSH: the pod runs as a job whose start command decodes the
  bootstrap from an env variable, heartbeats to R2 and stops itself.

## Host

*pending*

## Stage results

*pending*

## Decision table

*pending*

## Effort estimate for adoption

*pending*

## Limitations

*pending*
