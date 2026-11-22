# Archive

Content that is no longer on `main` but is kept in git history under a tag. Nothing was lost: `main`'s history holds every
line, and the tags below mark the commits that still had it.

## Restore a file

```bash
git checkout <tag> -- <path>        # bring one file (or directory) back into the working tree
git show <tag>:<path>               # read it without touching the working tree
git switch -c <branch> <tag>        # look at the whole tree as it was
```

## Tags

| tag | points at | holds |
|---|---|---|
| `pre-cleanup-2026-11-21` | `main` before the cleanup | the whole tree as it was, including everything below |
| `pre-cleanup-2026-11-08` | an earlier snapshot | the tree before the H1 merge; historical only |
| `archive/provider-tooling-2026-11-21` | the commit before the provider tooling was removed | the RunPod and Vast adapters and creators, job mode, the laptop guard and its systemd installer, the rehearsal fixtures, the per-provider pod drivers, their tests, and `docs/runpod.md` |
| `archive/newton-spike-2026-11-21` | the tip of branch `newton-spike` | `spikes/newton/` in full, including the phase-5 files that were never on `main` |

## Provider tooling (tag `archive/provider-tooling-2026-11-21`)

Removed because it is provider-specific: pod creation and stop (RunPod and Vast), the provider adapters, job mode and its
entrypoint, the laptop session guard and its systemd unit, the budget and deadman tools, the rehearsal fixtures, the
per-provider pod drivers, the six test files that covered them, and the RunPod runbook (`docs/runpod.md`).

Kept on `main` and unaffected: the driver gate and allowlist, the launch probe, `kit_retry`, `run_resilient`, log capture,
redaction, the secret scan, the preflight script, the R2 sync and fetch scripts, and the scene tools.

### To be replaced

These four are the only entry points for jobs that remain in scope. Their replacements are the next task's work (a
provider-neutral host contract and doctor). Until then, retrieve the driver from the tag and run it as before.

| entry point | was | lives in |
|---|---|---|
| driver gate runner (host check + launch probe before any download) | `scripts/run_ga_stack_check_pod.sh`, with `scripts/pod_isaac_lib.sh` | `archive/provider-tooling-2026-11-21` |
| scene admission (fetch, validate and upload admitted scenes) | `scripts/run_scene_admission_pod.sh` | `archive/provider-tooling-2026-11-21` |
| scene evaluation | `scripts/run_scene_eval_pod.sh` | `archive/provider-tooling-2026-11-21` |
| SLAM evaluation (sim loop, mission bag, scoring, sync) | `scripts/run_slam_eval_remote.sh` | `archive/provider-tooling-2026-11-21` |

```bash
git checkout archive/provider-tooling-2026-11-21 -- scripts/run_slam_eval_remote.sh
```

## Legacy image recipe (no tag of its own)

`docker/remote/` (Isaac Sim 5.1 / Isaac Lab 2.3.2) was removed in the same cleanup and is recoverable from
`pre-cleanup-2026-11-21` or from commit `2a84c1f` (its last change):

```bash
git checkout pre-cleanup-2026-11-21 -- docker/remote
```

It built the published image `kevinkyawthiha/cerberus-remote:v3`. The image itself is untouched on Docker Hub. The
current recipe is `docker/remote_v6/`.

## Newton spike (tag `archive/newton-spike-2026-11-21`)

`spikes/newton/` was removed from `main`. The tag is the branch tip, which is a strict superset of what `main` had: both
`main` commits that touched the directory (`0ebfb8a`, `558df4b`) are ancestors of the tip, and nothing existed only on
`main`. Two files are newer on the tip than they were on `main` (same path, so no merge was needed):

| path | on `main` | on the tag (newer) |
|---|---|---|
| `spikes/newton/clip_stills.py` | `d2b8fa8` | `496a74e` |
| `spikes/newton/tasks.py` | `d50a50e` | `7206411` |

The findings stay in `docs/reports/newton_spike.md` (taken from the tag). Kit-less Newton is an experimental, optional
lane: its code is here, its results are documented.

## Removed without a tag

Deleted outright because they had no reference value and history keeps them: the Kit warm-cache bundle
(`scripts/kit_cache.*`), the FAQ superseded by `docs/isaac_lab_workflow.md`, the duplicate status report, the agent
planning artifacts under `docs/superpowers/` and `docs/archive/superpowers_plans/` (their design content was rewritten
as `docs/platform_config.md`), the agent handoffs in `plans/` (the four that carried design content became
`docs/worlds_integration_spec.md`, `docs/sensor_setup.md`, `docs/slam_design.md` and
`docs/adversarial_gate_method.md`), and two functions with no caller (`xacro_available`, `looks_like_path`).
