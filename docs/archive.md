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

### Replaced (2026-11-22)

The four job entry points were re-created provider-neutral in `scripts/host/`, from these archived originals as
reference (not copies: the pod gate and dead-man switch are gone, paths are environment variables, and the provider
self-terminate became an explicit stop hook). The originals stay in the tag; the replacements are what to run now.

| entry point | archived original | provider-neutral replacement |
|---|---|---|
| GA-stack bring-up checks | `scripts/run_ga_stack_check_pod.sh` | `scripts/host/run_stack_check.sh` |
| scene admission (fetch, validate, upload) | `scripts/run_scene_admission_pod.sh` | `scripts/host/run_scene_admission.sh` |
| scene evaluation | `scripts/run_scene_eval_pod.sh` | `scripts/host/run_scene_eval.sh` |
| SLAM evaluation | `scripts/run_slam_eval_remote.sh` | `scripts/host/run_slam_eval.sh` |

Each replacement starts with `--help`, supports `--dry-run`, and expects the host to pass
`scripts/host/host_doctor.py` (`docs/host_contract.md`). For the archived originals themselves:

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

## Session 4 (2026-11-24, tag `archive/session4-2026-11-24`)

Removed from `main` because the Brev relay was dropped and the session reports had served their purpose; all files
were uploaded to R2 `archive/2026-11-24/repo-tidy/` (21 files, byte-verified) before deletion:

| path | why |
|---|---|
| `docs/reports/session3/` | session-3 ledger, log, state and handoff; superseded by `docs/reports/jeplo_spike.md`, `docs/showcase.md` and `docs/dev_machine.md` |
| `docs/reports/first_scene_session.md` | session-1/2 narrative; its lasting lessons are in `docs/dev_machine.md` "Known issues" |
| `docs/reports/render_unblock.md` | RTX/material recipe; the lasting steps moved to "Known issues" (material conversion wrapper + reboot) |
| `docs/reports/kit_crash_report.md`, `kit_reliability.md` | Kit stability notes, superseded by "Known issues" |
| `docs/reports/brev_instances.md`, `brev_reply.md`, `brev_ssh_issue.md` | Brev is no longer used |
| `scripts/host/brev_*.sh`, `tests/test_brev_wrappers.py` | Brev wrappers and their tests |
| `scripts/host/r2_channel.{py,sh}`, `tests/test_r2_channel.py` | R2 command channel, not used by the Hyperstack plain-SSH path |

Restore with `git checkout archive/session4-2026-11-24 -- <path>`.
