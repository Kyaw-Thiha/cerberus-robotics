# Cleanup report, 2026-11-21

What was removed, what replaced it, and where it lives now. The audit that decided it is `docs/cleanup_candidates.md`;
the object-storage audit is `docs/storage_audit.md`; the archive index and restore commands are `docs/archive.md`.

## Repository

| | before (`e281cc7`) | after (`5d55828`, plus the final docs commit) |
|---|---|---|
| tracked files | 531 | 456 |
| text lines (binaries and `pixi.lock` excluded) | 78,152 | 63,017 |
| `pixi run pytest -q` | 1359 passed, 20 skipped | 1285 passed, 20 skipped |
| `pixi run lint` | clean, 18 ignored rule families | clean, 8 ignored rule families |
| Markdown links | 0 broken | 0 broken |
| backend table | fresh | fresh (`gen_backend_table.py --check` exit 0) |
| colcon build | — | 3 packages (`common`, `system`) to a `/tmp` prefix, exit 0 |

Test count: 74 fewer. 69 were the tests of removed provider tooling; 6 more were the tests of removed modules inside
`test_pod_probe_tools.py`; one redaction test moved from the removed file into `test_redact.py`.

Lines by top-level directory, before → after:

| | before | after |
|---|---|---|
| `common` | 15,089 | 15,119 |
| `docs` | 11,377 | 6,855 |
| `policy` | 13,163 | 13,103 |
| `scripts` | 21,879 | 17,165 |
| `slam` | 5,807 | 5,806 |
| `planning` | 2,181 | 2,186 |
| `docker` | 1,544 | 1,345 |
| `spikes` | 4,151 | 0 |
| `plans` | 1,521 | 0 |

## What moved where

| removed | how it is preserved |
|---|---|
| provider tooling: RunPod and Vast creators and adapters, job mode, guards, rehearsal fixtures, per-provider pod drivers, their tests, the RunPod runbook | tag `archive/provider-tooling-2026-11-21` |
| `spikes/newton/` | tag `archive/newton-spike-2026-11-21` (the branch tip, a strict superset of `main`'s copy) |
| `docker/remote/` (Sim 5.1 / Lab 2.3.2) | tag `pre-cleanup-2026-11-21`, commit `2a84c1f`; the published image is untouched on Docker Hub |
| the whole pre-cleanup tree | tag `pre-cleanup-2026-11-21`, plus the git bundle in `backups/` on R2 |
| Kit warm-cache bundle, FAQ, duplicate status report, agent planning artifacts, six agent handoffs, two dead functions | deleted; history keeps them |

Converted rather than lost: `docs/worlds_integration_spec.md`, `docs/sensor_setup.md`, `docs/slam_design.md`,
`docs/adversarial_gate_method.md` (handoffs that live code cites), and `docs/platform_config.md` (the two platform
design specs). `docs/reports/newton_spike.md` was replaced with the newer version from the tag.

## Tags

- `pre-cleanup-2026-11-21` — `main` before any removal (`e281cc7`).
- `archive/provider-tooling-2026-11-21` — the commit before the provider removal (`0453378`).
- `archive/newton-spike-2026-11-21` — the `newton-spike` tip (`5c76143`).
- `pre-cleanup-2026-11-08` — an older safety tag, kept.

`docs/archive.md` lists what each holds, the four job entry points that currently have no driver, and the restore
command.

## Branches and worktrees

Before: `main` (75 ahead of origin), `newton-spike` in its own worktree, `cleanup-2026-11-08`, plus two remote
branches. After: `main` only, one worktree, no stash. `newton-spike` was tagged then deleted; `cleanup-2026-11-08` and
`h1-docs-interfaces` were deleted locally and on the remote after its content was re-applied (the ruff burn-down) or
confirmed superseded (`blind_ppo.yaml` became `blind_ppo_flat.yaml` in H3, which is in `main`).

## Object storage

Measured with `rclone size --fast-list` before and after; deletes ran object by object after a per-object dry run.

| prefix | before | after |
|---|---|---|
| bucket | 23,994 objects / 27.315 GiB | 23,875 / 27.209 GiB |
| `cache/` | 1 / 8.3 MiB | 0 |
| `worlds/rehearsal/` | 5 / 815 B | 0 |
| `scene_admission/` | 68 / 72.1 MiB | 7 / 4.6 MiB |
| `spike/` (Newton bundles held back) | 45 / 138 MiB | 45 / 138 MiB |
| `jobs/` (status, logs, tails only) | 596 / 292 MiB | 559 / 289 MiB |
| `migration/` (per-pod code bundles) | 308 / 225 MiB | 293 / 196 MiB |
| `locomotion/` (all checkpoints held) | 465 / 1.297 GiB | 465 / 1.297 GiB |
| `worlds/` | 22,492 / 25.044 GiB | 22,487 / 25.044 GiB |

Total freed: 119 objects, 107.77 MiB (113.0 MB), 0.4% of the bucket.

**The bucket is not "clean" and was never going to be.** The preprocessed scene cache is 92% of it: `worlds/` holds
25.04 of 27.21 GiB, of which `worlds/grscenes/home/` is 22,482 objects and 24.88 GiB of GRScenes scenes, materials and
model geometry. That cache is deliberate, it is expensive to rebuild, and it stays. The deletions here are per-job and
per-experiment leftovers; the substantial reduction available is not in this cleanup but in whether that cache is
wanted at all.

Guards applied to the delete list, per the owner's conditions: everything under `jobs/newton-*` held; all checkpoints
(intermediate, adversary, staged candidate sets) held; all videos and stills held; SLAM rosbags held; and any object
whose name is cited in `docs/` held. That removed 60 of the 179 candidates; the list is in
`~/cerberus-backups/r2-audit/manifests2/DROPPED.tsv`, with the per-object delete and dry-run logs beside it.

## Commits

```
0453378 docs: cleanup candidates and object storage audit (Phase A, nothing removed)
5ee3131 archive: provider tooling and per-provider pod drivers
116eec1 delete: Kit warm-cache bundle
6d63f95 archive: Newton spike code, keep the report from the branch tip
4b92907 delete: legacy docker/remote image recipe
d32a74e plans: turn the design handoffs into docs, delete the agent instructions
7b84a1e docs: one platform configuration doc replaces the superseded specs
225cb4c code: remove two functions with no caller
b624ca2 docs: README, structure and links follow the cleanup
775a305 docs: object storage layout and retention rules
5d55828 style: enable ten ignored ruff families and fix their hits
```

## Left alone, deliberately

- The generic host tooling stays where it is: driver gate and allowlist, launch probe, `kit_retry`, `run_resilient`,
  `job_attempt_log`, `kit_logs`, `launch_gate`, `launch_hygiene`, `pod_launch_probe_env`, `redact`, `scan_secrets`,
  `pod_guard`, `preflight_local`, the R2 sync and fetch scripts, and the scene tools. A later task reorganises them.
- `scripts/rehearsal/run_lite.sh` is the only driver of `build_lite_scene.py` and the lite scene is in scope, so it
  stayed even though its folder name now overstates it.
- `scripts/preprocess_scene.py`, `finalize_ga_checkpoint.py` and `scan_secrets.py` were kept despite thin references.
- Pod-composed helpers that only tests call (`scene_env.py`, `video_clips.py`, several `nav_metrics` helpers) were
  reported, not removed: they cannot be reachability-checked without Isaac.
- `RateLimitedChannel` in `attack_math.py` is superseded by the new command limiter, but three docs describe it as
  unrelated infrastructure and one test covers it, so it stayed.
- Duplicated helpers (parity, ONNX fallbacks, attack characterization) were reported, not merged.
- Reports under `docs/reports/` are evidence: unchanged, except `newton_spike.md` (newer version) and one method
  reference that now points at `docs/adversarial_gate_method.md`.

## Commands run for the push (already executed)

```bash
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  push https://github.com/Kyaw-Thiha/cerberus-robotics.git main:refs/heads/main
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  push https://github.com/Kyaw-Thiha/cerberus-robotics.git \
  refs/tags/pre-cleanup-2026-11-21 refs/tags/archive/provider-tooling-2026-11-21 refs/tags/archive/newton-spike-2026-11-21
git ... push <url> --delete cleanup-2026-11-08
git ... push <url> --delete h1-docs-interfaces
```

Remote state now: `refs/heads/main` at `5d55828`, `refs/heads/newton-spike` still present (its tip is in
`archive/newton-spike-2026-11-21`; delete it with the same `--delete` command if it should go), and the four tags.

## Open items for the owner

- `origin/newton-spike` still exists on GitHub; its content is the archive tag. Say the word and it is deleted.
- R2 lifecycle rules still need to be set in the dashboard; the recommended set is in `docs/storage.md`.
- The next task's provider-neutral host contract should supply replacements for the four archived entry points listed in
  `docs/archive.md`.
