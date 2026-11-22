# Object storage audit: R2 bucket `cerberus-robotics` (2026-11-21)

Read-only audit. Nothing has been deleted. Object lists behind the DELETE and REVIEW rows are saved outside the repo at
`~/cerberus-backups/r2-audit/candidates_DELETE.tsv` and `candidates_REVIEW.tsv` (path, bytes, rule).

Method: `rclone lsf -R --format sp` per prefix for object-level detail (no modification time: `--format t` costs one
request per object and stalled a 12k-object tree for half an hour), and `rclone size --fast-list` for authoritative
counts. Nothing was downloaded. Sizes are GiB (2^30).

## Totals

| measure | objects | size |
|---|---|---|
| bucket today | 23,994 | 27.31 GiB |
| projected after the DELETE rows | 22,626 | 25.91 GiB |
| projected after DELETE and REVIEW | 21,558 | 25.59 GiB |

The cache is 92% of the bucket: `worlds/grscenes/home/` alone is 22,482 objects and 24.88 GiB of GRScenes assets plus
shared `Materials` and `models`. Everything else together is 1,502 objects and 2.27 GiB.

## Top level

| prefix | objects | size | oldest | newest | recommendation | why |
|---|---|---|---|---|---|---|
| `backups/` | 1 | 3.6 MiB | 2026-11-21 | 2026-11-21 | **KEEP** | the pre-cleanup git bundle; the backup of record for this work |
| `cache/` | 1 | 8.3 MiB | 2026-11-13 | 2026-11-13 | **DELETE** | Kit warm-cache bundle; the hypothesis it served was refuted (also removed from the repo, row 9 of the cleanup audit) |
| `jobs/` | 596 | 0.28 GiB | 2026-11-14 | 2026-11-16 | **mixed** | per-job logs and outputs; job-mode attempts are superseded |
| `locomotion/` | 465 | 1.21 GiB | 2026-09-02 | 2026-11-08 | **mixed** | training runs; intermediates dominate |
| `migration/` | 308 | 0.23 GiB | 2026-10-11 | 2026-11-16 | **mixed** | evidence of the Phase 0.5 gate and the H4 navigation work, plus per-job code bundles |
| `scene_admission/` | 68 | 72.1 MiB | 2026-10-11 | 2026-10-11 | **REVIEW → DELETE** | per-scene launch logs of one completed admission run; the verdict itself is `worlds/grscenes/admission.json` (kept) |
| `slam/` | 18 | 0.25 GiB | 2026-09-24 | 2026-09-25 | **KEEP** (REVIEW on the bags) | SLAM evaluation evidence, in scope |
| `spike/` | 45 | 0.12 GiB | 2026-10-11 | 2026-11-16 | **DELETE** | per-pod code bundles and a staged copy of a checkpoint: temporary by construction |
| `worlds/` | 22,492 | 25.04 GiB | 2025-01-22 | 2026-11-16 | **KEEP** (one fixture DELETE) | the preprocessed scene cache and the lite scene |

## Second level

`K/D/R` counts objects; the last two columns are the bytes removed by each decision.

| prefix | objects | size | oldest | newest | recommendation | delete | review |
|---|---|---|---|---|---|---|---|
| `backups/` | 1 | 3.6 MiB | 2026-11-21 | 2026-11-21 | KEEP | — | — |
| `cache/kit/` | 1 | 8.3 MiB | 2026-11-13 | 2026-11-13 | DELETE | 8.3 MiB | — |
| `jobs/h4job-*` (22 job dirs) | 382 | 3.0 MiB | 2026-11-14 | 2026-11-16 | DELETE | 3.0 MiB | — |
| `jobs/newton-*` (10 job dirs) | 214 | 0.28 GiB | 2026-11-14 | 2026-11-16 | mixed (199K/15D) | 0.10 GiB | — |
| `locomotion/checkpoints/` | 1 | 6.6 MiB | 2026-09-08 | 2026-09-08 | KEEP | — | — |
| `locomotion/unitree_go2_flat/` | 113 | 0.09 GiB | 2026-09-02 | 2026-09-03 | mixed (41K/72D) | 0.07 GiB | — |
| `locomotion/unitree_go2_rough/` | 351 | 1.20 GiB | 2026-09-03 | 2026-11-08 | mixed (211K/140D) | 0.90 GiB | — |
| `migration/adversarial_gate_20261108T051752Z/` | 11 | 3.7 MiB | 2026-11-08 | 2026-11-08 | KEEP | — | — |
| `migration/adversary_adversary_20261108T051752Z/` | 83 | 0.14 GiB | 2026-11-08 | 2026-11-08 | mixed (23K/60D) | 0.14 GiB | — |
| `migration/attack_videos_20261108T212630Z/` | 6 | 2.1 MiB | 2026-11-08 | 2026-11-08 | KEEP | — | — |
| `migration/characterization_20261108T212213Z/` | 2 | 1.6 MiB | 2026-11-08 | 2026-11-08 | KEEP | — | — |
| `migration/ga_smoke_20261106T213400Z/` | 11 | 0.4 MiB | 2026-11-06 | 2026-11-06 | KEEP | — | — |
| `migration/ga_smoke_20261106T214925Z/` | 12 | 0.5 MiB | 2026-11-06 | 2026-11-06 | KEEP | — | — |
| `migration/ga_upgrade_2026-10-25/` | 9 | 1.0 MiB | 2026-10-25 | 2026-10-25 | KEEP | — | — |
| `migration/gate_eval_20261108T055128Z/` | 1 | 9.5 KiB | 2026-11-08 | 2026-11-08 | KEEP | — | — |
| `migration/h4_nav/` (20 runs) | 164 | 0.07 GiB | 2026-10-11 | 2026-11-16 | mixed (117K/47D) | 0.06 GiB | — |
| `migration/scene_eval_20261108T213809Z/` | 9 | 2.2 MiB | 2026-11-08 | 2026-11-08 | KEEP | — | — |
| `scene_admission/reports/` | 68 | 72.1 MiB | 2026-10-11 | 2026-10-11 | REVIEW → DELETE | — | 72.1 MiB |
| `slam/fast_lio2/` | 18 | 0.25 GiB | 2026-09-24 | 2026-09-25 | KEEP (7 bags: REVIEW) | — | 0.25 GiB |
| `spike/newton-*` (15 runs) | 45 | 0.12 GiB | 2026-10-11 | 2026-11-16 | DELETE | 0.12 GiB | — |
| `worlds/grscenes/admission.json` | 1 | 30 KiB | 2026-10-11 | 2026-10-11 | KEEP | — | — |
| `worlds/grscenes/home/` | 22,482 | 24.88 GiB | 2025-01-22 | 2026-10-11 | KEEP | — | — |
| `worlds/grscenes_lite/` | 4 | 0.16 GiB | 2026-11-16 | 2026-11-16 | KEEP | — | — |
| `worlds/rehearsal/` | 5 | 815 B | 2026-11-15 | 2026-11-15 | DELETE | 815 B | — |

`worlds/grscenes/home/` breaks down as `Materials/` 4,000 objects / 4.66 GiB, `models/` 18,352 / 20.16 GiB and
`scenes/` 129 / 59.2 MiB. Newest and oldest modification times for the cache are preserved source timestamps
(2025-01-22 comes from the GRScenes assets), not upload dates.

## What the repository references (so must be kept)

| prefix | referenced by |
|---|---|
| `worlds/grscenes/home/` | `scripts/fetch_grscenes_from_r2.py` (`DEFAULT_PREFIX`), `scripts/fetch_grscenes_to_r2.py`, `docs/worlds.md`, `docs/h4_report.md`, `docs/reports/scene_eval_phase0.md` |
| `worlds/grscenes/admission.json` | `common/cerberus_worlds/admission.py`, `scripts/validate_scenes.py`, `docs/worlds.md`, `docs/reports/scene_eval_phase0.md` |
| `worlds/grscenes_lite/` | `docs/runpod.md`, `scripts/rehearsal/run_lite.sh` (kept: it is the lite-scene driver) |
| `locomotion/unitree_go2_rough/` | `docs/artifact_storage.md`, `common/cerberus_policies/cards/heightscan_ppo.yaml` (`checkpoint_url` names `2026-11-07_07-29-40`), `docs/h2_report.md`, `docs/h3_report.md`, `policy/locomotion/runs/RUN_001.md` |
| `locomotion/unitree_go2_flat/` | `docs/artifact_storage.md`, `policy/locomotion/runs/RUN_001.md` |
| `locomotion/checkpoints/go2_locomotion_v1_candidate.pt` | `policy/locomotion/README.md`, `policy/locomotion/runs/RUN_001.md`, `RUN_002.md`, `docs/reports/phase0_ga_parity.md`, `system/cerberus_bringup/config/platforms.yaml` |
| `migration/ga_smoke_*`, `migration/gate_eval_*`, `migration/adversary_*`, `migration/characterization_*`, `migration/attack_videos_*`, `migration/scene_eval_*` | `docs/h2_report.md` (bucket table), `docs/reports/phase05_adversarial_report.md`, `docs/reports/scene_eval_phase0.md`, `docs/arena_integration_log.md` |
| `migration/h4_nav/20261110T0857Z/`, `.../20261113T1915Z/` | `docs/h4_report.md` (occupancy grids, kinematic aggregates) |
| `jobs/newton-20261116T061409Z/`, `jobs/newton-20261116T071200Z/`, `jobs/newton-20261115T153614Z/` | `docs/reports/newton_spike.md` (phase 5 artifacts; "checkpoints stay on R2") |
| `slam/fast_lio2/` | `docs/arena_integration_log.md`, `docs/runpod.md` (R2 mirror of the eval run dirs) |
| `scene_admission/reports/` | `docs/arena_integration_log.md` (calls them artifacts; the verdict is in `admission.json`) |
| `backups/` | created by this cleanup; the backup of record |

## Rules behind the DELETE rows

1. **Intermediate checkpoints.** Keep the last `model_<n>.pt` per run directory plus `model_best_*.pt`, `exported/`,
   `eval/`, `videos/`, `params/`, `git/` and the run's `latest.json`/`meta.json`. Delete the earlier periodic
   `model_N.pt` files (every 50 iterations). The cited ones survive: `model_1499.pt` (final) and `model_1300.pt`
   (RUN_001's resume point) in `2026-09-03_06-56-49`, and `model_best_{easy,medium,hard}.pt` everywhere.
2. **Adversary.** Keep `model_1000.pt` (the frozen adversary named in `gate_verdict.json`) and the tensorboard
   events; delete the 60 earlier periodics.
3. **Code bundles.** Every `*/code/*.bundle` is the source bundle a past job fetched. Delete; the repo and the
   archive tags hold the code.
4. **Staged inputs.** `migration/h4_nav/*/inputs/checkpoints/` holds six identical copies of the GA candidate
   `go2_locomotion_ga_v1_candidate.{pt,onnx,onnx.data,meta.json}`. Keep the newest
   (`20261115T174943Z/`), which is the **only ONNX copy of that checkpoint on R2**, delete the other five.
5. **Per-job probe output.** `evidence/`, `session/`, `repo_outputs/kit_logs/`, `launch_probe*.csv`, `status.json`,
   `log_tail.txt` of superseded job-mode attempts: delete.
6. **Fixtures and caches.** `worlds/rehearsal/` (the tiny rehearsal scene the archived rehearsal driver fetched) and
   `cache/kit/`: delete.
7. **Newton per-run checkpoints.** In `jobs/newton-20261115T035749Z/rl_logs/` and the probe dirs, keep the last
   checkpoint of each directory, delete the earlier ones. The phase-5 retrain's checkpoints stay in full because the
   report tells the reader to fetch `model_1250.pt` and neighbours.

## The two REVIEW rows

- **`slam/fast_lio2/` 7 rosbags (0.25 GiB).** Raw mission bags behind the SLAM evaluation. The scored results,
  `smoke_check.json`, `sim_loop_report.json` and the videos are kept either way. Keep by default because SLAM
  evaluation is in scope and the bags are its raw evidence; delete only to reclaim the space.
- **`scene_admission/reports/` 68 objects (72.1 MiB).** Per-scene Isaac launch logs from the 2026-10-11 admission
  run plus `SUPERVISOR_STATUS.txt` and a gate render. The admitted-scene verdicts live in
  `worlds/grscenes/admission.json`, which is kept. Recommended DELETE.

## Lifecycle rules to set (dashboard)

| prefix | rule | why |
|---|---|---|
| `jobs/` | expire objects 90 days after upload | per-job logs are diagnostic, not evidence; the cited Newton job dirs are exempted by copying them to `migration/` or by a manual hold |
| `spike/` | expire 30 days | temporary pod bundles |
| `cache/` | expire 30 days | the warm-cache experiment is closed; this also stops it growing again |
| `worlds/grscenes/home/` | no expiry; keep versions off | the preprocessed cache is expensive to rebuild |
| `migration/`, `locomotion/`, `slam/`, `backups/`, `worlds/grscenes_lite/` | no expiry | evidence, final checkpoints, backup |
