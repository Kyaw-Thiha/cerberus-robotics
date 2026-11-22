# Object storage

The R2 bucket `cerberus-robotics`: what each prefix is for and how long it should live. The measured contents and the
audit that produced the delete list are in `docs/storage_audit.md`.

Access is an S3-compatible token scoped to this bucket only (bucket listing is denied by design). The tooling builds a
remote from `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` and `R2_BUCKET` exactly as
`scripts/sync_to_r2.py` does; see `docs/artifact_storage.md` for the upload flow. Never build the remote from a stored
config file.

## Layout

| prefix | what it holds | retention |
|---|---|---|
| `worlds/grscenes/home/` | GRScenes home mirror: `scenes/`, shared `Materials/` and `models/`, `manifest.json`. The preprocessed cache that scene work reads. Private, CC BY-NC-SA, never publish. | keep; no expiry. Expensive to rebuild (about 25 GB). |
| `worlds/grscenes/admission.json` | the scene admission manifest: one row per scene, verdict and validated spawns | keep; regenerated only when scenes are re-admitted |
| `worlds/grscenes_lite/` | the lite variant per scene (`exact/`, occupancy, `lite_report.json`) | keep; rebuilt by the lite-scene driver |
| `locomotion/<experiment>/<run>/` | training artifacts: final and best checkpoints, `exported/`, `eval/`, `videos/`, `params/`, `git/`, `latest.json` (the newest run) | keep; final checkpoints cited in reports are permanent |
| `migration/<job>/` | short-lived job evidence: gate verdicts, adversary, attack videos, smoke runs, H4 navigation grids | keep the evidence cited in reports; per-job code bundles and staged inputs are disposable |
| `jobs/<job-id>/` | per-job logs, status and outputs, written by the job runner as it runs | 90 days after upload; keep anything cited in a report |
| `spike/` | temporary bundles and staged inputs used to bootstrap a spike pod | 30 days |
| `slam/<backend>/<run>/` | SLAM evaluation runs: mission bag, video, `latest.json` | keep; the bags are the raw evidence |
| `scene_admission/` | per-scene launch logs from an admission run | 30 days; the verdict lives in `worlds/grscenes/admission.json` |
| `cache/` | Kit startup-cache bundles (experiment closed; the code is gone) | 30 days, and nothing writes here any more |
| `backups/` | git bundles, one per cleanup | keep; the newest is the backup of record |

## Rules

- **Artifacts live here, never in the repo.** Checkpoints, videos, bags, scene assets and per-job logs are gitignored in
  the repository because they are large and regenerable or machine-specific.
- **A checkpoint is kept once, at its final path.** Per run keep the last periodic checkpoint, `model_best_*.pt`,
  `exported/` (ONNX plus meta) and the eval outputs; earlier periodic checkpoints are disposable.
- **What a report cites is permanent.** If a document names a run directory or an object, that prefix is kept regardless
  of the rules above.
- **Nothing is deleted without a manifest.** The delete lists that produced the 2026-11-21 cleanup are stored outside the
  repository next to the git bundles, and every deletion starts with a dry run.
- **The scene cache is pod-only.** It is read and written on the GPU host; nothing scene-sized is downloaded to a
  laptop.

## Lifecycle rules to set (dashboard)

| prefix | rule |
|---|---|
| `jobs/` | expire 90 days after upload, except the Newton phase-5 job directories a report cites |
| `spike/` | expire 30 days |
| `cache/` | expire 30 days |
| `scene_admission/` | expire 90 days |
| `worlds/`, `locomotion/`, `migration/`, `slam/`, `backups/` | no expiry |
