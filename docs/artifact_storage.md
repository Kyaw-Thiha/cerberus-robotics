# Cerberus — Permanent Artifact Storage (Cloudflare R2)

Reference doc for where training artifacts end up long-term, and why. Every
rented GPU instance is treated as disposable (see `isaac_lab_workflow.md`) —
this is the "sync before you kill it" half of that policy.

## Why R2, and why rclone

S3-compatible, and storage-only pricing fits "cheap, permanent home for
checkpoints/videos/logs after a disposable pod dies" better than a persistent
GPU-attached volume. For the CLI: Cloudflare's own docs rank three tools by
use case — Wrangler for single-object ops with no credential setup, AWS CLI
for existing AWS workflows, and **rclone for bulk object operations,
migrations, and syncing directories**, which is exactly what this is. Went
with rclone accordingly, configured entirely via `RCLONE_CONFIG_*` env vars
(no `rclone config` / config file needed) so it fits the same "everything
from `.env`" pattern as `WANDB_API_KEY`.

## Bucket layout

Mirrors the local layout under a per-module prefix, so `scripts/sync_to_r2.py`
is one reusable script, not a locomotion-specific one:

```
r2://<bucket>/
  <module>/                         # e.g. "locomotion" -- matches policy/<module>/
    <experiment_name>/                # e.g. "unitree_go2_flat", agent_cfg.experiment_name
      latest.json                       # {"run": "<run_timestamp>"} -- newest run for this experiment
      <run_timestamp>/                  # one training run, same name as the local log_dir
        checkpoints/
          model_0.pt, model_50.pt, ...      # every periodic checkpoint, not just best/final
          model_best_easy.pt / _medium.pt / _hard.pt
        events.out.tfevents...
        params/env.yaml, params/agent.yaml
        git/isaaclab.diff, git/cerberus.diff
        videos/                            # if --video was used
        eval/
          push_recovery_<eval_timestamp>.csv   # any eval sweep run against this checkpoint
```

Local path `policy/locomotion/checkpoints/unitree_go2_flat/2026-09-05_14-32-10/`
becomes `r2://<bucket>/locomotion/unitree_go2_flat/2026-09-05_14-32-10/` — same
relative shape, different root, with one deliberate difference: `model_*.pt`/
`model_best_*.pt` (which rsl_rl's `OnPolicyRunner` saves at the run dir's own
local root, not nested) get uploaded into their own `checkpoints/` subfolder on
R2 instead, so the run's R2 folder reads cleanly as one artifact type per
subfolder rather than dozens of checkpoint files mixed in at the root
alongside `params/`, `git/`, tensorboard events, etc.

**Why eval nests inside the run, not a separate top-level `eval/` collection:**
one R2 prefix then holds everything about a checkpoint — training artifacts and
every eval sweep ever run against it — without cross-referencing filenames to
figure out which checkpoint an eval CSV belongs to.

**Why every periodic checkpoint, not just best/final:** each is small (the
actor/critic MLPs involved are at most a few hundred KB–low MB; even hundreds
of them across many runs is negligible against R2's storage pricing), so there's
no real cost tradeoff — keeping all of them preserves the ability to resume
training from, or inspect, an arbitrary earlier iteration later.

**Why a `latest.json` pointer:** makes "grab the newest Rough checkpoint" a
single known-key fetch (`rclone cat cerberus_r2:<bucket>/locomotion/unitree_go2_rough/latest.json`)
instead of listing and sorting run folders — useful for eval, and later for
the ROS2 inference node picking up a fresh checkpoint.

## Two subcommands: sync-run vs sync-eval

Eval doesn't always run alongside training — often later, sometimes on a
different pod, against a checkpoint that isn't sitting next to a full local
run tree anymore. So `scripts/sync_to_r2.py` has two subcommands rather than
one:

- **`sync-run`** — uploads a full local run directory (checkpoints, videos,
  tensorboard events, params, git diffs) and updates `latest.json` for that
  experiment.
- **`sync-eval`** — uploads just an eval output file to an *existing* run's
  `.../eval/` folder, derived from the `--checkpoint` path used to invoke
  `push_recovery_eval.py` (must be the canonical local path:
  `.../checkpoints/<experiment_name>/<run_timestamp>/model_X.pt`) — no full
  run tree needs to be present locally.

## Running the sync

Standalone script, invoked manually for now (not yet wired into an automated
training-completion wrapper — see the automation design discussion in the
session that added this):

```bash
set -a; source .env; set +a   # loads R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET

python3 scripts/sync_to_r2.py sync-run \
    --module locomotion \
    --run-dir policy/locomotion/checkpoints/unitree_go2_flat/2026-09-05_14-32-10

python3 scripts/sync_to_r2.py sync-eval \
    --module locomotion \
    --checkpoint policy/locomotion/checkpoints/unitree_go2_flat/2026-09-05_14-32-10/model_250.pt \
    --eval-file policy/locomotion/eval/results/push_recovery_2026-09-06_09-00-00.csv
```

Requires `rclone` (v1.59+). Verified end-to-end against the real bucket in
this session (both subcommands, uploaded + inspected via `rclone tree` +
cleaned up via `rclone delete`/`rmdirs` — `rclone purge`'s bucket-versioning
preflight check 403's under this account's API token scope, unrelated to the
actual delete, which works fine). **rclone is already present on the local
machine but not yet in `docker/Dockerfile.remote`.** Since the artifacts live
on the pod, the natural place to run this is on the pod itself (right before
killing it), which means `rclone` needs adding to the Dockerfile before this
is usable end-to-end there. A local install is only needed for testing, or
re-syncing something already pulled down to the local machine.
