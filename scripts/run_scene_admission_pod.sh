#!/usr/bin/env bash
# Runs ON THE POD (image cerberus-remote-v6, code synced to /workspace/cerberus). Takes the pod from an empty
# disk to populated data/scenes/grscenes/{home,commercial}/ + data/scenes/grscenes/admission.json (and optionally
# data/scenes/lightwheel/admission.json) and uploads the ADMITTED scenes to the private R2 prefix
# worlds/grscenes/<category>/. Leaves the pod running (the orchestrator deletes it).
#
# Per category, in order (each step is skipped if its marker under $STATE says it is done, so rerunning the same
# command after a pod restart resumes; extraction, validation and upload are idempotent too):
#   1. fetch    scripts/fetch_grscenes_to_r2.py --extract-only: reads the archive over HTTP ranges (no 74.7 GB
#               download) and extracts ONLY the selected scenes + the models/Materials they reference into
#               $STAGE/<category>/extract, links data/scenes/grscenes/<category> to it.
#   2. validate scripts/validate_scenes.py (parent in plain python3; every phase a fresh isaaclab.sh child with a
#               hard timeout), manifest rewritten after every scene. A phase that hits its timeout is recorded as
#               `fail` with a `timeout ... at stage ...` reason: re-run the same command with `--retry-failed`
#               (it resumes from the manifest) to redo only those rows before concluding a scene is bad.
#   3. upload   only status=pass scenes + the shared assets they reference (deduplicated) + admission.json.
#
# Usage (nohup-friendly; run from the repo root on the pod):
#   nohup bash scripts/run_scene_admission_pod.sh > /workspace/scene_admission.log 2>&1 &
#   bash scripts/run_scene_admission_pod.sh --dry-run                    # print the commands, run nothing
#
# Default plan (LIMITED): the 8 smallest home + 2 smallest commercial scenes (smallest start_result_navigation.usd
# first, so the known-good MVUHLWYKTKJ5EAABAAAAACY8_usd is included). Flags:
#   --categories home,commercial   which archives (default both)
#   --limit N                      N smallest scenes per selected category (default: home 8, commercial 2)
#   --all                          FULL sweep: every scene (downloads the whole 74.7 GB, extracts with 7z)
#   --render                       also require a non-blank RGB frame (driver must render: 570/580, not 595)
#   --combined-env-phases          one env launch (scale envs x the long step count) instead of the 4-env + 16-env
#                                  pair: same checks applied to the one run, ~1/3 less per-scene time
#   --arena-probe N                Arena shared-scene probe for the first N admitted scenes (default 0 = off)
#   --lightwheel N                 also sweep N stratified Lightwheel kitchens (default 0 = off; SDK downloads
#                                  57-112 MB per kitchen)
#   --max-runtime-h H              stop starting new scenes after H hours of validation
#   --purge                        delete $STAGE/<category> after a successful upload (shrinks a full sweep's peak)
#   --skip-fetch | --skip-validate | --skip-upload
#   --dry-run
# Env: STAGE (default /workspace/grscenes), CERBERUS_SCENES_DIR (default <repo>/data/scenes), R2_* from .env.
#
# Expected cost (A5000 pod, $0.27/h; the image is 35 GB, so container disk >= 80 GB for the limited run):
#   limited: disk ~ 15 GB extracted (measured: 2 commercial scenes = 3.1 GiB, 1 home scene = 1.3 GiB; shared models
#            overlap) + 35 GB image -> 80 GB disk is ample. Time ~ 10-15 min fetch + 3-5 min per scene to validate
#            (3 Isaac launches each) = ~ 0.8-1.3 h for 10 scenes + ~ 5 min upload.
#   full:    disk: home parts 50.0 GB + extracted home (UNVERIFIED, <= ~100 GB) while extracting; then commercial
#            parts 23.8 GB + extracted 48.2 GB (36.2 models + 11.9 Materials, measured from the archive's central
#            directory) -> peak ~ 35 + 100 + 24 + 48 ~ 207 GB without --purge, ~ 185 GB with it; use >= 250 GB
#            container disk. Time: ~ 3 h home download + extraction (earlier run), ~ 1.2 h commercial, ~ 99 scenes x
#            4 min validation = ~ 6.5 h -> roughly 11 h, about $3.
set -uo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Host gate + launch probe (docs/runpod.md "Pod hosts"). Both run before any download or Kit launch: a driver
# build that segfaults Isaac Sim stops the session here, while the pod still holds no data.
export CERBERUS_ON_POD=1
bash "$REPO_ROOT/scripts/pod_driver_gate.sh" || { echo "[pod] driver gate rejected this host; stopping" >&2; exit 2; }
bash "$REPO_ROOT/scripts/pod_launch_probe.sh" || { echo "[pod] launch probe failed on this host; stopping" >&2; exit 3; }
# ISAACLAB_PATH etc.; R2_* credentials. Sourced, never printed.
if [[ -f /etc/cerberus_env.sh ]]; then source /etc/cerberus_env.sh; fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi

STAGE="${STAGE:-/workspace/grscenes}"
export CERBERUS_SCENES_DIR="${CERBERUS_SCENES_DIR:-$REPO_ROOT/data/scenes}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Scene assets are pod-only (docs/worlds.md); this unlocks the guarded fetch/validate scripts.
export CERBERUS_ON_POD=1
STATE="$STAGE/state"
CATEGORIES="home,commercial"
LIMIT=""
ALL=0; RENDER=0; PURGE=0; DRY=0; COMBINED=0
ARENA_PROBE=0; LIGHTWHEEL=0; MAX_RUNTIME_H=""
DO_FETCH=1; DO_VALIDATE=1; DO_UPLOAD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --categories) CATEGORIES="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --all) ALL=1; shift ;;
        --render) RENDER=1; shift ;;
        --combined-env-phases) COMBINED=1; shift ;;
        --arena-probe) ARENA_PROBE="$2"; shift 2 ;;
        --lightwheel) LIGHTWHEEL="$2"; shift 2 ;;
        --max-runtime-h) MAX_RUNTIME_H="$2"; shift 2 ;;
        --purge) PURGE=1; shift ;;
        --skip-fetch) DO_FETCH=0; shift ;;
        --skip-validate) DO_VALIDATE=0; shift ;;
        --skip-upload) DO_UPLOAD=0; shift ;;
        --dry-run) DRY=1; shift ;;
        *) echo "[scene_admission] unknown argument: $1" >&2; exit 2 ;;
    esac
done

say() { echo "[scene_admission $(date -u +%H:%M:%S)] $*"; }
run() { echo "+ $*"; if [[ $DRY -eq 0 ]]; then "$@"; fi; }

# rclone copyto with the remote configured from R2_* env vars (same settings as scripts/sync_to_r2.py); no output of values.
r2_copyto() {
    RCLONE_CONFIG_CERBERUS_R2_TYPE=s3 RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare \
    RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    RCLONE_CONFIG_CERBERUS_R2_ACL=private RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true \
        rclone copyto "$1" "cerberus_r2:${R2_BUCKET}/$2"
}

need_r2() {
    local missing=()
    for v in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET; do
        [[ -n "${!v:-}" ]] || missing+=("$v")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        say "ERROR: missing ${missing[*]} (put them in .env at the repo root); skipping uploads"
        return 1
    fi
}

limit_for() {  # $1 category -> scenes to take
    if [[ -n "$LIMIT" ]]; then echo "$LIMIT"; elif [[ "$1" == home ]]; then echo 8; else echo 2; fi
}

if [[ -z "${ISAACLAB_PATH:-}" ]]; then export ISAACLAB_PATH=/workspace/isaaclab; fi
mkdir -p "$STATE"
RC=0
SEL_DESC="limited"; [[ $ALL -eq 1 ]] && SEL_DESC="ALL scenes"
say "plan: $SEL_DESC, categories=$CATEGORIES, stage=$STAGE, scenes dir=$CERBERUS_SCENES_DIR, render=$RENDER, arena_probe=$ARENA_PROBE, lightwheel=$LIGHTWHEEL"
df -h "$STAGE" 2>/dev/null | tail -n 1 | sed 's/^/[scene_admission] disk: /'

IFS=',' read -r -a CAT_LIST <<< "$CATEGORIES"
for cat in "${CAT_LIST[@]}"; do
    n="$(limit_for "$cat")"
    if [[ $ALL -eq 1 ]]; then sel=(--all); tag=all; else sel=(--scenes "$n"); tag="n$n"; fi
    stage_cat="$STAGE/$cat"

    if [[ $DO_FETCH -eq 1 && ! -f "$STATE/fetch_${cat}_$tag.done" ]]; then
        say "[$cat] fetch ($tag)"
        if run python3 scripts/fetch_grscenes_to_r2.py --category "$cat" "${sel[@]}" --extract-only \
                --stage "$stage_cat" --link-to "$CERBERUS_SCENES_DIR/grscenes/$cat"; then
            [[ $DRY -eq 1 ]] || touch "$STATE/fetch_${cat}_$tag.done"
        else
            say "[$cat] fetch FAILED"; RC=1; continue
        fi
    fi

    if [[ $DO_VALIDATE -eq 1 ]]; then
        say "[$cat] validate"
        vargs=(--source grscenes --categories "$cat" --resume)
        [[ $ALL -eq 1 ]] || vargs+=(--limit "$n")
        [[ $RENDER -eq 1 ]] && vargs+=(--render)
        [[ $COMBINED -eq 1 ]] && vargs+=(--combined-env-phases)
        [[ "$ARENA_PROBE" -gt 0 ]] && vargs+=(--arena-probe "$ARENA_PROBE")
        [[ -n "$MAX_RUNTIME_H" ]] && vargs+=(--max-runtime-h "$MAX_RUNTIME_H")
        run python3 scripts/validate_scenes.py "${vargs[@]}" || { say "[$cat] validate exited non-zero"; RC=1; }
    fi

    if [[ $DO_UPLOAD -eq 1 && ! -f "$STATE/upload_${cat}_$tag.done" ]]; then
        say "[$cat] upload admitted scenes"
        ids="$STATE/admitted_$cat.txt"
        if [[ $DRY -eq 1 ]]; then
            echo "+ (list admitted $cat scenes from $CERBERUS_SCENES_DIR/grscenes/admission.json > $ids)"
            admitted_n=1
        else
            # Writes the ids and prints how many: the guard below branches on the COUNT, never on the file
            # size (an empty list must write an empty file, not a lone newline that a -s test reads as content).
            admitted_n="$(python3 - "$cat" "$ids" <<'PY'
import sys

from common.cerberus_worlds import admission

try:
    ids = admission.admitted_scene_ids("grscenes", sys.argv[1])
except FileNotFoundError:
    ids = []
print(admission.write_scene_ids_file(sys.argv[2], (i.split(":", 1)[1] for i in ids)))
PY
)"
        fi
        if [[ $DRY -eq 0 && "$admitted_n" == "0" ]]; then
            say "[$cat] no admitted scenes; nothing to upload"
        elif need_r2 || [[ $DRY -eq 1 ]]; then
            if run python3 scripts/fetch_grscenes_to_r2.py --category "$cat" --from-tree \
                    --scene-ids-file "$ids" --stage "$stage_cat"; then
                [[ $DRY -eq 1 ]] || touch "$STATE/upload_${cat}_$tag.done"
                if [[ $PURGE -eq 1 && $DRY -eq 0 ]]; then
                    say "[$cat] purging $stage_cat"; rm -rf "$stage_cat"
                fi
            else
                say "[$cat] upload FAILED"; RC=1
            fi
        else
            RC=1
        fi
    fi
done

if [[ $DO_UPLOAD -eq 1 && -f "$CERBERUS_SCENES_DIR/grscenes/admission.json" ]] && need_r2; then
    run r2_copyto "$CERBERUS_SCENES_DIR/grscenes/admission.json" worlds/grscenes/admission.json || RC=1
fi

if [[ "$LIGHTWHEEL" -gt 0 && $DO_VALIDATE -eq 1 ]]; then
    say "[lightwheel] validate $LIGHTWHEEL stratified kitchens"
    largs=(--source lightwheel --lightwheel-subset "$LIGHTWHEEL" --resume)
    [[ $RENDER -eq 1 ]] && largs+=(--render)
    [[ -n "$MAX_RUNTIME_H" ]] && largs+=(--max-runtime-h "$MAX_RUNTIME_H")
    run python3 scripts/validate_scenes.py "${largs[@]}" || { say "[lightwheel] validate exited non-zero"; RC=1; }
    if [[ $DO_UPLOAD -eq 1 && -f "$CERBERUS_SCENES_DIR/lightwheel/admission.json" ]] && need_r2; then
        run r2_copyto "$CERBERUS_SCENES_DIR/lightwheel/admission.json" worlds/lightwheel/admission.json || RC=1
    fi
fi

if [[ $DRY -eq 0 ]]; then
    python3 - <<'PY'
from collections import Counter

from common.cerberus_worlds import admission

for source in ("grscenes", "lightwheel"):
    try:
        rows = admission.load_manifest(source)
    except FileNotFoundError:
        continue
    by_cat = Counter((r.category, r.status) for r in rows)
    print(f"[scene_admission] {source}: {len(rows)} rows {dict(sorted(by_cat.items()))}")
PY
    touch "$STATE/DONE"
fi
say "finished (rc=$RC); the pod is left running, delete it explicitly"
exit "$RC"
