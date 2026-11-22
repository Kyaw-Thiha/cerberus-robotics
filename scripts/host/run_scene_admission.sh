#!/usr/bin/env bash
# Scene admission: take a host from an empty disk to populated scene directories plus an admission manifest, and
# upload the admitted scenes to the private bucket. Re-created from `scripts/run_scene_admission_pod.sh` at tag
# archive/provider-tooling-2026-11-21: same three steps and the same resumable markers, without the pod-only
# parts (dead-man switch, provider calls). Upload is a copy, never a delete.
#
# Per category (each step is skipped when its marker under $STAGE/state says it is done, so a re-run resumes;
# extraction, validation and upload are idempotent):
#   1 fetch     fetch_grscenes_to_r2.py --extract-only: HTTP ranges, extracts only the selected scenes
#   2 validate  validate_scenes.py: manifest rewritten after every scene; --retry-failed resumes
#   3 upload    only admitted scenes + the shared assets they reference + admission.json
#
# Usage:
#   scripts/host/run_scene_admission.sh [--categories home,commercial] [--limit N] [--all] [--render]
#       [--combined-env-phases] [--arena-probe N] [--lightwheel N] [--max-runtime-h H] [--purge]
#       [--skip-fetch|--skip-validate|--skip-upload] [--stage DIR] [--env-file FILE] [--skip-gate] [--dry-run]
#
# Expected cost is in the archived header (image ~35 GB, so >= 80 GB disk for the limited run).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=session_lib.sh
source "$REPO_ROOT/scripts/host/session_lib.sh"
cerberus_defaults

STAGE="${CERBERUS_SCENE_STAGE:-$DATA_DIR/grscenes}"
export CERBERUS_SCENES_DIR="${CERBERUS_SCENES_DIR:-$DATA_DIR/scenes}"
STATE="$STAGE/state"
CATEGORIES="home,commercial"
LIMIT=""
ALL=0; RENDER=0; PURGE=0; DRY=0; COMBINED=0; GATE=1
ARENA_PROBE=0; LIGHTWHEEL=0; MAX_RUNTIME_H=""
DO_FETCH=1; DO_VALIDATE=1; DO_UPLOAD=1
ENV_FILE="${CERBERUS_ENV_FILE:-$REPO_ROOT/.env}"
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
        --stage) STAGE="$2"; STATE="$2/state"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --skip-gate) GATE=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "[scene_admission] unknown argument: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$STATE"
SUMMARY="$STAGE/summary.txt"
FAILS=0
RC=0
run() { if [[ $DRY -eq 1 ]]; then echo "[dry-run] $*"; else "$@"; fi; }
limit_for() { if [[ -n "$LIMIT" ]]; then echo "$LIMIT"; elif [[ "$1" == home ]]; then echo 8; else echo 2; fi; }

cerberus_load_env "$ENV_FILE" || cerberus_say "no credentials file ($ENV_FILE); uploads will be skipped"
export CERBERUS_ON_POD=1
if [[ $GATE -eq 1 ]]; then
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] driver_gate.sh && launch_probe.sh"; else
        cerberus_gate || { cerberus_record gate FAIL "gate or probe rejected this host"; exit 2; }
    fi
fi

SEL_DESC="limited"; [[ $ALL -eq 1 ]] && SEL_DESC="ALL scenes"
cerberus_say "plan: $SEL_DESC, categories=$CATEGORIES, stage=$STAGE, scenes dir=$CERBERUS_SCENES_DIR"
df -h "$STAGE" 2>/dev/null | tail -n 1 | sed 's/^/[scene_admission] disk: /'

IFS=',' read -r -a CAT_LIST <<< "$CATEGORIES"
for cat in "${CAT_LIST[@]}"; do
    n="$(limit_for "$cat")"
    if [[ $ALL -eq 1 ]]; then sel=(--all); tag=all; else sel=(--scenes "$n"); tag="n$n"; fi
    stage_cat="$STAGE/$cat"

    if [[ $DO_FETCH -eq 1 && ! -f "$STATE/fetch_${cat}_$tag.done" ]]; then
        cerberus_say "[$cat] fetch ($tag)"
        if run python3 scripts/fetch_grscenes_to_r2.py --category "$cat" "${sel[@]}" --extract-only \
                --stage "$stage_cat" --link-to "$CERBERUS_SCENES_DIR/grscenes/$cat"; then
            [[ $DRY -eq 1 ]] || touch "$STATE/fetch_${cat}_$tag.done"
        else
            cerberus_say "[$cat] fetch FAILED"; RC=1; continue
        fi
    fi

    if [[ $DO_VALIDATE -eq 1 ]]; then
        cerberus_say "[$cat] validate"
        vargs=(--source grscenes --categories "$cat" --resume)
        [[ $ALL -eq 1 ]] || vargs+=(--limit "$n")
        [[ $RENDER -eq 1 ]] && vargs+=(--render)
        [[ $COMBINED -eq 1 ]] && vargs+=(--combined-env-phases)
        [[ "$ARENA_PROBE" -gt 0 ]] && vargs+=(--arena-probe "$ARENA_PROBE")
        [[ -n "$MAX_RUNTIME_H" ]] && vargs+=(--max-runtime-h "$MAX_RUNTIME_H")
        run python3 scripts/validate_scenes.py "${vargs[@]}" || { cerberus_say "[$cat] validate exited non-zero"; RC=1; }
    fi

    if [[ $DO_UPLOAD -eq 1 && ! -f "$STATE/upload_${cat}_$tag.done" ]]; then
        cerberus_say "[$cat] upload admitted scenes"
        ids="$STATE/admitted_$cat.txt"
        if [[ $DRY -eq 1 ]]; then
            echo "[dry-run] list admitted $cat scenes -> $ids"
            admitted_n=1
        else
            # Branch on the COUNT, never on file size: an empty list must be an empty file, not a lone newline.
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
            cerberus_say "[$cat] no admitted scenes; nothing to upload"
        elif [[ $DRY -eq 1 ]] || cerberus_r2 lsf "cerberus_r2:${R2_BUCKET}" --max-depth 1 >/dev/null 2>&1; then
            if run python3 scripts/fetch_grscenes_to_r2.py --category "$cat" --from-tree \
                    --scene-ids-file "$ids" --stage "$stage_cat"; then
                [[ $DRY -eq 1 ]] || touch "$STATE/upload_${cat}_$tag.done"
                if [[ $PURGE -eq 1 && $DRY -eq 0 ]]; then
                    cerberus_say "[$cat] purging $stage_cat"; rm -rf "$stage_cat"
                fi
            else
                cerberus_say "[$cat] upload FAILED"; RC=1
            fi
        else
            RC=1
        fi
    fi
done

if [[ $DO_UPLOAD -eq 1 && -f "$CERBERUS_SCENES_DIR/grscenes/admission.json" ]] \
        && { [[ $DRY -eq 1 ]] || cerberus_r2 copyto "$CERBERUS_SCENES_DIR/grscenes/admission.json" \
                "cerberus_r2:${R2_BUCKET}/worlds/grscenes/admission.json"; }; then
    :
else
    [[ $DO_UPLOAD -eq 1 && $DRY -eq 0 ]] && RC=1
fi

if [[ "$LIGHTWHEEL" -gt 0 && $DO_VALIDATE -eq 1 ]]; then
    cerberus_say "[lightwheel] validate $LIGHTWHEEL stratified kitchens"
    largs=(--source lightwheel --lightwheel-subset "$LIGHTWHEEL" --resume)
    [[ $RENDER -eq 1 ]] && largs+=(--render)
    [[ -n "$MAX_RUNTIME_H" ]] && largs+=(--max-runtime-h "$MAX_RUNTIME_H")
    run python3 scripts/validate_scenes.py "${largs[@]}" || { cerberus_say "[lightwheel] validate exited non-zero"; RC=1; }
    if [[ $DO_UPLOAD -eq 1 && -f "$CERBERUS_SCENES_DIR/lightwheel/admission.json" ]]; then
        run cerberus_r2 copyto "$CERBERUS_SCENES_DIR/lightwheel/admission.json" \
            "cerberus_r2:${R2_BUCKET}/worlds/lightwheel/admission.json" || RC=1
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
cerberus_say "finished (rc=$RC); this host is left running, stop it explicitly"
exit "$RC"
