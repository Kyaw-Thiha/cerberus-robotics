#!/usr/bin/env bash
# Bootstrap a host for the stack: caches, image, scene, and the checks that make the next step worth running.
# Idempotent: every step is skipped when its result is already present, so a re-run after a restart resumes.
#
#   1. disk      free space at the data dir (fails before a 35 GB pull when there is no room)
#   2. caches    persistent directories for the Kit/shader, pip and Hugging Face caches (bind-mount targets)
#   3. image     `docker pull` of the remote image, with retry; skipped when the tag is already local
#   4. scene     one GRScenes home fetched into $CERBERUS_SCENES_DIR, skipped when already there
#   5. checkpoint the frozen GA candidate shipped in the repo (small; fails fast when missing)
#
# Usage:
#   scripts/host/bootstrap_host.sh [--data-dir DIR] [--image IMG] [--scene ID] [--env-file FILE]
#                                  [--min-disk-gb N] [--pull] [--skip image|scene|checkpoint] [--dry-run]
#
# The image and the checkpoint are public; the scene comes from the private bucket and needs R2_* credentials
# (from --env-file or the environment). Nothing secret is printed.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=session_lib.sh
source "$REPO_ROOT/scripts/host/session_lib.sh"

DATA_DIR_OVERRIDE=""
IMAGE="${CERBERUS_IMAGE:-kevinkyawthiha/cerberus-remote-v6:v6-ga}"
SCENE="${CERBERUS_SCENE:-MV7J6NIKTKJZ2AABAAAAADQ8_usd}"
ENV_FILE="${CERBERUS_ENV_FILE:-$REPO_ROOT/.env}"
MIN_DISK_GB="100"
PULL=0
DRY=0
SKIP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR_OVERRIDE="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --scene) SCENE="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --min-disk-gb) MIN_DISK_GB="$2"; shift 2 ;;
        --pull) PULL=1; shift ;;
        --skip) SKIP="$SKIP ${2//,/ }"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "bootstrap_host: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ -n "$DATA_DIR_OVERRIDE" ]] && CERBERUS_DATA_DIR="$DATA_DIR_OVERRIDE"
cerberus_defaults
mkdir -p "$RESULTS_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="$RESULTS_DIR/bootstrap-$STAMP"
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
FAILS=0
skipped() { [[ " $SKIP " == *" $1 "* ]]; }
run() { if [[ $DRY -eq 1 ]]; then echo "[dry-run] $*"; else "$@"; fi; }

cerberus_say "data dir $DATA_DIR, results $RESULTS_DIR, image $IMAGE"

# ---- 1. disk ----------------------------------------------------------------------------------------------------
if skipped disk; then cerberus_record disk SKIP -; else
    free_gb="$(python3 - "$DATA_DIR" <<'PY'
import shutil, sys
from pathlib import Path
p = Path(sys.argv[1])
while not p.exists() and p != p.parent:
    p = p.parent
print(f"{shutil.disk_usage(p).free / 1e9:.1f}")
PY
)"
    if python3 -c "import sys; sys.exit(0 if float('$free_gb') >= float('$MIN_DISK_GB') else 1)"; then
        cerberus_record disk PASS "${free_gb}GB free at $DATA_DIR"
    else
        cerberus_record disk FAIL "${free_gb}GB free, need ${MIN_DISK_GB}GB (pass --data-dir on a bigger volume)"
    fi
fi

# ---- 2. caches --------------------------------------------------------------------------------------------------
if skipped caches; then cerberus_record caches SKIP -; else
    for d in kit pip hf shader; do run mkdir -p "$DATA_DIR/cache/$d"; done
    cat > "$REPORT_DIR/cache_mounts.txt" <<EOF
# Bind mounts that survive a container restart (pass these to docker run -v):
$DATA_DIR/cache/kit:/isaac-sim/kit/cache
$DATA_DIR/cache/shader:/isaac-sim/.cache
$DATA_DIR/cache/pip:/root/.cache/pip
$DATA_DIR/cache/hf:/root/.cache/huggingface
EOF
    cerberus_record caches PASS "$DATA_DIR/cache (see cache_mounts.txt)"
fi

# ---- 3. image ---------------------------------------------------------------------------------------------------
if skipped image; then cerberus_record image SKIP -; else
    if [[ $PULL -eq 0 ]] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
        cerberus_record image PASS "$IMAGE already local"
    else
        pulled=0
        for attempt in 1 2 3; do
            cerberus_say "docker pull $IMAGE (attempt $attempt/3)"
            if run docker pull "$IMAGE" >> "$REPORT_DIR/image_pull.log" 2>&1; then pulled=1; break; fi
            sleep $((attempt * 10))
        done
        if [[ $pulled -eq 1 ]]; then cerberus_record image PASS "$IMAGE"; else
            cerberus_record image FAIL "docker pull failed 3x; see $REPORT_DIR/image_pull.log (re-run to resume)"
        fi
    fi
fi

# ---- 4. scene ---------------------------------------------------------------------------------------------------
if skipped scene; then cerberus_record scene SKIP -; else
    if [[ -d "$SCENES_DIR/scenes/${SCENE}_usd" ]] && [[ -s "$SCENES_DIR/manifest.json" ]]; then
        cerberus_record scene PASS "$SCENE already at $SCENES_DIR"
    else
        if cerberus_load_env "$ENV_FILE"; then
            cerberus_say "fetching scene $SCENE into $SCENES_DIR"
            if run env CERBERUS_ON_POD=1 python3 "$REPO_ROOT/scripts/fetch_grscenes_from_r2.py" \
                    --scene-id "$SCENE" --dest "$SCENES_DIR" >> "$REPORT_DIR/scene_fetch.log" 2>&1; then
                cerberus_record scene PASS "$SCENES_DIR"
            else
                cerberus_record scene FAIL "fetch failed; see $REPORT_DIR/scene_fetch.log (re-run resumes)"
            fi
        else
            cerberus_record scene FAIL "no credentials file ($ENV_FILE); pass --env-file with R2_* set"
        fi
    fi
fi

# ---- 5. checkpoint ----------------------------------------------------------------------------------------------
if skipped checkpoint; then cerberus_record checkpoint SKIP -; else
    ckpt="policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt"
    if [[ -s "$ckpt" ]]; then
        cerberus_record checkpoint PASS "$ckpt"
    else
        cerberus_record checkpoint FAIL "$ckpt missing; it is tracked in git, so check out the full repo"
    fi
fi

cerberus_say "done: $FAILS failed step(s). Summary: $SUMMARY"
exit "$FAILS"
