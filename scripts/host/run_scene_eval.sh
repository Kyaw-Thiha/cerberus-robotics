#!/usr/bin/env bash
# Scene eval: fetch homes, preprocess them to the metric USD, load each with a Go2 and record what the frozen
# policy can and cannot be driven with. Re-created from `scripts/run_scene_eval_pod.sh` at tag
# archive/provider-tooling-2026-11-21: same steps, no dead-man switch and no provider calls, the report lands
# under $CERBERUS_RESULTS_DIR, and every path is an environment variable with the image layout as default.
#
# Per scene:
#   1 fetch       scripts/fetch_grscenes_from_r2.py (private mirror; needs R2_*)
#   2 preprocess  resolve the metric USD through the plain_usd backend
#   3 load+stand  200 zero-action steps with a Go2 (Kit; a scene that cannot host the robot fails here)
#   4 drive       not attempted: the plain_usd navigation env is 33-dim with no velocity command, so it cannot
#                 host the 235-dim locomotion policy (docs/reports/scene_eval_phase0.md)
#
# Usage: scripts/host/run_scene_eval.sh [--scenes id1,id2] [--num_envs 16] [--report-dir DIR] [--cap-s 3600]
#                                       [--env-file FILE] [--skip-gate] [--dry-run]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=session_lib.sh
source "$REPO_ROOT/scripts/host/session_lib.sh"
cerberus_defaults

SCENES="MV7J6NIKTKJZ2AABAAAAADQ8_usd,MV7J6NIKTKJZ2AABAAAAADY8_usd,MVUCSQAKTKJ5EAABAAAAAAI8_usd"
NUM_ENVS=16
REPORT_DIR=""
CAP_S="${SCENE_EVAL_CAP_S:-3600}"
FETCH_TIMEOUT="${SCENE_FETCH_TIMEOUT_S:-1800}"
ENV_FILE="${CERBERUS_ENV_FILE:-$REPO_ROOT/.env}"
GATE=1
DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenes) SCENES="$2"; shift 2 ;;
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --cap-s) CAP_S="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --skip-gate) GATE=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "[scene_eval] unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$REPORT_DIR" ]] || REPORT_DIR="$RESULTS_DIR/scene_eval_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
FAILS=0
START=$(date +%s)
elapsed() { echo $(( $(date +%s) - START )); }
run() { if [[ $DRY -eq 1 ]]; then echo "[dry-run] $*"; else "$@"; fi; }

cerberus_load_env "$ENV_FILE" || cerberus_say "no credentials file ($ENV_FILE)"
export CERBERUS_ON_POD=1   # unlocks the guarded fetch/preprocess scripts (scene assets are never laptop-sized)
if [[ $GATE -eq 1 ]]; then
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] driver_gate.sh && launch_probe.sh"; else
        cerberus_gate || { cerberus_record gate FAIL "gate or probe rejected this host"; exit 2; }
    fi
fi

IFS=',' read -r -a SCENE_LIST <<< "$SCENES"
for scene in "${SCENE_LIST[@]}"; do
    scene="$(echo "$scene" | xargs)"
    [[ -z "$scene" ]] && continue
    if (( $(elapsed) > CAP_S )); then
        cerberus_say "cap reached ($(elapsed)s > ${CAP_S}s): not starting $scene"
        cerberus_record "$scene" SKIP cap
        continue
    fi

    cerberus_say "fetch $scene"
    if [[ $DRY -eq 1 ]]; then
        echo "[dry-run] fetch_grscenes_from_r2.py --scene-id $scene --dest $SCENES_DIR"
    else
        if ( cd "$REPO_ROOT" && timeout "$FETCH_TIMEOUT" python3 scripts/fetch_grscenes_from_r2.py \
                --scene-id "$scene" --dest "$SCENES_DIR" > "$REPORT_DIR/${scene}_fetch.log" 2>&1 ); then
            cerberus_record "${scene}_fetch" PASS -
        else
            cerberus_record "${scene}_fetch" FAIL "see ${scene}_fetch.log"
            continue
        fi
    fi

    cerberus_say "preprocess $scene"
    if [[ $DRY -eq 1 ]]; then
        echo "[dry-run] resolve metric USD for grscenes:$scene"
    elif ( cd "$REPO_ROOT" && timeout 300 python3 - "$scene" "$REPORT_DIR/${scene}_preprocess.json" <<'PY'
import json, sys
from common.cerberus_worlds.registry import get_backend

scene_id, out = sys.argv[1], sys.argv[2]
backend = get_backend("plain_usd")
payload = {"describe": backend.describe_scene(f"grscenes:{scene_id}"),
           "loaded": backend.load_scene(f"grscenes:{scene_id}")}
open(out, "w").write(json.dumps(payload, indent=2, default=str) + "\n")
PY
    ); then
        cerberus_record "${scene}_preprocess" PASS "$REPORT_DIR/${scene}_preprocess.json"
    else
        cerberus_record "${scene}_preprocess" FAIL "scene USD could not be resolved (see console)"
        continue
    fi

    cerberus_say "load + stand $scene"
    if [[ $DRY -eq 1 ]]; then
        echo "[dry-run] arena_spike_go2.py --backend plain_usd --scene grscenes:$scene --num_envs $NUM_ENVS"
    elif ( cd "$REPO_ROOT" && timeout 900 bash scripts/host/kit_retry.sh -- "$LAB" -p scripts/arena_spike_go2.py \
            --backend plain_usd --scene "grscenes:$scene" --num_envs "$NUM_ENVS" --steps 200 --headless \
            --report "$REPORT_DIR/${scene}_spike.json" > "$REPORT_DIR/${scene}_spike.log" 2>&1 ); then
        cerberus_record "${scene}_load" PASS "$REPORT_DIR/${scene}_spike.json"
    else
        cerberus_record "${scene}_load" FAIL "see ${scene}_spike.log"
    fi
done

cerberus_record drive SKIP "plain_usd navigation env (33-dim, no velocity command) cannot host the 235-dim locomotion policy"
cerberus_say "total GPU time: $(elapsed)s (cap ${CAP_S}s)"
cerberus_say "done: $FAILS failed step(s). Summary: $SUMMARY"
exit "$FAILS"
