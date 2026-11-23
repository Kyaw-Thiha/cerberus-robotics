#!/usr/bin/env bash
# Run the scene plan on any host that passed host_doctor, one stage at a time, each relaunched after a crash.
#
# Stages (default: all):
#   stand   1 env, 200 zero-command steps, contract assertions              -> assertions.json
#   demo    cameras on, scripted path through the slew limiter (~90 s)      -> clips/*.mp4 + metrics JSON
#   smoke   scripted metrics only, cameras off                             -> metrics JSON
#   nav     A* + pure pursuit through the frozen policy, N seeded pairs    -> summary.json
#   replay  one success and one failure pair with cameras                  -> clips/*.mp4 + replay.json
#
# Every stage's verdict comes from its artifact, not from an exit code: Isaac's SimulationApp exits 0 whatever
# happened, so a stage "passes" only when its files are present and parse. Stages run through
# scripts/host/run_resilient.sh, which relaunches a crashed stage and skips finished work on re-run.
#
# Usage:
#   scripts/host/run_session.sh [--scene ID] [--pairs 10] [--stages stand,demo,smoke,nav,replay]
#                               [--out-dir DIR] [--env-file FILE] [--skip-gate] [--shutdown-after] [--dry-run]
#
# --shutdown-after powers the machine off after a run; it is refused unless this really is a VM (or --force-shutdown
# is given), because a laptop or a shared workstation must not be switched off by a script.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=session_lib.sh
source "$REPO_ROOT/scripts/host/session_lib.sh"

SCENE="${CERBERUS_SCENE:-MV7J6NIKTKJZ2AABAAAAADQ8_usd}"
PAIRS="${CERBERUS_NAV_PAIRS:-10}"
STAGES="${CERBERUS_STAGES:-stand,demo,smoke,nav,replay}"
OUT_DIR=""
ENV_FILE="${CERBERUS_ENV_FILE:-$REPO_ROOT/.env}"
SKIP_GATE=0
SHUTDOWN=0
FORCE_SHUTDOWN=0
DRY=0
INTERNAL_STAGE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --pairs) PAIRS="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --skip-gate) SKIP_GATE=1; shift ;;
        --shutdown-after) SHUTDOWN=1; shift ;;
        --force-shutdown) SHUTDOWN=1; FORCE_SHUTDOWN=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --internal-stage) INTERNAL_STAGE="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "run_session: unknown argument $1" >&2; exit 2 ;;
    esac
done
cerberus_defaults
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
[[ -n "$OUT_DIR" ]] || OUT_DIR="$RESULTS_DIR/session-$STAMP"
REPORT_DIR="$OUT_DIR"
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
FAILS=0
wanted() { [[ ",$STAGES," == *",$1,"* ]]; }

cerberus_load_env "$ENV_FILE" || cerberus_say "no credentials file ($ENV_FILE); stages that need the bucket will say so"
export CERBERUS_ON_POD=1   # scene assets are host-only, never a laptop checkout (docs/worlds.md)
export CERBERUS_ALLOW_BAD_DRIVER_RENDER="${CERBERUS_ALLOW_BAD_DRIVER_RENDER:-1}"
export CERBERUS_SKIP_MDL_PATH="${CERBERUS_SKIP_MDL_PATH:-1}"
# This Lab build does not print "Simulation App Startup Complete"; the env build is the real start marker.
export CERBERUS_STARTED_MARKER="${CERBERUS_STARTED_MARKER:-[scene_env] build timings}"
export CERBERUS_MAX_LAUNCHES="${CERBERUS_MAX_LAUNCHES:-30}"
cerberus_say "scene $SCENE, stages $STAGES, out $OUT_DIR"

if [[ $SKIP_GATE -eq 0 ]]; then
    if [[ $DRY -eq 1 ]]; then
        echo "[dry-run] driver_gate.sh && launch_probe.sh"
    else
        cerberus_gate || { cerberus_record gate FAIL "see above"; cerberus_say "stopping: a host that fails the gate will not run Kit"; exit 2; }
        cerberus_record gate PASS -
    fi
fi

lab_py() { if [[ -x "$LAB" ]]; then echo "bash $REPO_ROOT/scripts/host/kit_retry.sh -- $LAB -p"; else echo "$PY"; fi; }
# shellcheck disable=SC2046
isaac() { if [[ -x "$LAB" ]]; then bash "$REPO_ROOT/scripts/host/kit_retry.sh" -- "$LAB" -p "$@"; else "$PY" "$@"; fi; }

stage_stand() {
    local out="$REPORT_DIR/stand"
    isaac "$REPO_ROOT/scripts/scene_eval.py" --scene "grscenes:$SCENE" --num_envs 1 --steps 200 \
        --no-video --out-dir "$out"
}
stage_demo() {
    local out="$REPORT_DIR/demo"
    isaac "$REPO_ROOT/scripts/scene_eval.py" --scene "grscenes:$SCENE" --num_envs 1 --steps 200 \
        --scripted --record-clip --video --video-steps 10000 --video-fps 50 --video-downscale 1 \
        --demo-seconds 90 --demo-speed 0.3 --video-resolution 1920x1080 \
        --clip-name demo_scene1.mp4 --out-dir "$out"
}
stage_smoke() {
    local out="$REPORT_DIR/smoke"
    isaac "$REPO_ROOT/scripts/scene_eval.py" --scene "grscenes:$SCENE" --num_envs 1 --steps 200 \
        --scripted --no-video --out-dir "$out"
}
stage_nav() {
    local out="$REPORT_DIR/nav"
    isaac "$REPO_ROOT/scripts/nav_eval.py" --scenes "grscenes:$SCENE" --pairs "$PAIRS" --num_envs 4 \
        --no-video --out-dir "$out"
}
stage_replay() {
    local out="$REPORT_DIR/replay"
    isaac "$REPO_ROOT/scripts/nav_eval.py" --scenes "grscenes:$SCENE" --pairs "$PAIRS" --num_envs 4 \
        --replay-only --replay-outcomes success,failure --video --out-dir "$out"
}

# artifact_ok <stage> <dir>: the stage is judged on what it wrote.
artifact_ok() {
    case "$1" in
        stand) [[ -s "$2/assertions.json" ]] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('passed') is not False else 1)" "$2/assertions.json" ;;
        demo) compgen -G "$2/clips/*.mp4" >/dev/null ;;
        smoke) compgen -G "$2/metrics*.json" >/dev/null || compgen -G "$2/*.json" >/dev/null ;;
        nav) [[ -s "$2/summary.json" ]] ;;
        replay) [[ -s "$2/replay.json" ]] || compgen -G "$2/clips/*.mp4" >/dev/null ;;
        *) return 1 ;;
    esac
}

# A resilient relaunch runs one stage only and reports its exit code; the parent judges the artifact.
if [[ -n "$INTERNAL_STAGE" ]]; then
    cerberus_load_env "$ENV_FILE" || true
    export CERBERUS_ON_POD=1
    export CERBERUS_ALLOW_BAD_DRIVER_RENDER="${CERBERUS_ALLOW_BAD_DRIVER_RENDER:-1}"
export CERBERUS_SKIP_MDL_PATH="${CERBERUS_SKIP_MDL_PATH:-1}"
# This Lab build does not print "Simulation App Startup Complete"; the env build is the real start marker.
export CERBERUS_STARTED_MARKER="${CERBERUS_STARTED_MARKER:-[scene_env] build timings}"
export CERBERUS_MAX_LAUNCHES="${CERBERUS_MAX_LAUNCHES:-30}"
    "stage_$INTERNAL_STAGE"
    exit $?
fi
for stage in stand demo smoke nav replay; do
    wanted "$stage" || continue
    if [[ $DRY -eq 1 ]]; then
        echo "[dry-run] stage $stage -> $REPORT_DIR/$stage"
        continue
    fi
    start=$(date +%s)
    bash "$REPO_ROOT/scripts/host/run_resilient.sh" --name "$stage" --max-restarts 2 -- \
        bash "$0" --internal-stage "$stage" --scene "$SCENE" --pairs "$PAIRS" --out-dir "$REPORT_DIR" \
        --env-file "$ENV_FILE" --skip-gate >> "$REPORT_DIR/$stage.log" 2>&1
    rc=$?
    if [[ $rc -eq 0 ]] && artifact_ok "$stage" "$REPORT_DIR/$stage"; then
        cerberus_record "$stage" PASS "$(( $(date +%s) - start ))s $REPORT_DIR/$stage"
    else
        cerberus_record "$stage" FAIL "rc=$rc artifact missing or failing; see $REPORT_DIR/$stage.log"
    fi
done

cerberus_say "done: $FAILS failed stage(s). Summary: $SUMMARY"

if [[ $SHUTDOWN -eq 1 ]]; then
    if [[ $FORCE_SHUTDOWN -eq 1 ]] || systemd-detect-virt --quiet 2>/dev/null; then
        cerberus_say "shutting this machine down (VM or --force-shutdown)"
        [[ $DRY -eq 1 ]] || sudo shutdown -h +1 "cerberus run_session finished"
    else
        cerberus_say "refusing --shutdown-after: this is not a VM (pass --force-shutdown to override)"
    fi
fi
exit "$FAILS"
