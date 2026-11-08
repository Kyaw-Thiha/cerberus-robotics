#!/usr/bin/env bash
# Scene eval (non-blocking): fetch homes, run the plain_usd preprocess, load them, and record what the
# frozen locomotion policy can and cannot be driven with.
#
# What this does, in order, per scene:
#   1 fetch    scripts/fetch_grscenes_from_r2.py pulls the scene + shared assets from the private R2 mirror
#   2 preprocess `load_scene` resolves the metric USD (GRScenes are centimetre-authored; Isaac does not
#              convert, so a cached metre-scale wrapper is generated beside the raw USD)
#   3 load     scripts/arena_spike_go2.py --backend plain_usd builds the env, spawns a Go2 and steps it with
#              zero actions; a scene that cannot host the robot fails here
#   4 drive    NOT ATTEMPTED: see the note printed at the end and docs/reports/scene_eval_phase0.md. The
#              plain_usd env exposes a navigation observation space (33 dims, no velocity command) and the
#              candidate needs 235 dims including a 187-cell height scan; composing the locomotion task into
#              a USD scene is a task package, not a sanity check.
#
# Cap: SCENE_EVAL_CAP_S (default 3600) of GPU time; the loop stops starting new scenes past it.
#
# Usage (on the pod): scripts/run_scene_eval_pod.sh [--scenes id1,id2,id3] [--num_envs 16] [--report-dir DIR]
set -uo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
if [[ -f /etc/cerberus_env.sh ]]; then source /etc/cerberus_env.sh; fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CERBERUS_ALLOW_BAD_DRIVER_RENDER=1
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
LAB="$ISAACLAB_PATH/isaaclab.sh"
CAP_S="${SCENE_EVAL_CAP_S:-3600}"

SCENES="MV7J6NIKTKJZ2AABAAAAADQ8_usd,MV7J6NIKTKJZ2AABAAAAADY8_usd,MVUCSQAKTKJ5EAABAAAAAAI8_usd"
NUM_ENVS=16
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="data/reports/scene_eval_$STAMP"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenes) SCENES="$2"; shift 2 ;;
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        *) echo "[scene] unknown argument: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
START=$(date +%s)
FAILS=0
say() { echo "[scene $(date -u +%H:%M:%S)] $*"; }
record() { echo "$1 $2 $3" | tee -a "$SUMMARY"; [[ "$2" == PASS || "$2" == SKIP ]] || FAILS=$((FAILS + 1)); }
elapsed() { echo $(( $(date +%s) - START )); }

IFS=',' read -r -a SCENE_LIST <<< "$SCENES"
for scene in "${SCENE_LIST[@]}"; do
    scene="$(echo "$scene" | xargs)"
    [[ -z "$scene" ]] && continue
    if (( $(elapsed) > CAP_S )); then
        say "cap reached ($(elapsed)s > ${CAP_S}s): not starting $scene"
        record "$scene" SKIP "cap"
        continue
    fi

    say "fetch $scene"
    if ! timeout 600 python3 scripts/fetch_grscenes_from_r2.py --scene-id "$scene" > "$REPORT_DIR/${scene}_fetch.log" 2>&1; then
        record "${scene}_fetch" FAIL "see ${scene}_fetch.log"
        continue
    fi
    record "${scene}_fetch" PASS "-"

    # Preprocess: resolve the metric USD (pure python + a pxr child). Record the resolved paths.
    if timeout 300 python3 - "$scene" "$REPORT_DIR/${scene}_preprocess.json" <<'PY'
import json, sys
from common.cerberus_worlds.registry import get_backend

scene_id, out = sys.argv[1], sys.argv[2]
backend = get_backend("plain_usd")
description = backend.describe_scene(f"grscenes:{scene_id}")
loaded = backend.load_scene(f"grscenes:{scene_id}")
payload = {"describe": description, "loaded": loaded}
open(out, "w").write(json.dumps(payload, indent=2, default=str) + "\n")
print(json.dumps({"usd_path": loaded.get("usd_path"), "variant": loaded.get("variant")}, indent=1))
PY
    then
        record "${scene}_preprocess" PASS "$REPORT_DIR/${scene}_preprocess.json"
    else
        record "${scene}_preprocess" FAIL "see console (scene USD could not be resolved)"
        continue
    fi

    say "load + stand $scene"
    if timeout 900 "$LAB" -p scripts/arena_spike_go2.py \
        --backend plain_usd --scene "grscenes:$scene" --num_envs "$NUM_ENVS" --steps 200 --headless \
        --report "$REPORT_DIR/${scene}_spike.json" > "$REPORT_DIR/${scene}_spike.log" 2>&1; then
        record "${scene}_load" PASS "$REPORT_DIR/${scene}_spike.json"
    else
        record "${scene}_load" FAIL "see ${scene}_spike.log"
    fi
done

record drive SKIP "plain_usd navigation env (33-dim obs, no velocity command) cannot host the 235-dim locomotion policy; see docs/reports/scene_eval_phase0.md"
say "total GPU time: $(elapsed)s (cap ${CAP_S}s)"
say "done: $FAILS failed step(s). Summary: $SUMMARY"
exit "$FAILS"
