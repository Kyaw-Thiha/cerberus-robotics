#!/usr/bin/env bash
# One SLAM evaluation mission: backend adapter + ROS2 relay + Isaac Lab sim loop together, the mission recorded,
# scored with evo, synced to the bucket, then an optional caller-supplied stop command. Re-created from
# `scripts/run_slam_eval_remote.sh` at tag archive/provider-tooling-2026-11-21. Changed: the pod gate and the
# pod dead-man switch are gone (the host gate is shared), the image paths are environment variables
# (SIM_ENV, COLCON_WS, ISAACLAB_PATH) instead of hardcoded /pixi-project, and the provider self-terminate is
# replaced by --stop-when-done '<command>', which is what a Brev or cloud user actually wants to run.
#
# Prerequisites (checked in seconds before any node starts): the sim env's activation script, and for a real
# backend also its evo CLIs, activation script and colcon workspace, all from the image and the extracted
# backend artifact (scripts/extract_slam_backend_artifact.sh).
#
# Usage:
#   scripts/host/run_slam_eval.sh --backend <name> --mission <name> --sensor-rig <rig> --checkpoint <path> \
#       --timeout <duration> [--scene stock|grscenes:<id>] [--spawn-index N | --spawn x,y,yaw,floor_z] \
#       [--stop-when-done '<cmd>'] [--no-sync] [--skip-gate] [--dry-run]
#   scripts/host/run_slam_eval.sh --smoke --mission household --sensor-rig lidar_camera \
#       --checkpoint policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.onnx --timeout 30m --dry-run
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=session_lib.sh
source "$REPO_ROOT/scripts/host/session_lib.sh"
cerberus_defaults

SIM_ENV="${CERBERUS_SIM_ENV:-/pixi-project}"
COLCON_WS="${CERBERUS_COLCON_WS:-/colcon_ws}"
SIM_ROS2_BIN="$SIM_ENV/.pixi/envs/sim-ros2/bin"
BACKEND=""; MISSION=""; SENSOR_RIG=""; CHECKPOINT=""; SCENE="stock"
SPAWN_INDEX=""; SPAWN=""; JOB_TIMEOUT=""; STOP_CMD=""; DO_SYNC=1; SMOKE=0; GATE=1; DRY=0
ENV_FILE="${CERBERUS_ENV_FILE:-$REPO_ROOT/.env}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        --mission) MISSION="$2"; shift 2 ;;
        --sensor-rig) SENSOR_RIG="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --scene) SCENE="$2"; shift 2 ;;
        --spawn-index) SPAWN_INDEX="$2"; shift 2 ;;
        --spawn) SPAWN="$2"; shift 2 ;;
        --timeout) JOB_TIMEOUT="$2"; shift 2 ;;
        --stop-when-done) STOP_CMD="$2"; shift 2 ;;
        --no-sync) DO_SYNC=0; shift ;;
        --smoke) SMOKE=1; shift ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --skip-gate) GATE=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "[run_slam_eval] unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ "$SMOKE" == "1" && -n "$BACKEND" ]]; then
    echo "[run_slam_eval] --smoke runs without a backend; drop --backend $BACKEND" >&2; exit 2
fi
[[ "$SMOKE" == "1" ]] && BACKEND="smoke"
if [[ -z "$BACKEND" || -z "$MISSION" || -z "$SENSOR_RIG" || -z "$CHECKPOINT" || -z "$JOB_TIMEOUT" ]]; then
    echo "usage: $0 --backend <name> --mission <name> --sensor-rig <rig> --checkpoint <path> --timeout <m|s> ..." >&2
    exit 2
fi

cerberus_load_env "$ENV_FILE" || true
export CERBERUS_ON_POD=1
if [[ $GATE -eq 1 ]]; then
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] driver_gate.sh && launch_probe.sh"; else
        cerberus_gate || { echo "[run_slam_eval] gate or probe rejected this host" >&2; exit 2; }
    fi
fi

RUN_TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$RESULTS_DIR/slam/$BACKEND/$RUN_TIMESTAMP"
BAG_PATH="$RUN_DIR/bag"
SIM_DONE_FILE="$RUN_DIR/.sim_exited"
cerberus_say "backend=$BACKEND mission=$MISSION rig=$SENSOR_RIG scene=$SCENE run=$RUN_TIMESTAMP"

SCENE_ARGS=(--scene "$SCENE")
[[ -n "$SPAWN_INDEX" ]] && SCENE_ARGS+=(--spawn-index "$SPAWN_INDEX")
# --spawn=VALUE, not two words: a negative x would be read as a flag.
[[ -n "$SPAWN" ]] && SCENE_ARGS+=("--spawn=$SPAWN")

PREFLIGHT_OK=1
# require_file <path> <what-it-is-for>; a dry run only reports what the real run would need.
require_file() {
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] requires $1 ($2)"; return 0; fi
    if [[ ! -f "$1" ]]; then echo "[run_slam_eval] ERROR: missing $1 ($2)" >&2; PREFLIGHT_OK=0; fi
}
require_file "$SIM_ENV/activate.sh" "sim-ros2 activation script, from the sim image"
if [[ "$SMOKE" != "1" ]]; then
    for cli in evo_ape evo_rpe; do
        if [[ $DRY -eq 1 ]]; then echo "[dry-run] requires $SIM_ROS2_BIN/$cli (evo, from the sim image)"
        elif [[ ! -x "$SIM_ROS2_BIN/$cli" ]]; then
            echo "[run_slam_eval] ERROR: missing $SIM_ROS2_BIN/$cli (evo, from the sim image)" >&2; PREFLIGHT_OK=0
        fi
    done
    require_file "$SIM_ENV/activate-${BACKEND}.sh" "backend activation script (extract_slam_backend_artifact.sh)"
    require_file "$COLCON_WS/install/setup.bash" "backend colcon workspace, same artifact"
fi
require_file "$CHECKPOINT" "--checkpoint (relative paths resolve from the repo root)"
if [[ $DRY -eq 1 ]]; then echo "[dry-run] requires isaaclab.sh at $LAB"
elif [[ ! -x "$LAB" ]]; then echo "[run_slam_eval] ERROR: no isaaclab.sh at $LAB (set ISAACLAB_PATH)" >&2; PREFLIGHT_OK=0; fi
if [[ "$PREFLIGHT_OK" != "1" ]]; then exit 1; fi

if [[ $DRY -eq 1 ]]; then
    echo "[dry-run] run dir $RUN_DIR"
    echo "[dry-run] backend=$BACKEND mission=$MISSION scene=$SCENE rig=$SENSOR_RIG timeout=$JOB_TIMEOUT"
    echo "[dry-run] sim loop: isaaclab.sh -p slam/slam/ground_truth/sim_ground_truth_loop.py ${SCENE_ARGS[*]}"
    [[ -n "$STOP_CMD" ]] && echo "[dry-run] stop when done: $STOP_CMD"
    exit 0
fi
mkdir -p "$RUN_DIR"

# 0. LiDAR proxy: the raw GRScenes house has empty Mesh prims, so the LiDAR casts against a cached proxy mesh.
if [[ "$SCENE" != "stock" ]]; then
    echo "[run_slam_eval] building/validating the LiDAR proxy mesh for $SCENE..."
    bash "$REPO_ROOT/scripts/host/kit_retry.sh" -- "$LAB" -p scripts/build_lidar_proxy.py --scene "$SCENE" || {
        echo "[run_slam_eval] ERROR: LiDAR proxy build failed for $SCENE" >&2; exit 1; }
fi

PIDS=()
ALGORITHM_SCRIPT="$REPO_ROOT/slam/slam/backends/${BACKEND}/launch_algorithm.sh"
if [[ "$SMOKE" == "1" ]]; then
    echo "[run_slam_eval] --smoke: no backend algorithm, no scoring, no sync."
elif [[ -f "$ALGORITHM_SCRIPT" ]]; then
    ( cd "$REPO_ROOT" && exec bash -c "source '$SIM_ENV/activate-${BACKEND}.sh' && source '$COLCON_WS/install/setup.bash' && bash '$ALGORITHM_SCRIPT'" ) &
    PIDS+=($!)
else
    echo "[run_slam_eval] WARNING: no $ALGORITHM_SCRIPT; running the adapter alone" >&2
fi
if [[ "$SMOKE" != "1" ]]; then
    ( cd "$REPO_ROOT" && exec bash -c "source '$SIM_ENV/activate-${BACKEND}.sh' && source '$COLCON_WS/install/setup.bash' && python3 -m slam.slam.backends.${BACKEND}.adapter_launch" ) &
    PIDS+=($!)
fi
( cd "$REPO_ROOT" && exec bash -c "source '$SIM_ENV/activate.sh' && python3 -m slam.slam.ground_truth.ros2_bridge_relay" ) &
RELAY_PID=$!
PIDS+=("$RELAY_PID")
sleep 3

RECORD_EXTRA=""
[[ "$SMOKE" == "1" ]] && RECORD_EXTRA="--smoke"
( cd "$REPO_ROOT" && exec bash -c "source '$SIM_ENV/activate.sh' && python3 slam/slam/eval/record_mission_bag.py \
    --backend '$BACKEND' --mission '$MISSION' --run-timestamp '$RUN_TIMESTAMP' --sensor-rig '$SENSOR_RIG' \
    --sim-done-file '$SIM_DONE_FILE' $RECORD_EXTRA" ) &
BAG_PID=$!

( cd "$REPO_ROOT" && timeout "$JOB_TIMEOUT" bash "$REPO_ROOT/scripts/host/kit_retry.sh" -- "$LAB" -p \
    slam/slam/ground_truth/sim_ground_truth_loop.py \
    --mission "$MISSION" --sensor-rig "$SENSOR_RIG" --checkpoint "$CHECKPOINT" "${SCENE_ARGS[@]}" \
    --backend "$BACKEND" --run-timestamp "$RUN_TIMESTAMP" --headless )
SIM_EXIT_CODE=$?
echo "[run_slam_eval] sim loop exited with code $SIM_EXIT_CODE"

mkdir -p "$RUN_DIR" && touch "$SIM_DONE_FILE"
wait "$BAG_PID"; BAG_EXIT_CODE=$?
kill "${PIDS[@]}" 2>/dev/null
wait "${PIDS[@]}" 2>/dev/null

if [[ "$SMOKE" == "1" ]]; then
    SMOKE_EXIT_CODE=0
    [[ "$SIM_EXIT_CODE" == "0" ]] || { echo "[smoke] FAIL sim loop exit $SIM_EXIT_CODE" >&2; SMOKE_EXIT_CODE=1; }
    [[ "$BAG_EXIT_CODE" == "0" ]] || { echo "[smoke] FAIL recorder exit $BAG_EXIT_CODE" >&2; SMOKE_EXIT_CODE=1; }
    "$SIM_ROS2_BIN/python3" "$REPO_ROOT/slam/slam/eval/smoke_check.py" finalize --run-dir "$RUN_DIR" \
        --expect-scene "$SCENE" || SMOKE_EXIT_CODE=1
    echo "[run_slam_eval] smoke exit $SMOKE_EXIT_CODE; nothing synced, host left running."
    exit "$SMOKE_EXIT_CODE"
fi

if [[ "$SIM_EXIT_CODE" != "0" ]]; then
    echo "[run_slam_eval] ERROR: sim loop failed ($SIM_EXIT_CODE) -- see $RUN_DIR/sim_loop_report.json; not scoring" >&2
    [[ $DO_SYNC -eq 1 ]] && cerberus_sync "$RUN_DIR" "slam/$BACKEND/$RUN_TIMESTAMP" || true
    exit "$SIM_EXIT_CODE"
fi
if [[ "$BAG_EXIT_CODE" != "0" ]]; then
    echo "[run_slam_eval] ERROR: recorder failed ($BAG_EXIT_CODE); not scoring" >&2
    exit "$BAG_EXIT_CODE"
fi

# Never score a run that collided: the schedule is open loop, so the numbers would measure the collision.
SCORING_VERDICT=$(python3 - "$RUN_DIR/sim_loop_report.json" <<'PYEOF'
import json, sys
try:
    block = json.load(open(sys.argv[1])).get("collision") or {}
except (OSError, ValueError) as e:
    print(f"unreadable:{type(e).__name__}")
else:
    reasons = block.get("invalid_reasons") or ["no collision block in the report"]
    print("yes" if block.get("valid_for_scoring") else "no:" + ",".join(reasons))
PYEOF
)
if [[ "$SCORING_VERDICT" != "yes" ]]; then
    echo "[run_slam_eval] ERROR: collision check did not pass ($SCORING_VERDICT); not scoring" >&2
    [[ $DO_SYNC -eq 1 ]] && cerberus_sync "$RUN_DIR" "slam/$BACKEND/$RUN_TIMESTAMP" || true
    exit 3
fi

( cd "$REPO_ROOT" && PATH="$SIM_ROS2_BIN:$PATH" "$SIM_ROS2_BIN/python3" - "$BAG_PATH" "$BACKEND" <<'PYEOF'
import sys
from slam.slam.eval.score_with_evo import run_evo_scoring
run_evo_scoring(sys.argv[1], sys.argv[2])
PYEOF
)
SCORE_EXIT_CODE=$?
echo "[run_slam_eval] scoring exited with code $SCORE_EXIT_CODE"

if [[ $DO_SYNC -eq 1 ]]; then
    echo "[run_slam_eval] syncing $RUN_DIR ..."
    if ! cerberus_sync "$RUN_DIR" "slam/$BACKEND/$RUN_TIMESTAMP"; then
        echo "[run_slam_eval] ERROR: sync failed; not running the stop command. Sync or investigate manually." >&2
        exit "$SCORE_EXIT_CODE"
    fi
fi

if [[ -n "$STOP_CMD" ]]; then
    echo "[run_slam_eval] running stop command: $STOP_CMD"
    bash -lc "$STOP_CMD" || echo "[run_slam_eval] WARNING: stop command failed; stop the host yourself" >&2
fi
exit "$SCORE_EXIT_CODE"
