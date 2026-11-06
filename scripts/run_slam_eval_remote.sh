#!/usr/bin/env bash
# Runs ON THE POD. Orchestrates one remote SLAM eval mission end-to-end:
# backend adapter + ros2_bridge_relay + Isaac Lab sim loop launched
# together, record_mission_bag.py records the mission, score_with_evo.py
# scores it, results sync to R2, pod self-terminates on success. Mirrors
# scripts/run_remote_job.sh's safety properties (timeout, sync regardless
# of outcome, self-terminate only on success) but is SLAM-eval-shaped
# rather than reusing that script directly -- its find_run_dir() assumes a
# training-shaped single-checkpoints-dir contract that doesn't cleanly fit
# slam/slam/eval/results/<backend>/<run_timestamp>/.
#
# Prerequisites this script assumes are ALREADY done before it runs (checked
# up front, see "Preflight" below):
#   - Cerberus code synced to this pod's checkout (tar-over-SSH, per
#     docs/runpod.md), this script run from that checkout's repo root.
#   - The target backend's artifact extracted+synced onto this pod via
#     scripts/extract_slam_backend_artifact.sh (locally) -- landing at
#     /colcon_ws/install, /pixi-project/.pixi/envs/slam-<backend>, and
#     /pixi-project/activate-<backend>.sh (see that script's header for why
#     the activation script is renamed per-backend rather than using the
#     generic /pixi-project/activate.sh docker/slam/*/Dockerfile normally
#     uses -- that fixed filename is already claimed on this pod by the sim
#     image's own sim-ros2 env). /pixi-project/.pixi/envs already exists on
#     the image (it holds sim-ros2), so the artifact's `mv` into it works.
#   - pixi's sim-ros2 env (rclpy, pyzmq, msgpack, evo) already provisioned
#     into this pod's image at /pixi-project/.pixi/envs/sim-ros2 +
#     /pixi-project/activate.sh (docker/remote_v6/Dockerfile, the NEW stack:
#     Isaac Sim 6.2 / Lab 3.0 GA; the old docker/remote image has the same
#     layout). Isaac Lab itself runs in Isaac Sim's own Python, not in that env.
#   - For --scene grscenes:<id>: the GRScenes data present (scripts/
#     fetch_grscenes_from_r2.py) and a spawn: an admission manifest row
#     (data/scenes/<source>/admission.json, written by scripts/
#     validate_scenes.py) or an explicit --spawn x,y,yaw,floor_z.
#
# Usage:
#   scripts/run_slam_eval_remote.sh \
#     --backend fast_lio2 --mission household --sensor-rig lidar_camera \
#     --checkpoint <rsl-rl 5.x model.pt | exported policy .onnx> \
#     [--scene stock|grscenes:<id>] [--spawn-index N | --spawn x,y,yaw,floor_z] \
#     --timeout 20m [--no-terminate]
#
# --smoke: the single cheap plumbing check that needs NO backend artifact on the
# pod. Runs the sim loop + the rclpy relay + a topic counter/bag recorder
# (record_mission_bag.py --smoke), with no backend adapter/algorithm, no evo
# scoring, no R2 sync and no self-terminate. Afterwards slam/slam/eval/
# smoke_check.py finalize prints one line per sensor topic (message count,
# stamps, bag count), the sim-side metadata (scene, spawn, ACTIVE physics +
# actuator mode, policy file type, final base height/tilt), writes
# <run dir>/smoke_check.json and this script exits non-zero if any check fails
# (a sensor with zero messages included). No --backend: results land under
# slam/slam/eval/results/smoke/<run_timestamp>/.
#   scripts/run_slam_eval_remote.sh --smoke --mission household --sensor-rig lidar_camera \
#     --checkpoint policy/locomotion/checkpoints/go2_locomotion_flat_v1_candidate.onnx \
#     --scene grscenes:MVUHLWYKTKJ5EAABAAAAACY8_usd --spawn 0,0,0,0 --timeout 30m
set -u

# Isaac Sim's launcher and Kit need a TERM over non-tty ssh (docs/runpod.md).
export TERM=xterm

if [[ -f /etc/cerberus_env.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/cerberus_env.sh
fi

BACKEND=""
MISSION=""
SENSOR_RIG=""
CHECKPOINT=""
SCENE="stock"
SPAWN_INDEX=""
SPAWN=""
JOB_TIMEOUT=""
NO_TERMINATE=0
SMOKE=0

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
        --no-terminate) NO_TERMINATE=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        *) echo "[run_slam_eval_remote] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$SMOKE" == "1" ]]; then
    if [[ -n "$BACKEND" ]]; then
        echo "[run_slam_eval_remote] ERROR: --smoke runs without a backend; drop --backend $BACKEND" >&2
        exit 1
    fi
    BACKEND="smoke"  # only names the results dir
fi

if [[ -z "$BACKEND" || -z "$MISSION" || -z "$SENSOR_RIG" || -z "$CHECKPOINT" || -z "$JOB_TIMEOUT" ]]; then
    echo "Usage: $0 --backend <name> --mission <name> --sensor-rig <lidar_camera|camera_only|camera_stereo> --checkpoint <path.pt|path.onnx> --timeout <duration> [--scene stock|grscenes:<id>] [--spawn-index N | --spawn x,y,yaw,floor_z] [--no-terminate]   (or: --smoke instead of --backend)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$REPO_ROOT/slam/slam/eval/results/$BACKEND/$RUN_TIMESTAMP"
BAG_PATH="$RUN_DIR/bag"
SIM_DONE_FILE="$RUN_DIR/.sim_exited"

echo "[run_slam_eval_remote] backend=$BACKEND mission=$MISSION sensor_rig=$SENSOR_RIG scene=$SCENE spawn_index=${SPAWN_INDEX:-default} spawn=${SPAWN:-manifest} run_timestamp=$RUN_TIMESTAMP"

# --spawn-index/--spawn are passed through only when given; the stock-ground path just gets --scene stock.
SCENE_ARGS=(--scene "$SCENE")
[[ -n "$SPAWN_INDEX" ]] && SCENE_ARGS+=(--spawn-index "$SPAWN_INDEX")
# `--spawn=VALUE`, not `--spawn VALUE`: a negative x ("-4.1,1.3,0,0") would otherwise be read by argparse as a flag.
[[ -n "$SPAWN" ]] && SCENE_ARGS+=("--spawn=$SPAWN")

# Preflight: fail in seconds, before any node or the (multi-minute) Isaac boot starts.
SIM_ROS2_BIN="/pixi-project/.pixi/envs/sim-ros2/bin"
PREFLIGHT_OK=1
preflight_require() {  # <test-flag> <path> <what-it-is-for>
    if [ ! "$1" "$2" ]; then
        echo "[run_slam_eval_remote] ERROR: missing $2 ($3)" >&2
        PREFLIGHT_OK=0
    fi
}
preflight_require -f /pixi-project/activate.sh "sim-ros2 activation script, from the sim image"
if [[ "$SMOKE" != "1" ]]; then
    preflight_require -x "$SIM_ROS2_BIN/evo_ape" "evo scoring CLI, pip-installed into sim-ros2 by the sim image"
    preflight_require -x "$SIM_ROS2_BIN/evo_rpe" "evo scoring CLI, pip-installed into sim-ros2 by the sim image"
    preflight_require -f "/pixi-project/activate-${BACKEND}.sh" "backend activation script; run scripts/extract_slam_backend_artifact.sh $BACKEND and sync it (its mv needs /pixi-project/.pixi/envs, which the image provides)"
    preflight_require -f /colcon_ws/install/setup.bash "backend colcon workspace, same artifact"
fi
preflight_require -f "$(cd "$REPO_ROOT" && realpath -m "$CHECKPOINT")" "--checkpoint (relative paths resolve from the repo root)"
if [[ -z "${ISAACLAB_PATH:-}" || ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
    echo "[run_slam_eval_remote] ERROR: ISAACLAB_PATH is unset or has no isaaclab.sh (is /etc/cerberus_env.sh sourced?)" >&2
    PREFLIGHT_OK=0
fi
if [[ "$PREFLIGHT_OK" != "1" ]]; then
    exit 1
fi

# 0. LiDAR proxy mesh for a scene run. Isaac Lab's MultiMeshRayCaster cannot parse a raw GRScenes house (about a
#    third of its Mesh prims are empty), so the LiDAR casts against <scene folder>/lidar_proxy.usd instead, built
#    once from the valid geometry (scripts/build_lidar_proxy.py; cached, so reruns skip it). Needs pxr only.
if [[ "$SCENE" != "stock" ]]; then
    echo "[run_slam_eval_remote] building/validating the LiDAR proxy mesh for $SCENE..."
    ( cd "$REPO_ROOT" && "${ISAACLAB_PATH}/isaaclab.sh" -p scripts/build_lidar_proxy.py --scene "$SCENE" ) || {
        echo "[run_slam_eval_remote] ERROR: LiDAR proxy build failed for $SCENE" >&2
        exit 1
    }
fi

# 1a. The backend's own SLAM algorithm node -- launch_algorithm.sh is a
#     per-backend script (slam/slam/backends/<name>/launch_algorithm.sh)
#     since each backend's real launch command (executable name, config
#     file, topic remaps) is genuinely backend-specific; not every backend
#     has one written yet, so this step is skipped with a loud warning
#     rather than failing the whole run if it's missing.
ALGORITHM_SCRIPT="$REPO_ROOT/slam/slam/backends/${BACKEND}/launch_algorithm.sh"
ALGORITHM_PID=""
BACKEND_PID=""
if [[ "$SMOKE" == "1" ]]; then
    echo "[run_slam_eval_remote] --smoke: no backend algorithm/adapter, no scoring, no R2 sync, no self-terminate."
elif [[ -f "$ALGORITHM_SCRIPT" ]]; then
    ( cd "$REPO_ROOT" && exec bash -c "source /pixi-project/activate-${BACKEND}.sh && source /colcon_ws/install/setup.bash && bash '$ALGORITHM_SCRIPT'" ) &
    ALGORITHM_PID=$!
else
    echo "[run_slam_eval_remote] WARNING: no $ALGORITHM_SCRIPT -- running the adapter alone, against whatever (if anything) is already publishing the backend's native odom topic." >&2
fi

# 1b. Backend adapter -- own subshell, own sourced env (its extracted
#    activation script + colcon workspace), so it never touches this
#    script's own shell or collides with the relay's sim-ros2 env below.
if [[ "$SMOKE" != "1" ]]; then
    ( cd "$REPO_ROOT" && exec bash -c "source /pixi-project/activate-${BACKEND}.sh && source /colcon_ws/install/setup.bash && python3 -m slam.slam.backends.${BACKEND}.adapter_launch" ) &
    BACKEND_PID=$!
fi

# 2. ROS2-relay half of the sim bridge -- sim-ros2's own activation script,
#    the generic /pixi-project/activate.sh (owned by docker/remote/Dockerfile
#    on this image, not the backend's -- see this script's header note).
( cd "$REPO_ROOT" && exec bash -c "source /pixi-project/activate.sh && python3 -m slam.slam.ground_truth.ros2_bridge_relay" ) &
RELAY_PID=$!

sleep 3  # let both nodes finish coming up before the sim starts publishing

# 3. Record the mission's contract topics -- started CONCURRENTLY with (just
#    ahead of) the sim loop below, not after it: the sim loop is the only
#    process that ever publishes pose/sensor data onto ROS2 (via the relay),
#    so a bag recorded after it has already exited and closed its socket
#    would capture nothing. The recorder publishes no commands (the sim loop
#    applies the canned schedule itself, its process has no rclpy) and stops
#    when $SIM_DONE_FILE appears, which this script creates below once the sim
#    loop has exited -- NOT after a wall-clock mission length, since the sim
#    usually runs slower than real time and a timer truncates the bag.
#    Backgrounded (not `wait`ed on yet) so the sim loop can run in the foreground.
# Needs rclpy/ros2 -- this script's own shell never sources any pixi env itself,
# only the backgrounded per-step subshells do, so this call needs the same
# explicit sourcing or it runs against whatever bare python3 is first on PATH
# and fails with "ModuleNotFoundError: No module named 'rclpy'" (confirmed by
# hitting exactly that against a real pod run).
#    --smoke: count the relay's topics instead (backend-free check, see the header).
RECORD_EXTRA=""
if [[ "$SMOKE" == "1" ]]; then
    RECORD_EXTRA="--smoke"
fi
( cd "$REPO_ROOT" && exec bash -c "source /pixi-project/activate.sh && python3 slam/slam/eval/record_mission_bag.py \
    --backend '$BACKEND' --mission '$MISSION' --run-timestamp '$RUN_TIMESTAMP' --sensor-rig '$SENSOR_RIG' --sim-done-file '$SIM_DONE_FILE' $RECORD_EXTRA" ) &
BAG_PID=$!

# 4. Isaac Lab sim loop -- the actual bounded-duration piece; `timeout`
#    wraps this specifically (backend adapter + relay are long-running
#    servers with no natural end, only meaningful to wrap the sim step).
#    --headless is the compat flag common/cerberus_compat/launcher.py adds when
#    the Lab 3.0 launcher dropped it; --enable_cameras is not passed (Lab 3.0
#    does not register it) -- the loop sets it on its own namespace, since
#    every rig renders cameras. The loop resolves --scene/--spawn* itself, sets
#    MDL_SYSTEM_PATH before Kit starts, and writes sim_loop_report.json into
#    the run dir (synced below).
( cd "$REPO_ROOT" && timeout "$JOB_TIMEOUT" "${ISAACLAB_PATH}/isaaclab.sh" -p slam/slam/ground_truth/sim_ground_truth_loop.py \
    --mission "$MISSION" --sensor-rig "$SENSOR_RIG" --checkpoint "$CHECKPOINT" "${SCENE_ARGS[@]}" \
    --backend "$BACKEND" --run-timestamp "$RUN_TIMESTAMP" --headless )
SIM_EXIT_CODE=$?
echo "[run_slam_eval_remote] sim loop exited with code $SIM_EXIT_CODE"

mkdir -p "$RUN_DIR" && touch "$SIM_DONE_FILE"  # tells the recorder the sim is done (also after a crash or timeout)
wait "$BAG_PID"
BAG_EXIT_CODE=$?

kill "$RELAY_PID" ${BACKEND_PID:+"$BACKEND_PID"} ${ALGORITHM_PID:+"$ALGORITHM_PID"} 2>/dev/null
wait "$RELAY_PID" ${BACKEND_PID:+"$BACKEND_PID"} ${ALGORITHM_PID:+"$ALGORITHM_PID"} 2>/dev/null

if [[ "$SMOKE" == "1" ]]; then
    SMOKE_EXIT_CODE=0
    if [[ "$SIM_EXIT_CODE" != "0" ]]; then
        echo "[smoke] FAIL sim loop exited with code $SIM_EXIT_CODE (see $RUN_DIR/sim_loop_report.json)" >&2
        SMOKE_EXIT_CODE=1
    fi
    if [[ "$BAG_EXIT_CODE" != "0" ]]; then
        echo "[smoke] FAIL topic recorder exited with code $BAG_EXIT_CODE" >&2
        SMOKE_EXIT_CODE=1
    fi
    "$SIM_ROS2_BIN/python3" "$REPO_ROOT/slam/slam/eval/smoke_check.py" finalize --run-dir "$RUN_DIR" --expect-scene "$SCENE" || SMOKE_EXIT_CODE=1
    echo "[run_slam_eval_remote] smoke exit code $SMOKE_EXIT_CODE; pod left running (smoke never syncs to R2 or self-terminates)."
    exit "$SMOKE_EXIT_CODE"
fi

if [[ "$SIM_EXIT_CODE" != "0" ]]; then
    # The sim loop's report (sim_loop_report.json, written first even on a crash) says why; a bag from a failed
    # mission is not scoreable, so keep the evidence, skip scoring and leave the pod running.
    echo "[run_slam_eval_remote] ERROR: sim loop failed (exit $SIM_EXIT_CODE) -- see $RUN_DIR/sim_loop_report.json. Skipping scoring; NOT self-terminating." >&2
    python3 "$REPO_ROOT/scripts/sync_to_r2.py" sync-run --module slam --run-dir "$RUN_DIR" || echo "[run_slam_eval_remote] WARNING: R2 sync of the failed run also failed." >&2
    exit "$SIM_EXIT_CODE"
fi

if [[ "$BAG_EXIT_CODE" != "0" ]]; then
    echo "[run_slam_eval_remote] ERROR: record_mission_bag.py failed (exit $BAG_EXIT_CODE) -- skipping scoring, leaving pod running for manual inspection." >&2
    exit "$BAG_EXIT_CODE"
fi

# Refuse to score a run whose collision/clearance check did not pass. The canned schedule is open loop, so a run
# that pushed into a wall or furniture would produce drift numbers that measure the collision, not the backend.
# slam/slam/ground_truth/collision_monitor.py owns the verdict and its (provisional) thresholds; the sim loop
# records it in sim_loop_report.json every step, so it is present even after a crash.
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
    echo "[run_slam_eval_remote] ERROR: collision check did not pass ($SCORING_VERDICT) -- skipping scoring (the bag would measure a collision, not the backend); NOT self-terminating." >&2
    python3 "$REPO_ROOT/scripts/sync_to_r2.py" sync-run --module slam --run-dir "$RUN_DIR" || echo "[run_slam_eval_remote] WARNING: R2 sync of the unscored run also failed." >&2
    exit 3
fi

# 5. Score. score_with_evo.py shells out to the `evo_ape`/`evo_rpe` CLIs (pip
#    package `evo`), which live in the sim image's sim-ros2 pixi env (pixi.toml's
#    [feature.sim-ros2.pypi-dependencies]); prepend that env's bin/ directly
#    rather than sourcing its activation script into this shell.
( cd "$REPO_ROOT" && PATH="$SIM_ROS2_BIN:$PATH" "$SIM_ROS2_BIN/python3" - "$BAG_PATH" "$BACKEND" <<'PYEOF'
import sys
from slam.slam.eval.score_with_evo import run_evo_scoring
run_evo_scoring(sys.argv[1], sys.argv[2])
PYEOF
)
SCORE_EXIT_CODE=$?
echo "[run_slam_eval_remote] scoring exited with code $SCORE_EXIT_CODE"

# 6. Final sync -- record_mission_bag.py already synced the bag itself;
#    this covers the scored .zip results evo_ape/evo_rpe wrote alongside
#    it, outside record_mission_bag.py's own sync scope.
echo "[run_slam_eval_remote] Final sync: $RUN_DIR -> R2 (module=slam)..."
SYNC_OK=1
python3 "$REPO_ROOT/scripts/sync_to_r2.py" sync-run --module slam --run-dir "$RUN_DIR" || SYNC_OK=0

if [[ "$SYNC_OK" != "1" ]]; then
    echo "[run_slam_eval_remote] ERROR: R2 sync failed -- NOT self-terminating. Investigate, then sync/terminate manually." >&2
    exit "$SCORE_EXIT_CODE"
fi

if [[ "$NO_TERMINATE" == "1" ]]; then
    echo "[run_slam_eval_remote] --no-terminate given -- sync succeeded, leaving pod running."
    exit "$SCORE_EXIT_CODE"
fi

if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "[run_slam_eval_remote] WARNING: RUNPOD_POD_ID not set (not running on a RunPod pod?) -- skipping self-terminate." >&2
    exit "$SCORE_EXIT_CODE"
fi

echo "[run_slam_eval_remote] Sync succeeded. Self-terminating pod $RUNPOD_POD_ID..."
if command -v runpodctl >/dev/null 2>&1; then
    runpodctl remove pod "$RUNPOD_POD_ID" || echo "[run_slam_eval_remote] WARNING: runpodctl remove pod failed (no API key on this image? RUNPOD_API_KEY is no longer in /etc/cerberus_env.sh) -- terminate the pod manually." >&2
elif [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    curl -sS -X DELETE "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" \
        -H "Authorization: Bearer $RUNPOD_API_KEY"
else
    echo "[run_slam_eval_remote] WARNING: no runpodctl and no RUNPOD_API_KEY -- cannot self-terminate. Terminate the pod manually." >&2
fi

exit "$SCORE_EXIT_CODE"
