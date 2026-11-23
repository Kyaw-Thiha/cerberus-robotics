#!/bin/bash
# Retrain the arm (go2_arx5) locomotion policy, relaunching until the final checkpoint exists.
#
# Why not exit codes: on this host class Kit's native layer exits 0 after a long sim block and the loader dies
# before our code runs, so the only trustworthy signal is the artifact (model_<iterations>.pt) in the run dir.
# Launch failures are handled inside by kit_retry; a post-start failure resumes from the newest checkpoint of the
# newest run directory.
#
# Usage: train_arm_retrain.sh [--iterations 1500] [--envs 4096] [--seed 42] [--max-launches 20]
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/workspace/cerberus-robotics}"
cd "$REPO_ROOT" || exit 1

ITERATIONS=1500; ENVS=4096; SEED=42; MAX_LAUNCHES="${CERBERUS_MAX_LAUNCHES:-20}"
RUN_NAME="${CERBERUS_RUN_NAME:-go2_arx5_ga_v1}"
LOG_ROOT="${CERBERUS_LOG_ROOT:-$REPO_ROOT/policy/locomotion/checkpoints}"
while [ $# -gt 0 ]; do
  case "$1" in
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --envs) ENVS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-launches) MAX_LAUNCHES="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    *) echo "train_arm_retrain: unknown argument $1" >&2; exit 2 ;;
  esac
done

LAB="${ISAACLAB_PATH:-/workspace/isaaclab}/isaaclab.sh"
python_bin() { if [ -x "$LAB" ]; then echo "$LAB -p"; else echo python3; fi; }

final_marker() {
  # the runner writes <log_root>/<task>/<timestamp>_<run_name>/model_<iter>.pt, and iterations are 0-based
  # (1500 iterations end at model_1499.pt), so accept either name.
  find "$LOG_ROOT" -maxdepth 3 \( -name "model_${ITERATIONS}.pt" -o -name "model_$((ITERATIONS - 1)).pt" \) \
    -print -quit 2>/dev/null
}

for launch in $(seq 1 "$MAX_LAUNCHES"); do
  done_marker="$(final_marker)"
  if [ -n "$done_marker" ]; then
    echo "[retrain] final checkpoint present: $done_marker"
    exit 0
  fi
  resume_args=""
  latest_run="$(ls -1dt "$LOG_ROOT"/*/*/ 2>/dev/null | head -1)"
  if [ "$launch" -gt 1 ] && [ -n "$latest_run" ]; then
    ckpt="$(ls -1t "$latest_run"model_*.pt 2>/dev/null | head -1)"
    if [ -n "$ckpt" ]; then
      run_dir_name="$(basename "$latest_run")"
      resume_args="--resume --load_run $run_dir_name --checkpoint $(basename "$ckpt")"
      echo "[retrain] launch $launch resumes from $ckpt"
    fi
  fi
  echo "[retrain] launch $launch/$MAX_LAUNCHES at $(date -u +%FT%TZ)"
  # shellcheck disable=SC2086
  CERBERUS_MAX_LAUNCHES="${CERBERUS_INNER_LAUNCHES:-25}" bash "$REPO_ROOT/scripts/host/kit_retry.sh" -- \
      $(python_bin) policy/locomotion/train.py --platform go2_arx5 --terrain rough \
      --num_envs "$ENVS" --max_iterations "$ITERATIONS" --seed "$SEED" \
      --logger tensorboard --headless --run_name "$RUN_NAME" $resume_args
  rc=$?
  echo "[retrain] launch $launch exited $rc at $(date -u +%FT%TZ)"
  sleep 10
done

echo "[retrain] gave up after $MAX_LAUNCHES launches without model_${ITERATIONS}.pt" >&2
exit 76
