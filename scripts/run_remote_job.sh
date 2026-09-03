#!/usr/bin/env bash
# Wraps a training command with a hard wall-clock timeout, an R2 artifact
# sync regardless of outcome, and pod self-termination -- so a real training
# run doesn't depend on any agent or human remembering to watch it, pull
# artifacts, and shut the pod down. See docs/artifact_storage.md.
#
# Deliberately generic (not locomotion-specific): takes the training command
# as trailing args, same --module reuse pattern as scripts/sync_to_r2.py.
#
# Usage:
#   scripts/run_remote_job.sh \
#     --module locomotion \
#     --checkpoints-dir policy/locomotion/checkpoints/unitree_go2_flat \
#     --timeout 4h \
#     -- \
#     ./isaaclab.sh -p /workspace/cerberus/policy/locomotion/train.py --task ... --headless
#
# Safety properties:
#   - Runs the training command under `timeout`, so even a hung/runaway job
#     is bounded -- this is the backstop that doesn't depend on this script,
#     or any agent, still being around to notice.
#   - Syncs to R2 periodically DURING the run (every --sync-interval, default
#     5m), not just once at the end -- a training pod has no persistent
#     network volume (deliberately, to stay cheap -- see docs/isaac_lab_workflow.md),
#     so if RunPod itself force-terminates the pod (e.g. account balance hits
#     $0 -- RunPod terminates a no-network-volume pod outright in that case,
#     disk and all) mid-run, this script's own end-of-run sync never gets a
#     chance to execute at all. Periodic syncing bounds the worst case to
#     --sync-interval of lost progress instead of the entire run. Safe to
#     call repeatedly against the same growing run directory: each sync is
#     scoped to that one run's specific R2 path (see sync_to_r2.py), and the
#     local source only ever grows during a run, so nothing already-uploaded
#     is ever at risk of being deleted by a later sync.
#   - Syncs to R2 regardless of the training command's exit code (success,
#     crash, or timeout kill) -- partial progress isn't lost.
#   - Only self-terminates the pod if the final sync actually succeeded. If
#     it didn't, the pod is left running on purpose so nothing gets lost --
#     investigate and sync/terminate manually.
#
# Requires: R2_* vars (see scripts/sync_to_r2.py) sourced into the
# environment already, and RUNPOD_POD_ID + RUNPOD_API_KEY for the
# self-terminate step -- docker/start.sh writes these to /etc/cerberus_env.sh
# at container start (PID 1 has them from RunPod, but a plain `ssh host
# "command"` session doesn't inherit them on its own -- see start.sh's
# comment), sourced here explicitly rather than relying on PAM/profile.d.
set -u

if [[ -f /etc/cerberus_env.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/cerberus_env.sh
fi

MODULE=""
CHECKPOINTS_DIR=""
JOB_TIMEOUT=""
SYNC_INTERVAL="5m"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --module)
            MODULE="$2"
            shift 2
            ;;
        --checkpoints-dir)
            CHECKPOINTS_DIR="$2"
            shift 2
            ;;
        --timeout)
            JOB_TIMEOUT="$2"
            shift 2
            ;;
        --sync-interval)
            SYNC_INTERVAL="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "[run_remote_job] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODULE" || -z "$CHECKPOINTS_DIR" || -z "$JOB_TIMEOUT" || $# -eq 0 ]]; then
    echo "Usage: $0 --module <name> --checkpoints-dir <path> --timeout <duration> [--sync-interval <duration>] -- <training command...>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Finds the most-recently-modified run dir under CHECKPOINTS_DIR -- the one
# this invocation created. Assumes one job at a time per pod (already true of
# this whole workflow: one dedicated pod per training run).
find_run_dir() {
    find "$CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

# Periodic background sync while training runs -- see the safety-properties
# comment above for why this exists. Runs as a detached loop; killed once the
# training command below exits, then one final sync happens after that.
(
    while true; do
        sleep "$SYNC_INTERVAL"
        PERIODIC_RUN_DIR="$(find_run_dir)"
        if [[ -n "$PERIODIC_RUN_DIR" ]]; then
            echo "[run_remote_job] Periodic sync: $PERIODIC_RUN_DIR -> R2..."
            python3 "$REPO_ROOT/scripts/sync_to_r2.py" sync-run --module "$MODULE" --run-dir "$PERIODIC_RUN_DIR" \
                || echo "[run_remote_job] WARNING: periodic sync failed, will retry next interval." >&2
        fi
    done
) &
SYNC_LOOP_PID=$!

echo "[run_remote_job] Starting (timeout $JOB_TIMEOUT, periodic sync every $SYNC_INTERVAL): $*"
timeout "$JOB_TIMEOUT" "$@"
TRAIN_EXIT_CODE=$?
echo "[run_remote_job] Command exited with code $TRAIN_EXIT_CODE"

kill "$SYNC_LOOP_PID" 2>/dev/null
wait "$SYNC_LOOP_PID" 2>/dev/null

RUN_DIR="$(find_run_dir)"

if [[ -z "$RUN_DIR" ]]; then
    echo "[run_remote_job] ERROR: no run directory found under $CHECKPOINTS_DIR -- skipping sync, leaving pod running for manual inspection." >&2
    exit "$TRAIN_EXIT_CODE"
fi

echo "[run_remote_job] Final sync: $RUN_DIR -> R2 (module=$MODULE)..."
SYNC_OK=1
python3 "$REPO_ROOT/scripts/sync_to_r2.py" sync-run --module "$MODULE" --run-dir "$RUN_DIR" || SYNC_OK=0

if [[ "$SYNC_OK" != "1" ]]; then
    echo "[run_remote_job] ERROR: R2 sync failed -- NOT self-terminating. Investigate, then sync/terminate manually." >&2
    exit "$TRAIN_EXIT_CODE"
fi

if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "[run_remote_job] WARNING: RUNPOD_POD_ID not set (not running on a RunPod pod?) -- skipping self-terminate." >&2
    exit "$TRAIN_EXIT_CODE"
fi

echo "[run_remote_job] Sync succeeded. Self-terminating pod $RUNPOD_POD_ID..."
if command -v runpodctl >/dev/null 2>&1; then
    runpodctl remove pod "$RUNPOD_POD_ID"
elif [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    curl -sS -X DELETE "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" \
        -H "Authorization: Bearer $RUNPOD_API_KEY"
else
    echo "[run_remote_job] WARNING: no runpodctl and no RUNPOD_API_KEY -- cannot self-terminate. Terminate the pod manually." >&2
fi

exit "$TRAIN_EXIT_CODE"
