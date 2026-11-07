#!/usr/bin/env bash
# GA retrain of the Go2 rough task (RUN_002 recipe, new stack) + finalize + evaluation, detached.
#
# Recipe is RUN_002's, unchanged: 1500 iterations, 4096 envs, seed 42, 24 steps/env, same task id and
# the same reward/PPO/network config the repo registers. The wall-clock cap is 2x RUN_002's measured
# training time (5306.8 s from its event file) => 2.95 h; the timeout enforces it.
#
# Steps: train -> finalize (copy + ONNX + sidecar) -> push-recovery sweep of the new candidate.
# Report dir: data/reports/ga_retrain_<stamp>/.
#
# Usage:
#   scripts/run_ga_retrain_pod.sh [--iterations 1500] [--num_envs 4096] [--cap_h 2.95] [--upload]
set -uo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
if [[ -f /etc/cerberus_env.sh ]]; then source /etc/cerberus_env.sh; fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CERBERUS_ALLOW_BAD_DRIVER_RENDER=1   # 595 hosts: the render guard would block --video; rendering itself is fine
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
LAB="$ISAACLAB_PATH/isaaclab.sh"

ITERATIONS=1500; NUM_ENVS=4096; SEED=42; CAP_H=2.95; UPLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --cap_h) CAP_H="$2"; shift 2 ;;
        --upload) UPLOAD=1; shift ;;
        *) echo "[p2] unknown argument: $1" >&2; exit 2 ;;
    esac
done
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="data/reports/ga_retrain_$STAMP"
SUMMARY="$REPORT_DIR/summary.txt"
mkdir -p "$REPORT_DIR"
say() { echo "[p2 $(date -u +%H:%M:%S)] $*"; }
# shellcheck source=scripts/pod_isaac_lib.sh
source "$REPO_ROOT/scripts/pod_isaac_lib.sh"
FAILS=0
record() { echo "$1 $2 $3" | tee -a "$SUMMARY"; [[ "$2" == PASS || "$2" == SKIP ]] || FAILS=$((FAILS + 1)); }
CAP_S="$(python3 -c "print(int(float('$CAP_H') * 3600))")"
say "report dir: $REPORT_DIR  cap: ${CAP_H}h (${CAP_S}s)  iterations=$ITERATIONS envs=$NUM_ENVS seed=$SEED"

# ---- train -------------------------------------------------------------------------------------
MARKER="$REPORT_DIR/.train_marker"; : > "$MARKER"
say "training (RUN_002 recipe, tensorboard logger)"
run_isaac train "$CAP_S" 2 -- "$LAB" -p policy/locomotion/train.py \
    --platform go2 --terrain rough --headless --logger tensorboard \
    --num_envs "$NUM_ENVS" --max_iterations "$ITERATIONS" --seed "$SEED"
rc=$?
say "training exit=$rc (137 = hit the ${CAP_H}h cap)"

RUN_DIR="$(find policy/locomotion/checkpoints -maxdepth 2 -name 'model_*.pt' -newer "$MARKER" -printf '%h\n' 2>/dev/null | head -1)"
EVENTS="$(find "${RUN_DIR:-/nonexistent}" -maxdepth 1 -name 'events.out.tfevents*' 2>/dev/null | head -1)"
if [[ $rc -eq 0 && -n "$RUN_DIR" && -n "$EVENTS" ]]; then
    cp -r "$RUN_DIR" "$REPORT_DIR/run_snapshot" 2>/dev/null
    record train PASS "$RUN_DIR"
else
    record train FAIL "exit=$rc run_dir='$RUN_DIR' events='$EVENTS'"
fi

# ---- rollout video (separate from training on purpose) -----------------------------------------
# Driver 595 crashes or hangs roughly half of camera-enabled Kit launches (docs/runpod.md). A video
# flag inside the retrain would put 1.5 h of training behind that coin flip, so the clip is captured
# afterwards from the finished checkpoint, with retries, and a failed clip does not invalidate the run.
if [[ -n "$RUN_DIR" ]]; then
    for attempt in 1 2 3 4; do
        run_isaac video 900 1 -- "$LAB" -p policy/locomotion/play.py \
            --platform go2 --terrain rough --headless --video --video_length 200 --num_envs 8
        clip="$(find "$RUN_DIR" -name '*.mp4' -size +0 2>/dev/null | tail -1)"
        [[ -n "$clip" ]] && break
        say "video attempt $attempt produced no clip; retrying"
    done
    if [[ -n "$clip" ]]; then
        cp "$clip" "$REPORT_DIR/rollout.mp4"
        record video PASS "$REPORT_DIR/rollout.mp4"
    else record video FAIL "no clip after 4 attempts (non-blocking for the checkpoint)"
    fi
fi

# ---- finalize: candidate .pt + .onnx + sidecar -------------------------------------------------
if [[ -n "$RUN_DIR" ]]; then
    say "finalizing candidate from $RUN_DIR"
    run_isaac finalize 1800 2 -- "$LAB" -p scripts/finalize_ga_checkpoint.py \
        --run-dir "$RUN_DIR" --name go2_locomotion_ga_v1_candidate --physics isaacsim_physx
    rc=$?
    CAND="policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate"
    if [[ -f "$CAND.pt" && -f "$CAND.onnx" && -f "$CAND.meta.json" ]]; then
        cp "$CAND.pt" "$CAND.onnx" "$CAND.meta.json" "$REPORT_DIR/"
        record finalize PASS "$(python3 -c "import json;d=json.load(open('$CAND.meta.json'));print('wall=%.0fs reward=%.2f physics=%s' % (d['wall_clock_s'], d['final_reward'], d['physics_backend']))")"
    else record finalize FAIL "exit=$rc (see finalize.log)"; fi
fi

# ---- evaluate the new candidate on the same sweep as the September reference --------------------
if [[ -f policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt ]]; then
    say "evaluating candidate"
    run_isaac eval 5400 4 -- "$LAB" -p policy/locomotion/eval/push_recovery_eval.py \
        --platform go2 --terrain rough \
        --checkpoint policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt \
        --headless \
        --contract_reference policy/locomotion/runs/2026-09-08_19-28-32_analysis \
        --magnitudes 0,40,80,120,160,200 --terrain_levels 0,4,8 --trials_per_cell 100 \
        --output "$REPORT_DIR/eval_rough_ga_candidate.csv"
    rc=$?
    if [[ $rc -eq 0 && -s "$REPORT_DIR/eval_rough_ga_candidate.csv" ]]; then
        record eval PASS "$REPORT_DIR/eval_rough_ga_candidate.csv"
    else record eval FAIL "exit=$rc (see eval.log)"; fi
else
    record eval SKIP "no candidate checkpoint"
fi

# ---- verdict against the September reference ---------------------------------------------------
if [[ -s "$REPORT_DIR/eval_rough_ga_candidate.csv" ]]; then
    python3 policy/locomotion/eval/make_parity_report.py \
        --reference policy/locomotion/runs/2026-09-08_19-28-32_analysis/eval/eval_rough_full_sweep.csv \
        --candidate "$REPORT_DIR/eval_rough_ga_candidate.csv" \
        --reference-label September --candidate-label "GA retrain" \
        --title "Go2 rough: September checkpoint vs GA retrain" \
        --out-md "$REPORT_DIR/parity.md" --out-plot "$REPORT_DIR/parity.png" \
        >> "$REPORT_DIR/parity.log" 2>&1
    record parity "$([[ -s $REPORT_DIR/parity.md ]] && echo PASS || echo FAIL)" "$REPORT_DIR/parity.md"
fi

if [[ $UPLOAD -eq 1 ]]; then
    if [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" && -n "${R2_BUCKET:-}" ]]; then
        RCLONE_CONFIG_CERBERUS_R2_TYPE=s3 RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare \
        RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
        RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
        RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        RCLONE_CONFIG_CERBERUS_R2_ACL=private RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true \
            rclone copy "$REPORT_DIR" "cerberus_r2:${R2_BUCKET}/migration/ga_retrain_$STAMP" --min-size 1B \
            && say "uploaded -> migration/ga_retrain_$STAMP/" || say "WARNING: upload failed"
    else say "WARNING: --upload without R2_* vars"; fi
fi

say "done: $FAILS failed step(s). Summary: $SUMMARY"
exit "$FAILS"
