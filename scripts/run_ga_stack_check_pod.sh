#!/usr/bin/env bash
# GA-stack bring-up checks + September-checkpoint parity eval, in one detached session.
#
#   1 env report      (python3, no Kit)
#   2 quat/physics    (standing Go2: projected_gravity == [0,0,-1]; resolved backend must be PhysX)
#   3 smoke           (64 envs, ~200 steps, no NaN, checkpoint written)
#   4 parity eval     (September checkpoint through the .onnx export, same grid as 2026-09-08)
#
# Every step's pass/fail is judged from its artifact, not its exit code: SimulationApp exits 0 whatever
# happened. Report dir: data/reports/ga_stack_<stamp>/. Uploads to R2 when the R2_* vars are present.
#
# Usage (on the pod, repo at /workspace/cerberus):
#   scripts/run_ga_stack_check_pod.sh [--skip eval|smoke|probe] [--upload]
set -uo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
if [[ -f /etc/cerberus_env.sh ]]; then source /etc/cerberus_env.sh; fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
LAB="$ISAACLAB_PATH/isaaclab.sh"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="data/reports/ga_stack_$STAMP"
SEPT_CHECKPOINT="${SEPT_CHECKPOINT:-policy/locomotion/checkpoints/go2_locomotion_v1_candidate.pt}"
SEPT_RUN_DIR="${SEPT_RUN_DIR:-policy/locomotion/runs/2026-09-08_19-28-32_analysis}"
PY="python3"
SKIP=""; UPLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip) SKIP="$SKIP $2"; shift 2 ;;
        --upload) UPLOAD=1; shift ;;
        *) echo "[p0p1] unknown argument: $1" >&2; exit 2 ;;
    esac
done
skipped() { [[ " $SKIP " == *" $1 "* ]]; }
say() { echo "[p0p1 $(date -u +%H:%M:%S)] $*"; }

mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
FAILS=0
record() { echo "$1 $2 $3" | tee -a "$SUMMARY"; [[ "$2" == PASS || "$2" == SKIP ]] || FAILS=$((FAILS + 1)); }

run_isaac() {  # run_isaac <name> <timeout_s> <attempts> -- cmd...
    local name="$1" tmo="$2" tries="$3" rc=1 i; shift 4
    for ((i = 1; i <= tries; i++)); do
        say "$name attempt $i/$tries"
        timeout --signal=KILL --kill-after=30 "$tmo" "$@" > "$REPORT_DIR/$name.attempt$i.log" 2>&1
        rc=$?
        cp "$REPORT_DIR/$name.attempt$i.log" "$REPORT_DIR/$name.log"
        say "$name attempt $i exit=$rc (124/137 = timeout)"
        [[ $rc -eq 0 ]] && return 0
    done
    return "$rc"
}

say "report dir: $REPORT_DIR  isaaclab: $LAB"
{ echo "stamp $STAMP"; echo "git ${CERBERUS_SYNC_COMMIT:-unknown}"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; } \
    | tee "$REPORT_DIR/env.txt" >/dev/null

# ---- 1. static environment report -------------------------------------------------------------
if skipped envreport; then record envreport SKIP -; else
    if "$PY" scripts/env_report.py --output outputs/env_report.json --also "$REPORT_DIR/env_report.json" > "$REPORT_DIR/envreport.log" 2>&1; then
        cp outputs/env_report.json "docs/reports/env_report_${STAMP%%T*}.json"
        record envreport PASS "$REPORT_DIR/env_report.json"
    else record envreport FAIL "see $REPORT_DIR/envreport.log"; fi
fi

# ---- 2. quat convention + resolved physics -----------------------------------------------------
if skipped probe; then record probe SKIP -; else
    run_isaac probe 1200 2 -- "$LAB" -p scripts/check_isaac_quat_convention.py \
        --task Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0 \
        --merge-into outputs/env_report.json \
        --output "$REPORT_DIR/quat_convention.json"
    rc=$?
    if "$PY" - "$REPORT_DIR/quat_convention.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("passed") is True else 1)
PY
    then record probe PASS "quat=$(jq -r .quat_convention "$REPORT_DIR/quat_convention.json" 2>/dev/null || echo '?')"
    else record probe FAIL "exit=$rc (see probe.log)"; fi
fi

# ---- 3. headless smoke: 64 envs, ~200 steps ----------------------------------------------------
if skipped smoke; then record smoke SKIP -; else
    MARKER="$REPORT_DIR/.smoke_marker"; : > "$MARKER"
    run_isaac smoke 1800 2 -- "$LAB" -p policy/locomotion/train.py \
        --platform go2 --terrain rough --headless --logger tensorboard \
        --num_envs 64 --max_iterations 8 --video_interval 100000
    rc=$?
    new_ckpt="$(find policy/locomotion/checkpoints -name 'model_*.pt' -newer "$MARKER" -size +0 2>/dev/null | tail -1)"
    if [[ $rc -eq 0 && -n "$new_ckpt" ]] && ! grep -qiE "nan|shape mismatch|Traceback" "$REPORT_DIR/smoke.log" \
        && grep -qE "Mean reward|mean_reward" "$REPORT_DIR/smoke.log"; then
        record smoke PASS "$new_ckpt"
    else
        record smoke FAIL "exit=$rc ckpt='$new_ckpt' nan_or_traceback=$(grep -ciE 'nan|shape mismatch|Traceback' "$REPORT_DIR/smoke.log")"
    fi
fi

# ---- 4. parity eval of the September checkpoint on this stack ---------------------------------
if skipped eval; then record eval SKIP -; else
    if [[ ! -f "$SEPT_CHECKPOINT" ]]; then
        record eval FAIL "missing $SEPT_CHECKPOINT"
    else
        ONNX="${SEPT_CHECKPOINT%.pt}.onnx"
        ONNX_ARGS=()
        [[ -f "$ONNX" ]] && ONNX_ARGS=(--policy_onnx "$ONNX")
        run_isaac eval 5400 2 -- "$LAB" -p policy/locomotion/eval/push_recovery_eval.py \
            --platform go2 --terrain rough \
            --checkpoint "$SEPT_CHECKPOINT" "${ONNX_ARGS[@]}" \
            --contract_reference "$SEPT_RUN_DIR" \
            --magnitudes 0,40,80,120,160,200 --terrain_levels 0,4,8 --trials_per_cell 100 \
            --output "$REPORT_DIR/eval_rough_september_ckpt_on_ga.csv"
        rc=$?
        if [[ $rc -eq 0 && -s "$REPORT_DIR/eval_rough_september_ckpt_on_ga.csv" ]]; then
            cp "$REPORT_DIR"/eval_rough_september_ckpt_on_ga*.txt "$REPORT_DIR"/ 2>/dev/null
            record eval PASS "$REPORT_DIR/eval_rough_september_ckpt_on_ga.csv"
        else record eval FAIL "exit=$rc"; fi
    fi
fi

if [[ $UPLOAD -eq 1 ]]; then
    if [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" && -n "${R2_BUCKET:-}" ]]; then
        RCLONE_CONFIG_CERBERUS_R2_TYPE=s3 RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare \
        RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
        RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
        RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        RCLONE_CONFIG_CERBERUS_R2_ACL=private RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true \
            rclone copy "$REPORT_DIR" "cerberus_r2:${R2_BUCKET}/migration/ga_stack_$STAMP" --min-size 1B \
            && say "uploaded -> migration/ga_stack_$STAMP/" || say "WARNING: upload failed"
    else say "WARNING: --upload without R2_* vars"; fi
fi

say "done: $FAILS failed step(s). Summary: $REPORT_DIR/summary.txt"
exit "$FAILS"
