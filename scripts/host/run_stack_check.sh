#!/usr/bin/env bash
# Host bring-up checks: does this machine run the GA stack, and does the frozen September checkpoint reproduce
# its push-recovery curve here? Steps are judged from their artifacts, not exit codes (SimulationApp exits 0
# whatever happened). Re-created from `scripts/run_ga_stack_check_pod.sh` at tag
# archive/provider-tooling-2026-11-21, minus the pod-only parts: no dead-man switch, no provider calls, and the
# report lands under $CERBERUS_RESULTS_DIR instead of a fixed pod path.
#
#   1 env report      python3 only, no Kit
#   2 quat/physics    standing Go2: projected_gravity == [0,0,-1]; the resolved backend must be PhysX
#   3 smoke           64 envs, a few iterations, no NaN, a checkpoint written
#   4 parity eval     the September checkpoint on the same grid as 2026-09-08
#
# Usage:
#   scripts/host/run_stack_check.sh [--skip envreport|probe|smoke|eval] [--upload] [--skip-gate] [--dry-run]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
# shellcheck source=session_lib.sh
source "$REPO_ROOT/scripts/host/session_lib.sh"
cerberus_defaults

SEPT_CHECKPOINT="${SEPT_CHECKPOINT:-policy/locomotion/checkpoints/go2_locomotion_v1_candidate.pt}"
SEPT_RUN_DIR="${SEPT_RUN_DIR:-policy/locomotion/runs/2026-09-08_19-28-32_analysis}"
SKIP=""
UPLOAD=0
GATE=1
DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip) SKIP="$SKIP ${2//,/ }"; shift 2 ;;
        --upload) UPLOAD=1; shift ;;
        --skip-gate) GATE=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "run_stack_check: unknown argument $1" >&2; exit 2 ;;
    esac
done
skipped() { [[ " $SKIP " == *" $1 "* ]]; }
mkdir -p "$RESULTS_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="$RESULTS_DIR/ga_stack_$STAMP"
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
FAILS=0

cerberus_load_env || true
cerberus_say "report dir: $REPORT_DIR  isaaclab: $LAB"
{ echo "stamp $STAMP"; echo "git $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null; } > "$REPORT_DIR/env.txt"

if [[ $GATE -eq 1 ]]; then
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] driver_gate.sh && launch_probe.sh"; else
        cerberus_gate || { cerberus_record gate FAIL "gate or probe rejected this host"; exit 2; }
    fi
fi

# ---- 1. static environment report --------------------------------------------------------------
if skipped envreport; then cerberus_record envreport SKIP -; else
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] env_report.py"; else
        if "$PY" scripts/env_report.py --output outputs/env_report.json --also "$REPORT_DIR/env_report.json" \
                > "$REPORT_DIR/envreport.log" 2>&1; then
            cerberus_record envreport PASS "$REPORT_DIR/env_report.json"
        else cerberus_record envreport FAIL "see $REPORT_DIR/envreport.log"; fi
    fi
fi

# ---- 2. quat convention + resolved physics -----------------------------------------------------
if skipped probe; then cerberus_record probe SKIP -; else
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] check_isaac_quat_convention.py"; else
        run_isaac probe 900 4 -- bash "$REPO_ROOT/scripts/host/kit_retry.sh" -- "$LAB" -p \
            scripts/check_isaac_quat_convention.py \
            --task Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0 \
            --merge-into outputs/env_report.json --output "$REPORT_DIR/quat_convention.json"
        rc=$?
        if "$PY" - "$REPORT_DIR/quat_convention.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if d.get("passed") is True else 1)
PY
        then cerberus_record probe PASS "$REPORT_DIR/quat_convention.json"
        else cerberus_record probe FAIL "exit=$rc (see probe.log)"; fi
    fi
fi

# ---- 3. headless smoke -------------------------------------------------------------------------
if skipped smoke; then cerberus_record smoke SKIP -; else
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] train.py smoke"; else
        MARKER="$REPORT_DIR/.smoke_marker"; : > "$MARKER"
        run_isaac smoke 1800 2 -- bash "$REPO_ROOT/scripts/host/kit_retry.sh" -- "$LAB" -p policy/locomotion/train.py \
            --platform go2 --terrain rough --headless --logger tensorboard \
            --num_envs 64 --max_iterations 8 --video_interval 100000
        rc=$?
        new_ckpt="$(find policy/locomotion/checkpoints -name 'model_*.pt' -newer "$MARKER" -size +0 2>/dev/null | tail -1)"
        # Not Kit's own chatter: the crash reporter has a "PythonTracebackStatus" field, and a bare "nan"
        # matches "nanoseconds".
        smoke_problem="$(grep -cE "Traceback \(most recent call last\)|(^|[^[:alnum:]])nan([^[:alnum:]]|$)|shape mismatch" "$REPORT_DIR/smoke.log")"
        if [[ $rc -eq 0 && -n "$new_ckpt" && "$smoke_problem" == "0" ]] \
            && grep -qE "Mean reward|mean_reward" "$REPORT_DIR/smoke.log"; then
            cerberus_record smoke PASS "$new_ckpt"
        else
            cerberus_record smoke FAIL "exit=$rc ckpt='$new_ckpt' problems=$smoke_problem"
        fi
    fi
fi

# ---- 4. parity eval of the September checkpoint ------------------------------------------------
if skipped eval; then cerberus_record eval SKIP -; else
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] push_recovery_eval.py"; elif [[ ! -f "$SEPT_CHECKPOINT" ]]; then
        cerberus_record eval FAIL "missing $SEPT_CHECKPOINT"
    else
        ONNX="${SEPT_CHECKPOINT%.pt}.onnx"
        ONNX_ARGS=()
        [[ -f "$ONNX" ]] && ONNX_ARGS=(--policy_onnx "$ONNX")
        run_isaac eval 5400 4 -- bash "$REPO_ROOT/scripts/host/kit_retry.sh" -- "$LAB" -p \
            policy/locomotion/eval/push_recovery_eval.py \
            --platform go2 --terrain rough --checkpoint "$SEPT_CHECKPOINT" "${ONNX_ARGS[@]}" --headless \
            --contract_reference "$SEPT_RUN_DIR" \
            --magnitudes 0,40,80,120,160,200 --terrain_levels 0,4,8 --trials_per_cell 100 \
            --output "$REPORT_DIR/eval_rough_september_ckpt_on_ga.csv"
        rc=$?
        if [[ $rc -eq 0 && -s "$REPORT_DIR/eval_rough_september_ckpt_on_ga.csv" ]]; then
            cerberus_record eval PASS "$REPORT_DIR/eval_rough_september_ckpt_on_ga.csv"
        else cerberus_record eval FAIL "exit=$rc"; fi
    fi
fi

if [[ $UPLOAD -eq 1 ]]; then
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] sync $REPORT_DIR -> migration/ga_stack_$STAMP"; else
        cerberus_sync "$REPORT_DIR" "migration/ga_stack_$STAMP" && cerberus_say "uploaded -> migration/ga_stack_$STAMP/" \
            || cerberus_say "WARNING: upload failed"
    fi
fi

cerberus_say "done: $FAILS failed step(s). Summary: $SUMMARY"
exit "$FAILS"
