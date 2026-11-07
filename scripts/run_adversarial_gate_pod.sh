#!/usr/bin/env bash
# Phase 0.5 adversarial gate: wrapper sanity -> adversary training -> per-modality gate verdict.
#
#   1 zero attack       (wrapper must reproduce the standard success rate; otherwise it measures itself)
#   2 dry run           (64-env zero-perturbation rollout, no NaN, frozen policy verified unchanged)
#   3 adversary train   (PPO, force + command channels, 3 h cap)
#   4 gate              (reduce-until-survives per modality -> gate_verdict.json)
#
# Report dir: data/reports/adversarial_gate_<stamp>/.
#
# Usage:
#   scripts/run_adversarial_gate_pod.sh --candidate <ckpt.pt> [--standard_success 0.94] \
#       [--cap_h 3] [--num_envs 256] [--iterations 1000] [--upload]
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
TASK="Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Play-v0"

CANDIDATE=""; STANDARD_SUCCESS=""; CAP_H=3; NUM_ENVS=256; ITERATIONS=1000; TRIALS=100; UPLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate) CANDIDATE="$2"; shift 2 ;;
        --standard_success) STANDARD_SUCCESS="$2"; shift 2 ;;
        --cap_h) CAP_H="$2"; shift 2 ;;
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --upload) UPLOAD=1; shift ;;
        *) echo "[p3] unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$CANDIDATE" ]] || { echo "[p3] --candidate is required" >&2; exit 2; }
[[ -f "$CANDIDATE" ]] || { echo "[p3] candidate not found: $CANDIDATE" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="data/reports/adversarial_gate_$STAMP"
SUMMARY="$REPORT_DIR/summary.txt"
mkdir -p "$REPORT_DIR"
say() { echo "[p3 $(date -u +%H:%M:%S)] $*"; }
FAILS=0
record() { echo "$1 $2 $3" | tee -a "$SUMMARY"; [[ "$2" == PASS || "$2" == SKIP ]] || FAILS=$((FAILS + 1)); }
CAP_S="$(python3 -c "print(int(float('$CAP_H') * 3600))")"

GATE="policy/locomotion/adversarial_robustness/eval/gate_check.py"
ADV="policy/locomotion/adversarial_robustness/train_adversary.py"
say "candidate: $CANDIDATE  cap: ${CAP_H}h  envs=$NUM_ENVS iterations=$ITERATIONS"

# ---- 1 + 2: wrapper checks ---------------------------------------------------------------------
ZERO_ARGS=()
[[ -n "$STANDARD_SUCCESS" ]] && ZERO_ARGS=(--standard_success "$STANDARD_SUCCESS")
timeout --signal=KILL --kill-after=30 1200 "$LAB" -p "$GATE" \
    --task "$TASK" --checkpoint "$CANDIDATE" --zero_attack_only --trials "$TRIALS" \
    "${ZERO_ARGS[@]}" --out_dir "$REPORT_DIR/zero_attack" > "$REPORT_DIR/zero_attack.log" 2>&1
rc=$?
if [[ -f "$REPORT_DIR/zero_attack/zero_attack_check.json" ]] && python3 -c "
import json,sys; d=json.load(open('$REPORT_DIR/zero_attack/zero_attack_check.json')); sys.exit(0 if d.get('passed') else 1)"; then
    record zero_attack PASS "$REPORT_DIR/zero_attack/zero_attack_check.json"
else
    record zero_attack FAIL "exit=$rc (see zero_attack.log)"
fi

timeout --signal=KILL --kill-after=30 1200 "$LAB" -p "$ADV" \
    --task "$TASK" --checkpoint "$CANDIDATE" --num_envs 64 --iterations 1 --dry_run \
    > "$REPORT_DIR/dry_run.log" 2>&1
rc=$?
if [[ $rc -eq 0 ]] && grep -q "dry run PASS" "$REPORT_DIR/dry_run.log"; then
    record dry_run PASS "$REPORT_DIR/dry_run.log"
else
    record dry_run FAIL "exit=$rc (see dry_run.log)"
fi

# Training is pointless if the wrapper does not reproduce the standard behaviour.
if [[ "$(grep -c '^zero_attack FAIL' "$SUMMARY" 2>/dev/null || echo 0)" != "0" ]]; then
    say "zero-attack check failed: not spending the ${CAP_H}h training cap on a mis-measuring wrapper"
    record train SKIP "blocked by zero_attack FAIL"
    record gate SKIP "blocked by zero_attack FAIL"
else
    ADV_DIR="policy/locomotion/checkpoints/adversary_$STAMP"
    timeout --signal=KILL --kill-after=60 "$CAP_S" "$LAB" -p "$ADV" \
        --task "$TASK" --checkpoint "$CANDIDATE" \
        --num_envs "$NUM_ENVS" --iterations "$ITERATIONS" --out_dir "$ADV_DIR" \
        > "$REPORT_DIR/train.log" 2>&1
    rc=$?
    ADV_MODEL="$(find "$ADV_DIR" -maxdepth 1 -name 'model_*.pt' 2>/dev/null | tail -1)"
    if [[ -n "$ADV_MODEL" ]]; then
        cp "$REPORT_DIR/train.log" "$REPORT_DIR/" 2>/dev/null
        cp "$ADV_DIR/train_manifest.json" "$REPORT_DIR/" 2>/dev/null
        record train PASS "$ADV_DIR (exit=$rc)"
    else
        record train FAIL "exit=$rc no checkpoint in $ADV_DIR"
    fi

    if [[ -n "$ADV_MODEL" ]]; then
        timeout --signal=KILL --kill-after=60 5400 "$LAB" -p "$GATE" \
            --task "$TASK" --checkpoint "$CANDIDATE" --adversary_dir "$ADV_DIR" \
            --modalities combined,force_only,command_only --trials "$TRIALS" \
            --zero_attack_check --out_dir "$REPORT_DIR/gate" > "$REPORT_DIR/gate.log" 2>&1
        rc=$?
        if [[ -f "$REPORT_DIR/gate/gate_verdict.json" ]]; then
            cp "$REPORT_DIR/gate/gate_verdict.json" "$REPORT_DIR/"
            record gate PASS "$(python3 -c "import json;d=json.load(open('$REPORT_DIR/gate_verdict.json'));print(d['verdict'], 'min=%.3f' % (d['minimum_inducing_score'] or -1))")"
        else
            record gate FAIL "exit=$rc (see gate.log)"
        fi
    else
        record gate SKIP "no trained adversary"
    fi
fi

# Promotion happens ONLY on PASS, and only through the finalize path so the sidecar carries the verdict.
if [[ -f "$REPORT_DIR/gate_verdict.json" ]] && python3 -c "
import json,sys; sys.exit(0 if json.load(open('$REPORT_DIR/gate_verdict.json'))['verdict']=='PASS' else 1)"; then
    say "gate PASSED: promoting to go2_locomotion_final"
    python3 - "$CANDIDATE" "$REPORT_DIR/gate_verdict.json" <<'PY'
import json, shutil, sys
from pathlib import Path
candidate, verdict_path = Path(sys.argv[1]), Path(sys.argv[2])
final = candidate.parent / "go2_locomotion_final.pt"
for path in (final, final.with_suffix(".onnx"), final.with_suffix(".meta.json")):
    if path.exists():
        sys.exit(f"refusing to overwrite {path}")
shutil.copy2(candidate, final)
meta = json.loads(candidate.with_suffix(".meta.json").read_text())
meta["gate"] = json.loads(verdict_path.read_text())
meta["promoted_from"] = str(candidate)
final.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
onnx = candidate.with_suffix(".onnx")
if onnx.is_file():
    shutil.copy2(onnx, final.with_suffix(".onnx"))
print(f"promoted -> {final}")
PY
    record promote "$([[ -f policy/locomotion/checkpoints/go2_locomotion_final.pt ]] && echo PASS || echo FAIL)" \
        "policy/locomotion/checkpoints/go2_locomotion_final.pt"
else
    say "gate did not pass: go2_locomotion_final.pt must NOT exist"
    [[ -f policy/locomotion/checkpoints/go2_locomotion_final.pt ]] \
        && record promote FAIL "final checkpoint exists despite a non-PASS verdict" \
        || record promote SKIP "no final checkpoint created (correct on FAIL)"
fi

if [[ $UPLOAD -eq 1 ]]; then
    if [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" && -n "${R2_BUCKET:-}" ]]; then
        RCLONE_CONFIG_CERBERUS_R2_TYPE=s3 RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare \
        RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
        RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
        RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        RCLONE_CONFIG_CERBERUS_R2_ACL=private RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true \
            rclone copy "$REPORT_DIR" "cerberus_r2:${R2_BUCKET}/migration/adversarial_gate_$STAMP" --min-size 1B \
            && say "uploaded -> migration/adversarial_gate_$STAMP/" || say "WARNING: upload failed"
    else say "WARNING: --upload without R2_* vars"; fi
fi

say "done: $FAILS failed step(s). Summary: $SUMMARY"
exit "$FAILS"
