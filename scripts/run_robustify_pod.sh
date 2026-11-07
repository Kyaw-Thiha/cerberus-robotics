#!/usr/bin/env bash
# Robustify round: fine-tune the candidate with adversarial rollouts, then re-attack it.
#
# Runs only when a gate verdict says FAIL. The paper's loop (Section IV-C/V-A.3):
#   attack -> fine-tune with a small share of rollouts under the discovered attack -> re-attack with a
#   NEW adversary (a stale adversary trained against the old weights is not a valid retest).
#
# This script does the middle step and re-runs the gate; it refuses to promote anything -- the promoted
# checkpoint is only ever produced by a PASS verdict, in run_adversarial_gate_pod.sh.
#
# Usage:
#   scripts/run_robustify_pod.sh --candidate <pt> --adversary_dir <dir> --verdict <gate_verdict.json>
#                                [--iterations 300] [--mix 0.05] [--round 1] [--upload]
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

CANDIDATE=""; ADVERSARY_DIR=""; VERDICT=""; ITERATIONS=300; MIX=0.05; ROUND=1; UPLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate) CANDIDATE="$2"; shift 2 ;;
        --adversary_dir) ADVERSARY_DIR="$2"; shift 2 ;;
        --verdict) VERDICT="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --mix) MIX="$2"; shift 2 ;;
        --round) ROUND="$2"; shift 2 ;;
        --upload) UPLOAD=1; shift ;;
        *) echo "[robustify] unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$CANDIDATE" && -n "$ADVERSARY_DIR" && -n "$VERDICT" ]] || {
    echo "[robustify] --candidate, --adversary_dir and --verdict are required" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="data/reports/robustify_round${ROUND}_$STAMP"
SUMMARY="$REPORT_DIR/summary.txt"
mkdir -p "$REPORT_DIR"
say() { echo "[robustify $(date -u +%H:%M:%S)] $*"; }
# shellcheck source=scripts/pod_isaac_lib.sh
source "$REPO_ROOT/scripts/pod_isaac_lib.sh"
FAILS=0
record() { echo "$1 $2 $3" | tee -a "$SUMMARY"; [[ "$2" == PASS || "$2" == SKIP ]] || FAILS=$((FAILS + 1)); }

# Only a failing gate justifies touching the policy.
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$VERDICT'))['verdict']=='PASS' else 1)"; then
    say "gate verdict is PASS: nothing to robustify"
    record robustify SKIP "gate already passed"
    exit 0
fi
say "gate FAILED: round $ROUND, mix=$MIX, iterations=$ITERATIONS"

NAME="go2_locomotion_v$((ROUND + 1))_candidate"   # round 1 -> v2_candidate (one naming scheme, per project_structure.md)
OUT_STEM="policy/locomotion/checkpoints/$NAME"
if [[ -e "$OUT_STEM.pt" ]]; then
    say "refusing to overwrite $OUT_STEM.pt"
    record finetune FAIL "checkpoint already exists"
    exit 1
fi

say "fine-tuning from $CANDIDATE with adversarial rollouts"
run_isaac finetune 5400 2 -- "$LAB" -p policy/locomotion/train.py \
    --platform go2 --terrain rough --headless --logger tensorboard \
    --resume --checkpoint "$CANDIDATE" \
    --num_envs 4096 --max_iterations "$ITERATIONS" \
    --adversarial_env policy.locomotion.adversarial_robustness.robustify_env:make_adversarial_wrapper \
    --adversarial_params "{\"adversary_dir\": \"$ADVERSARY_DIR\", \"mix_fraction\": $MIX}"
rc=$?

RUN_DIR="$(find policy/locomotion/checkpoints -maxdepth 3 -name 'model_*.pt' -newer "$REPORT_DIR" -printf '%h\n' 2>/dev/null | head -1)"
if [[ $rc -eq 0 && -n "$RUN_DIR" ]]; then
    record finetune PASS "$RUN_DIR"
else
    record finetune FAIL "exit=$rc run_dir='$RUN_DIR'"
fi

if [[ -n "$RUN_DIR" ]]; then
    run_isaac finalize 1800 2 -- "$LAB" -p scripts/finalize_ga_checkpoint.py \
        --run-dir "$RUN_DIR" --name "$NAME" --physics isaacsim_physx \
        --git-hash "${CERBERUS_SYNC_COMMIT:-$(cat "$REPO_ROOT/.sync_commit" 2>/dev/null || echo unknown)}"
    if [[ -f "$OUT_STEM.pt" && -f "$OUT_STEM.meta.json" ]]; then
        cp "$OUT_STEM.pt" "$OUT_STEM.onnx" "$OUT_STEM.meta.json" "$REPORT_DIR/" 2>/dev/null
        record finalize PASS "$OUT_STEM"
    else
        record finalize FAIL "missing $OUT_STEM artifacts"
    fi
fi

# Re-attack with a NEW adversary: the one that broke the old weights says nothing about the new ones.
if [[ -f "$OUT_STEM.pt" ]]; then
    say "re-attack round $ROUND (fresh adversary)"
    # The gate driver writes into its own timestamped directory; watch for a NEW one rather than
    # guessing its name.
    MARKER="$REPORT_DIR/.reattack_marker"; : > "$MARKER"
    setsid bash scripts/run_adversarial_gate_pod.sh \
        --candidate "$OUT_STEM.pt" --standard_fall_rate 0.0 \
        --cap_h 3 --num_envs 256 --iterations 1000 \
        > "$REPORT_DIR/gate_round$ROUND.out" 2>&1 < /dev/null &
    found=""
    for _ in $(seq 1 480); do
        found="$(find data/reports -maxdepth 3 -path '*adversarial_gate_*/gate/gate_verdict.json' -newer "$MARKER" 2>/dev/null | tail -1)"
        [[ -n "$found" ]] && break
        sleep 15
    done
    if [[ -n "$found" ]]; then
        cp "$found" "$REPORT_DIR/gate_verdict_round$ROUND.json"
        record reattack PASS "$(python3 -c "import json;d=json.load(open('$REPORT_DIR/gate_verdict_round$ROUND.json'));print(d['verdict'], d.get('minimum_inducing_score'))")"
    else
        record reattack FAIL "no gate verdict after the re-attack"
    fi
fi

if [[ $UPLOAD -eq 1 && -n "${R2_BUCKET:-}" ]]; then
    RCLONE_CONFIG_CERBERUS_R2_TYPE=s3 RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare \
    RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    RCLONE_CONFIG_CERBERUS_R2_ACL=private RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true \
        rclone copy "$REPORT_DIR" "cerberus_r2:${R2_BUCKET}/migration/robustify_round${ROUND}_$STAMP" --min-size 1B \
        && say "uploaded" || say "WARNING: upload failed"
fi

say "done: $FAILS failed step(s). Summary: $SUMMARY"
exit "$FAILS"
