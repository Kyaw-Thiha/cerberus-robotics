#!/usr/bin/env bash
# Lab 3.0 GA-stack smoke for ONE pod session (image cerberus-remote-v6:v6-ga). Verifies, with artifacts, the
# changes that could not be exercised off-pod:
#   fd35698  Go2 task configs imported from isaaclab_tasks.core (Lab 2.x path-alias shim removed)
#   65d6e19  rsl-rl 3.x branches removed (make_runner / export_policy / BestCheckpointOnPolicyRunner on rsl-rl >= 4)
#   (and gives the Arena spike PASS the artifact its first run lost: data/reports/ga_upgrade/arena_spike_go2.json was empty)
#
# Steps (each pass/fail is decided from ARTIFACTS, not exit codes: SimulationApp exits 0 whatever happened):
#   1 train   train.py --config smoke_test        PASS = exit 0 + a NEW model_*.pt written by this run
#   2 play    play.py --video on that checkpoint  PASS = exit 0 + a NEW non-empty .mp4 + policy.onnx/policy.pt exported
#   3 spike   arena_spike_go2.py --scene stock    PASS = JSON report exists, non-empty, "passed": true
#
# Usage (on the pod, repo synced to /workspace/cerberus):
#   scripts/run_ga_smoke_pod.sh [--report-dir DIR] [--skip train|play|spike]... [--upload] [--dry-run]
#
# Notes:
#   - Driver 595 pods hang or crash ~half of Kit launches (docs/arena_integration_log.md, GA section): every Isaac
#     launch runs under a hard `timeout` and is retried once. CERBERUS_ALLOW_BAD_DRIVER_RENDER=1 is set for play
#     (the render guard blocks 595; rendering itself was correct once Kit was up).
#   - Play loads the NEWEST run's newest checkpoint, so it must follow step 1: the repo's candidate .pt files are
#     legacy rsl-rl 3.x and are rejected on GA by design.
#   - Does NOT terminate the pod. --upload copies the report dir to r2:<bucket>/migration/ga_smoke_<utc stamp>/.
# Exit code: number of failed steps (0 = all passed), 2 = bad arguments.
set -uo pipefail
export TERM=xterm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Host gate + launch probe (docs/runpod.md "Pod hosts"). Both run before any download or Kit launch: a driver
# build that segfaults Isaac Sim stops the session here, while the pod still holds no data.
export CERBERUS_ON_POD=1
bash "$REPO_ROOT/scripts/pod_driver_gate.sh" || { echo "[pod] driver gate rejected this host; stopping" >&2; exit 2; }
bash "$REPO_ROOT/scripts/pod_launch_probe.sh" || { echo "[pod] launch probe failed on this host; stopping" >&2; exit 3; }
bash "$REPO_ROOT/scripts/pod_deadman.sh" || echo "[pod] WARNING: dead-man switch not armed" >&2
if [[ -f /etc/cerberus_env.sh ]]; then source /etc/cerberus_env.sh; fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# isaaclab.sh -p runs the Isaac python with the script's own dir on sys.path, not the repo root, so
# `import common` / `import policy` need this (same export as scripts/run_scene_admission_pod.sh).
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="data/reports/ga_smoke_$STAMP"
SKIP=""; UPLOAD=0; DRY=0
# Observed on the GA session: Kit launch ~10 s, Arena stock build 13 s + reset 32 s, mimic smoke ~23 s in-process,
# 100-frame play clip well under a minute. 600 s per attempt is ~6x headroom for a hang, not a slow run.
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-600}"; PLAY_TIMEOUT="${PLAY_TIMEOUT:-600}"; SPIKE_TIMEOUT="${SPIKE_TIMEOUT:-600}"
# Driver 595 crashes or hangs ~half of Kit launches (measured 2026-11-06: the spike crashed twice in a row while
# train and play launched first try). Four attempts makes that a non-issue; worst case is 4 x timeout per step.
ATTEMPTS="${ATTEMPTS:-4}"
# The spike step: a ground plane, not Arena's `stock` background. `--backend arena --scene stock` spawns the Galileo
# background, whose floor the robot sinks through (base below its own feet, 2026-11-06), so the standing check added to
# arena_spike_go2.py fails it by design -- a permanently red step is worse than no step. plain_usd `stock` is a real
# GroundPlaneCfg, so it checks "env builds, robot stands, 100 steps on GPU" with no scene dataset and no Arena.
# For real Arena coverage override e.g. SPIKE_ARGS="--backend arena --scene lightwheel:1:1" (needs the Lightwheel SDK
# download and exercises Arena's ResetBackgroundPhysics path, which reset_all exists for).
SPIKE_ARGS="${SPIKE_ARGS:---backend plain_usd --scene stock}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --skip) SKIP="$SKIP $2"; shift 2 ;;
        --upload) UPLOAD=1; shift ;;
        --dry-run) DRY=1; shift ;;
        *) echo "[ga_smoke] unknown argument: $1" >&2; exit 2 ;;
    esac
done
for s in $SKIP; do case "$s" in train|play|spike) ;; *) echo "[ga_smoke] bad --skip '$s' (train|play|spike)" >&2; exit 2 ;; esac; done

say() { echo "[ga_smoke $(date -u +%H:%M:%S)] $*"; }
skipped() { [[ " $SKIP " == *" $1 "* ]]; }
LAB="$ISAACLAB_PATH/isaaclab.sh"
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/summary.txt"
MARKER="$REPORT_DIR/.start_marker"; : > "$MARKER"
FAILS=0
record() { echo "$1 $2 $3" | tee -a "$SUMMARY"; [[ "$2" == PASS || "$2" == SKIP || "$2" == DRYRUN ]] || FAILS=$((FAILS + 1)); }

# run_isaac <name> <timeout_s> <attempts> -- cmd...   (retries on non-zero exit or timeout; logs to $REPORT_DIR/<name>.log)
run_isaac() {
    local name="$1" tmo="$2" tries="$3" rc=1 i; shift 4
    for ((i = 1; i <= tries; i++)); do
        say "$name attempt $i/$tries: $*"
        if [[ $DRY -eq 1 ]]; then return 0; fi
        timeout --signal=TERM --kill-after=30 "$tmo" "$@" > "$REPORT_DIR/$name.attempt$i.log" 2>&1
        rc=$?
        cp "$REPORT_DIR/$name.attempt$i.log" "$REPORT_DIR/$name.log"
        say "$name attempt $i exit=$rc (124 = timeout)"
        [[ $rc -eq 0 ]] && return 0
    done
    return "$rc"
}

newer_files() { find "$1" -type f -name "$2" -size +0 -newer "$MARKER" 2>/dev/null; }

if [[ $DRY -eq 0 && ! -x "$LAB" ]]; then say "ERROR: $LAB not found/executable (set ISAACLAB_PATH)"; exit 1; fi
if command -v python3 >/dev/null 2>&1; then PYTHON=python3
elif [[ -x "$ISAACLAB_PATH/_isaac_sim/python.sh" ]]; then PYTHON="$ISAACLAB_PATH/_isaac_sim/python.sh"
else PYTHON=""; fi
say "report dir: $REPORT_DIR   isaaclab: $LAB   python: ${PYTHON:-<none>}   dry-run: $DRY"
# the repo sync excludes .git, so take the commit from the caller when set (CERBERUS_SYNC_COMMIT)
{ echo "stamp $STAMP"; echo "git ${CERBERUS_SYNC_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null; } | tee "$REPORT_DIR/env.txt" >/dev/null

# ---- 1. train smoke ------------------------------------------------------------------------------------------
if skipped train; then record train SKIP "-"; else
    run_isaac train "$TRAIN_TIMEOUT" "$ATTEMPTS" -- "$LAB" -p policy/locomotion/train.py --config smoke_test --headless
    rc=$?
    if [[ $DRY -eq 1 ]]; then record train DRYRUN "-"; else
        ckpts="$(newer_files policy/locomotion/checkpoints 'model_*.pt')"
        if [[ $rc -eq 0 && -n "$ckpts" ]]; then record train PASS "$(echo "$ckpts" | tail -1)"
        else record train FAIL "exit=$rc new_checkpoints=$(echo -n "$ckpts" | grep -c .)"; fi
    fi
fi

# ---- 2. play + video + export, on the checkpoint step 1 just wrote ---------------------------------------------
if skipped play; then record play SKIP "-"; else
    export CERBERUS_ALLOW_BAD_DRIVER_RENDER=1   # the 595 guard blocks --video; rendering itself was fine on GA
    run_isaac play "$PLAY_TIMEOUT" "$ATTEMPTS" -- \
        "$LAB" -p policy/locomotion/play.py --headless --video --video_length 100 --num_envs 8
    rc=$?
    if [[ $DRY -eq 1 ]]; then record play DRYRUN "-"; else
        vids="$(newer_files policy/locomotion/checkpoints '*.mp4')"          # <run>/videos/play/
        onnx="$(newer_files policy/locomotion/checkpoints 'policy.onnx')"    # <run>/exported/
        jit="$(newer_files policy/locomotion/checkpoints 'policy.pt')"       # <run>/exported/
        # 124 (timeout) is acceptable here only WITH the artifacts: play.py breaks after --video_length steps, but
        # the trailing simulation_app.close() is known to hang (and then exits 0), so a kill after the clip is not a failure.
        if [[ ( $rc -eq 0 || $rc -eq 124 ) && -n "$vids" && -n "$onnx" && -n "$jit" ]]; then record play PASS "exit=$rc $(echo "$vids" | tail -1)"
        else record play FAIL "exit=$rc mp4=$(echo -n "$vids" | grep -c .) onnx=$(echo -n "$onnx" | grep -c .) jit=$(echo -n "$jit" | grep -c .)"; fi
    fi
fi

# ---- 3. Arena spike (stock scene) -- the PASS that previously had no artifact -----------------------------------
if skipped spike; then record spike SKIP "-"; else
    SPIKE_JSON="$REPORT_DIR/arena_spike_go2.json"
    # shellcheck disable=SC2086  # SPIKE_ARGS is intentionally word-split into separate flags
    run_isaac spike "$SPIKE_TIMEOUT" "$ATTEMPTS" -- "$LAB" -p scripts/arena_spike_go2.py $SPIKE_ARGS --num_envs 16 --steps 100 --headless --report "$SPIKE_JSON"
    rc=$?
    if [[ $DRY -eq 1 ]]; then record spike DRYRUN "-"; else
        if [[ -z "$PYTHON" ]]; then record spike FAIL "no python interpreter to read $SPIKE_JSON"
        elif [[ $rc -eq 0 && -s "$SPIKE_JSON" ]] && "$PYTHON" - "$SPIKE_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("passed") is True else 1)
PY
        then record spike PASS "$SPIKE_JSON"
        else record spike FAIL "exit=$rc report_bytes=$(stat -c %s "$SPIKE_JSON" 2>/dev/null || echo missing)"; fi
    fi
fi

# ---- artifacts ---------------------------------------------------------------------------------------------------
if [[ $DRY -eq 0 ]]; then
    mkdir -p "$REPORT_DIR/artifacts"
    newer_files policy/locomotion/checkpoints '*.mp4' | while read -r f; do cp "$f" "$REPORT_DIR/artifacts/"; done
fi
if [[ $UPLOAD -eq 1 ]]; then
    if [[ $DRY -eq 1 ]]; then say "would upload $REPORT_DIR -> r2:\$R2_BUCKET/migration/ga_smoke_$STAMP/"
    elif [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" && -n "${R2_BUCKET:-}" ]]; then
        RCLONE_CONFIG_CERBERUS_R2_TYPE=s3 RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare \
        RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
        RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
        RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
        RCLONE_CONFIG_CERBERUS_R2_ACL=private RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true \
            rclone copy "$REPORT_DIR" "cerberus_r2:${R2_BUCKET}/migration/ga_smoke_$STAMP" --min-size 1B \
            && say "uploaded to migration/ga_smoke_$STAMP/" || say "WARNING: upload failed; copy $REPORT_DIR off the pod manually"
    else say "WARNING: --upload needs R2_ACCOUNT_ID/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_BUCKET; copy $REPORT_DIR off the pod manually"; fi
fi

say "done: $FAILS failed step(s). Summary: $SUMMARY"
exit "$FAILS"
