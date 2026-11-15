#!/usr/bin/env bash
# Provider-neutral JOB entrypoint: runs inside the container, no SSH anywhere.
#
# Env (only these): R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET,
# CERBERUS_CODE_BUNDLE, CERBERUS_JOB_ID, CERBERUS_DEADLINE_MINUTES, CERBERUS_IMAGE_TAG.
# Optional: CONTAINER_ID, CONTAINER_API_KEY (Vast self-stop) or RUNPOD_POD_ID + RUNPOD_API_KEY (RunPod self-stop).
#
# Flow: rclone config -> fetch the code bundle -> clone -> gate -> physics (stand, scripted, nav N=10) ->
# optional replay -> sync outputs -> final status -> self-stop. A 30 s heartbeat uploads status.json and the log tail.
set -uo pipefail

REPO=/workspace/cerberus
JOB_ID="${CERBERUS_JOB_ID:-job-$(date -u +%Y%m%dT%H%M%SZ)}"
DEADLINE_MIN="${CERBERUS_DEADLINE_MINUTES:-85}"
IMG="${CERBERUS_IMAGE_TAG:-unknown}"
START=$(date +%s)
DEADLINE=$((START + DEADLINE_MIN * 60))
STAGE_FILE=/workspace/job_stage
LOG_FILE=/workspace/job.log
export CERBERUS_ON_POD=1
export PYTHONUNBUFFERED=1

mkdir -p /workspace /root
echo "bootstrap" > "$STAGE_FILE"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

rclone_env() {
    export RCLONE_CONFIG_CERBERUS_R2_TYPE=s3
    export RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare
    export RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:?}"
    export RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:?}"
    export RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID:?}.r2.cloudflarestorage.com"
    export RCLONE_CONFIG_CERBERUS_R2_ACL=private
    export RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true
}
rclone_env
REMOTE="cerberus_r2:${R2_BUCKET:?}"

stage() { echo "$1" > "$STAGE_FILE"; echo "[stage] $1 $(date -u +%H:%M:%SZ)"; }
remaining() { echo $(( DEADLINE - $(date +%s) )); }

launch_counts() {
    local csv="$REPO/outputs/launch_probe.csv"
    [ -f "$csv" ] || { echo "0 0"; return; }
    python3 - "$csv" <<'PY' 2>/dev/null || echo "0 0"
import csv, sys
ok = total = 0
for row in csv.DictReader(open(sys.argv[1], newline="")):
    if row.get("stage") == "launch":
        total += 1
        ok += int(row.get("verdict") == "STARTED")
print(ok, total)
PY
}

status_json() {
    local stage_name phase detail ep rem ok total elapsed
    stage_name=$(cat "$STAGE_FILE" 2>/dev/null || echo unknown)
    read -r ok total <<< "$(launch_counts)"
    elapsed=$(( $(date +%s) - START ))
    rem=$(remaining)
    python3 - "$JOB_ID" "$stage_name" "$elapsed" "$rem" "$ok" "$total" "$IMG" "${CERBERUS_STATUS_EXTRA:-}" <<'PY'
import json, os, sys, time
job, stage_name, elapsed, rem, ok, total, img, extra = sys.argv[1:9]
status = {"job_id": job, "stage": stage_name, "elapsed_s": int(elapsed),
          "deadline_remaining_s": int(rem), "launches_ok": int(ok), "launches_total": int(total),
          "image": img, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "provider": "runpod" if os.environ.get("RUNPOD_POD_ID") else "vast",
          "self_stop_ok": (open("/workspace/self_stop_ok").read().strip() == "true"
                           if os.path.exists("/workspace/self_stop_ok") else None),
          "pid": os.getpid()}
rate = os.environ.get("CERBERUS_HOURLY_RATE")
status["spend_estimate_usd"] = round(int(elapsed) / 3600 * float(rate), 3) if rate else None
if extra:
    status["detail"] = extra
print(json.dumps(status))
PY
}

heartbeat() {
    while true; do
        status_json > /workspace/job_status.json
        rclone copyto /workspace/job_status.json "$REMOTE/jobs/$JOB_ID/status.json" --no-check-dest >/dev/null 2>&1
        tail -c 8000 "$LOG_FILE" > /workspace/job_log_tail.txt 2>/dev/null
        rclone copyto /workspace/job_log_tail.txt "$REMOTE/jobs/$JOB_ID/log_tail.txt" --no-check-dest >/dev/null 2>&1
        sleep 30
    done
}

self_test() {
    # Harmless authenticated READ on the same API path the stop uses, so self-stop_ok is not a guess.
    local status=""
    if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
        runpodctl get pod "$RUNPOD_POD_ID" >/dev/null 2>&1 && status=200
    fi
    if [ -z "$status" ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
        status=$(curl -s -o /dev/null -w '%{http_code}'             -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Accept: application/json"             -H "User-Agent: cerberus-pod-tools/1.0" \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" || true)
    fi
    case "$status" in 2*) echo "[self_test] authenticated read -> HTTP $status"; echo true > /workspace/self_stop_ok ;;
        *) echo "[self_test] authenticated read -> HTTP ${status:-none}" >&2; echo false > /workspace/self_stop_ok ;;
    esac
}

self_stop() {
    # RunPod job pods: runpodctl when the binary is present, else the REST stop with the laptop's headers.
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
        local i status
        for i in 1 2 3; do
            if command -v runpodctl >/dev/null 2>&1; then
                if runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
                    echo "[self_stop] runpodctl stop ok"; return 0
                fi
                echo "[self_stop] runpodctl attempt $i failed"
            else
                status=$(curl -s -o /tmp/stop_body -w '%{http_code}' -X POST \
                    "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
                    -H "Authorization: Bearer ${RUNPOD_API_KEY:?}" -H "Content-Type: application/json" \
                    -H "Accept: application/json" -H "User-Agent: cerberus-pod-tools/1.0" \
                    -d '{}' || true)
                echo "[self_stop] REST attempt $i -> HTTP $status"
                case "$status" in 2*) return 0 ;; esac
            fi
            sleep 10
        done
        echo "false" > /workspace/self_stop_ok
        CERBERUS_STATUS_EXTRA="STOP_FAILED" status_json > /workspace/job_status.json
        rclone copyto /workspace/job_status.json "$REMOTE/jobs/$JOB_ID/status.json" >/dev/null 2>&1
        echo "[self_stop] STOP_FAILED; the laptop guard must act" >&2
        return 1
    fi
    # Vast uses the container credentials it injects
    if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
        local i status
        for i in 1 2 3; do
            status=$(curl -s -o /tmp/stop_body -w '%{http_code}' -X PUT \
                "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/" \
                -H "Authorization: Bearer ${CONTAINER_API_KEY}" \
                -H "Content-Type: application/json" \
                -H "Accept: application/json" \
                -H "User-Agent: cerberus-vast-tools/1.0" \
                -d '{"state":"stopped"}' || true)
            echo "[self_stop] attempt $i -> HTTP $status"
            case "$status" in 2*) return 0 ;; esac
            sleep 10
        done
        CERBERUS_STATUS_EXTRA="STOP_FAILED" status_json > /workspace/job_status.json
        rclone copyto /workspace/job_status.json "$REMOTE/jobs/$JOB_ID/status.json" >/dev/null 2>&1
        echo "[self_stop] STOP_FAILED; the laptop guard must act" >&2
        return 1
    fi
    echo "[self_stop] no provider credentials; leaving the container to the operator"
    return 0
}

finish() {
    local rc=$1
    stage "finishing"
    sync_outputs
    sleep 1   # let the tee drain so the uploaded tail includes the last lines
    CERBERUS_STATUS_EXTRA="rc=$rc" status_json > /workspace/job_status.json
    rclone copyto /workspace/job_status.json "$REMOTE/jobs/$JOB_ID/status.json" >/dev/null 2>&1
    tail -c 8000 "$LOG_FILE" > /workspace/job_log_tail.txt 2>/dev/null
    rclone copyto /workspace/job_log_tail.txt "$REMOTE/jobs/$JOB_ID/log_tail.txt" >/dev/null 2>&1
    self_stop || true
    exit "$rc"
}

sync_outputs() {
    local root
    root=$(cat /workspace/job_root 2>/dev/null || true)
    [ -n "$root" ] && [ -d "$root" ] && \
        rclone copy "$root" "$REMOTE/jobs/$JOB_ID/outputs/$(basename "$root")" --no-check-dest >/dev/null 2>&1
    [ -d "$REPO/outputs" ] && rclone copy "$REPO/outputs" "$REMOTE/jobs/$JOB_ID/repo_outputs" --no-check-dest >/dev/null 2>&1
    for f in assertions.json build_timings.json; do
        find "$REPO" -name "$f" -maxdepth 6 -print0 2>/dev/null | while IFS= read -r -d '' hit; do
            rclone copyto "$hit" "$REMOTE/jobs/$JOB_ID/files/$(basename "$(dirname "$hit")")/$f" >/dev/null 2>&1
        done
    done
}

trap 'finish 130' INT TERM
stage "bootstrap"
echo "[job] id=$JOB_ID image=$IMG deadline=${DEADLINE_MIN}min"
for v in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET CERBERUS_CODE_BUNDLE CERBERUS_JOB_ID; do
    [ -n "${!v:-}" ] || { echo "[job] MISSING ENV $v" >&2; finish 3; }
done
echo "[job] env ok (rclone remote $REMOTE)"
self_test

# A streaming image can start the entrypoint before its lower layers are materialized; physics must wait.
stage "wait-image"
LAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
for i in $(seq 1 60); do
    [ -f "$LAB_PATH/isaaclab.sh" ] && [ -d /isaac-sim ] && break
    [ $((i % 6)) -eq 0 ] && echo "[job] waiting for the image layers ($i/60): $LAB_PATH/isaaclab.sh"
    sleep 5
done
if [ ! -f "$LAB_PATH/isaaclab.sh" ]; then
    # RunPod defaults a 20 GB volume at /workspace, which shadows the image's /workspace (and its Lab).
    FOUND=$(find / -maxdepth 6 -name isaaclab.sh -not -path '/proc/*' -not -path '/sys/*' 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        LAB_PATH=$(dirname "$FOUND"); echo "[job] found Isaac Lab at $LAB_PATH"
    else
        echo "[job] Lab missing at $LAB_PATH; installing the pinned ref into the visible /workspace"
        rm -rf "$LAB_PATH"; mkdir -p "$LAB_PATH"
        git init "$LAB_PATH" >/dev/null 2>&1
        git -C "$LAB_PATH" remote add origin https://github.com/isaac-sim/IsaacLab.git
        git -C "$LAB_PATH" fetch --depth 1 origin e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4 >/dev/null 2>&1 \
            && git -C "$LAB_PATH" checkout --detach FETCH_HEAD >/dev/null 2>&1 \
            || { echo "[job] Lab fetch failed" >&2; finish 10; }
        ln -sfn /isaac-sim "$LAB_PATH/_isaac_sim"
        chmod +x "$LAB_PATH/isaaclab.sh"
        timeout 1500 bash -c "cd '$LAB_PATH' && ./isaaclab.sh --install all" > /workspace/lab_install.log 2>&1 \
            || { echo "[job] Lab install failed" >&2; tail -20 /workspace/lab_install.log; finish 10; }
        echo "[job] Lab installed from the pinned ref"
    fi
    export ISAACLAB_PATH="$LAB_PATH"
fi
echo "[job] lab present at $LAB_PATH (isaac-sim: $( [ -d /isaac-sim ] && echo yes || echo no ))"

heartbeat & HEARTBEAT_PID=$!

if [ ! -d "$REPO/.git" ]; then
    stage "fetch-code"
    echo "[job] fetching bundle $CERBERUS_CODE_BUNDLE"
    rclone copyto "$REMOTE/$CERBERUS_CODE_BUNDLE" /root/code.bundle || { echo "[job] bundle fetch failed" >&2; finish 4; }
    GIT_LFS_SKIP_SMUDGE=1 git clone /root/code.bundle "$REPO" >/dev/null 2>&1 || { echo "[job] clone failed" >&2; finish 4; }
    cd "$REPO" && GIT_LFS_SKIP_SMUDGE=1 git checkout -B h4-nav-in-scene origin/main >/dev/null 2>&1
fi
cd "$REPO" || finish 4
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "[job] code at $(git log --oneline -1)"

LAB="$LAB_PATH/isaaclab.sh"
S=MV7J6NIKTKJZ2AABAAAAADQ8_usd
CKPT=policy/locomotion/checkpoints/go2_locomotion_ga_v1_candidate.pt
ROOT="data/reports/h4_nav/job_$(date -u +%Y%m%dT%H%MZ)"
echo "$ROOT" > /workspace/job_root
mkdir -p "$ROOT"
if [ ! -f "$CKPT" ]; then
    BASE="${CERBERUS_CODE_BUNDLE%/code/*}"
    stage "fetch-inputs"
    echo "[job] fetching inputs from $BASE/inputs"
    rclone copy "$REMOTE/$BASE/inputs/checkpoints" "$REPO/policy/locomotion/checkpoints" --no-check-dest >/dev/null 2>&1
    rclone copy "$REMOTE/$BASE/inputs/data" "$REPO/data" --no-check-dest >/dev/null 2>&1
fi
[ -f "$CKPT" ] || { echo "[job] checkpoint missing" >&2; finish 5; }

stage "gate"
bash scripts/pod_driver_gate.sh || { echo "[job] gate rejected" >&2; finish 6; }

stage "launch-experiment"
for i in 1 2 3 4 5; do
    echo "[job] launch attempt $i/5"
    timeout 240 bash scripts/pod_launch_probe.sh --max-launches 1 || true
done
read -r LAUNCH_OK LAUNCH_TOTAL <<< "$(launch_counts)"
echo "[job] launches started $LAUNCH_OK/$LAUNCH_TOTAL"
if [ "$LAUNCH_OK" -lt 4 ]; then
    CERBERUS_STATUS_EXTRA="LAUNCH_LOW $LAUNCH_OK/$LAUNCH_TOTAL" status_json > /workspace/job_status.json
    rclone copyto /workspace/job_status.json "$REMOTE/jobs/$JOB_ID/status.json" >/dev/null 2>&1
    echo "[job] fewer than 4 of 5 launches started; stopping without physics" >&2
    finish 8
fi

stage "stand"
bash scripts/run_resilient.sh --name stand --max-restarts 5 -- \
  bash scripts/kit_retry.sh -- "$LAB" -p scripts/scene_eval.py --scene "grscenes:$S" --num_envs 1 --steps 200 \
  --checkpoint "$CKPT" --out-dir "$ROOT/stand_$S" --headless
python3 - "$ROOT/stand_$S/assertions.json" <<'PY' || { echo "[job] stand assertions failed" >&2; finish 7; }
import json, sys
d = json.load(open(sys.argv[1]))
print("assertions_passed", d.get("passed"))
for c in d.get("checks", []):
    print(" ", c.get("name"), c.get("passed"))
raise SystemExit(0 if d.get("passed") else 1)
PY

stage "scripted"
bash scripts/run_resilient.sh --name scripted --max-restarts 5 -- \
  bash scripts/kit_retry.sh -- "$LAB" -p scripts/scene_eval.py --scene "grscenes:$S" --num_envs 1 --steps 200 \
  --scripted --checkpoint "$CKPT" --out-dir "$ROOT/scripted_$S" --headless

stage "navigation"
bash scripts/run_resilient.sh --name nav --max-restarts 5 -- \
  bash scripts/kit_retry.sh -- "$LAB" -p scripts/nav_eval.py --scenes "grscenes:$S" --pairs 10 --seed 0 --num_envs 1 \
  --min-distance 3 --max-distance 10 --max-time 120 --checkpoint "$CKPT" --out-dir "$ROOT/nav_$S" --headless

if [ "$(remaining)" -ge 1200 ]; then
    stage "replay"
    bash scripts/run_resilient.sh --name replay --max-restarts 3 -- \
      bash scripts/kit_retry.sh -- "$LAB" -p scripts/nav_eval.py --scenes "grscenes:$S" --pairs 10 --seed 0 --num_envs 1 \
      --min-distance 3 --max-distance 10 --max-time 120 --checkpoint "$CKPT" --out-dir "$ROOT/nav_$S" --headless \
      --replay-only --replay-outcomes success,failure
else
    echo "[job] skipping replay: only $(remaining)s remain"
fi

stage "done"
finish 0
