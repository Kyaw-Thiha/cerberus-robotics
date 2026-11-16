#!/usr/bin/env bash
# Kit-less Newton spike, phase 5: the retrain. Pre-check, retrain at 4096 envs, export, eval, walking clips.
# One boot, no restarts; the pod stops itself at the end.
set -uo pipefail

REPO=/workspace/cerberus
LAB=/workspace/isaaclab
VENV=/opt/labenv
JOB_ID="${CERBERUS_JOB_ID:-newton5b-$(date -u +%Y%m%dT%H%M%SZ)}"
DEADLINE_MIN="${CERBERUS_DEADLINE_MINUTES:-45}"
IMG="${CERBERUS_IMAGE_TAG:-unknown}"
LAB_REF="${CERBERUS_LAB_REF:-v3.0.0}"
START=$(date +%s)
CREATED_EPOCH="${CERBERUS_CREATED_EPOCH:-$START}"
DEADLINE=$((START + DEADLINE_MIN * 60))
STAGE_FILE=/workspace/job_stage
LOG_FILE=/workspace/job.log
OUT=/workspace/newton_out
LOGS=/workspace/spike_logs
PHYSX_CKPT="$REPO/policy/locomotion/checkpoints/unitree_go2_rough/2026-11-07_07-29-40/model_1499.pt"
NEW_CKPT="$OUT/candidate/go2_newton_locomotion_v1_candidate.pt"
export CERBERUS_ON_POD=1 PYTHONUNBUFFERED=1

mkdir -p /workspace "$OUT" "$LOGS" /root
echo bootstrap > "$STAGE_FILE"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

stage() { echo "$1" > "$STAGE_FILE"; cd "$REPO" 2>/dev/null || true; echo "[stage] $1 $(date -u +%H:%M:%SZ) remaining=$(( (DEADLINE - $(date +%s)) / 60 ))min"; }
remaining_min() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }

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
BASE="$REMOTE/jobs/$JOB_ID"

status_json() {
    local stage_name elapsed rem
    stage_name=$(cat "$STAGE_FILE" 2>/dev/null || echo unknown)
    elapsed=$(( $(date +%s) - START ))
    rem=$(( DEADLINE - $(date +%s) ))
    python3 - "$JOB_ID" "$stage_name" "$elapsed" "$rem" "$IMG" <<'PY'
import json, os, sys, time
job, stage_name, elapsed, rem, img = sys.argv[1:6]
print(json.dumps({"job_id": job, "stage": stage_name, "elapsed_s": int(elapsed),
                  "deadline_remaining_s": int(rem), "image": img, "provider": "runpod",
                  "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "self_stop_ok": (open("/workspace/self_stop_ok").read().strip() == "true"
                                   if os.path.exists("/workspace/self_stop_ok") else None), "pid": os.getpid()}))
PY
}

heartbeat() {
    while true; do
        status_json > /workspace/job_status.json
        rclone copyto /workspace/job_status.json "$BASE/status.json" --no-check-dest >/dev/null 2>&1
        tail -c 12000 "$LOG_FILE" > /workspace/job_log_tail.txt 2>/dev/null
        rclone copyto /workspace/job_log_tail.txt "$BASE/log_tail.txt" --no-check-dest >/dev/null 2>&1
        sleep 30
    done
}

self_stop() {
    [ -n "${RUNPOD_POD_ID:-}" ] || return 0
    local i status
    for i in 1 2 3; do
        status=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY:?}" -H "Content-Type: application/json" \
            -H "Accept: application/json" -H "User-Agent: cerberus-pod-tools/1.0" -d '{}' || true)
        echo "[self_stop] REST attempt $i -> HTTP $status"
        case "$status" in 2*) return 0 ;; esac
        sleep 10
    done
    echo false > /workspace/self_stop_ok
    return 1
}

sync_outputs() {
    rclone copy "$OUT" "$BASE/outputs" --no-check-dest >/dev/null 2>&1 || true
    rclone copy "$LOGS" "$BASE/logs" --no-check-dest >/dev/null 2>&1 || true
    rclone copyto /workspace/job.log "$BASE/job.log" >/dev/null 2>&1 || true
}

finish() {
    local rc=$1
    stage finishing
    # the checkpoint and its siblings go under checkpoints/, never into the repo
    [ -d "$OUT/candidate" ] && rclone copy "$OUT/candidate" "$BASE/checkpoints" --no-check-dest >/dev/null 2>&1 || true

    sync_outputs
    status_json > /workspace/job_status.json
    rclone copyto /workspace/job_status.json "$BASE/status.json" >/dev/null 2>&1
    self_stop || true
    exit "$rc"
}
trap 'finish 130' INT TERM

stage bootstrap
echo "[job] id=$JOB_ID image=$IMG deadline=${DEADLINE_MIN}min"
for v in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET CERBERUS_CODE_BUNDLE CERBERUS_JOB_ID; do
    [ -n "${!v:-}" ] || { echo "[job] MISSING ENV $v" >&2; finish 3; }
done

stage deps
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends build-essential ca-certificates git curl unzip cmake \
    libglib2.0-0 libegl1 libgl1 libopengl0 libgomp1 libvulkan1 libx11-6 libxext6 libxrender1 \
    python3 python3-dev >/dev/null 2>&1
curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Accept: application/json" -H "User-Agent: cerberus-pod-tools/1.0" \
    "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" > /workspace/self_test_http 2>/dev/null || true
[ "$(cat /workspace/self_test_http)" = "200" ] && echo true > /workspace/self_stop_ok || echo false > /workspace/self_stop_ok
heartbeat & HEARTBEAT_PID=$!
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
fi
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --break-system-packages -q uv >/dev/null 2>&1 || true
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv || { echo "[deps] uv missing" >&2; finish 6; }
if ! rclone version 2>/dev/null | head -1 | grep -q 'v1.7'; then
    curl -fsSL -o /tmp/rclone.zip https://downloads.rclone.org/v1.75.1/rclone-v1.75.1-linux-amd64.zip \
        && unzip -q -o /tmp/rclone.zip -d /tmp/rclone \
        && install -m 0755 /tmp/rclone/rclone-v1.75.1-linux-amd64/rclone /usr/local/bin/rclone
fi

stage fetch-code
rclone copyto "$REMOTE/$CERBERUS_CODE_BUNDLE" /root/code.bundle || { echo "[job] bundle fetch failed" >&2; finish 4; }
ls -la /root/code.bundle
rm -rf "$REPO"
git init -q "$REPO" || { echo "[job] git init failed" >&2; finish 4; }
git -C "$REPO" fetch -f /root/code.bundle newton-spike:refs/heads/newton-spike \
    || { echo "[job] git fetch from the bundle failed" >&2; git -C "$REPO" bundle verify /root/code.bundle >&2 || true; finish 4; }
git -C "$REPO" checkout -f newton-spike || { echo "[job] checkout failed" >&2; finish 4; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "[job] code at $(git log --oneline -1)"
INPUT_BASE="${CERBERUS_CODE_BUNDLE%/code/*}"
rclone copy "$REMOTE/$INPUT_BASE/inputs/checkpoint" "$(dirname "$PHYSX_CKPT")" --no-check-dest >/dev/null 2>&1
[ -f "$PHYSX_CKPT" ] || { echo "[job] PhysX checkpoint missing" >&2; finish 5; }

stage install-lab
t0=$(date +%s)
git clone --depth 1 --branch "$LAB_REF" https://github.com/isaac-sim/IsaacLab.git "$LAB" \
    || { echo "[job] Isaac Lab clone failed" >&2; finish 6; }
export VIRTUAL_ENV="$VENV" UV_PROJECT_ENVIRONMENT="$VENV" UV_PYTHON=/usr/bin/python3
export UV_PYTHON_PREFERENCE=only-system UV_NO_CACHE=1 PIP_RETRIES=12 UV_HTTP_RETRIES=12
cd "$LAB" || finish 6
uv venv --seed "$VENV" >/dev/null 2>&1 || true
timeout --signal=KILL 1800 uv sync --frozen --inexact --extra video --extra importers > "$LOGS/uv_sync.log" 2>&1 \
    || { echo "[lab] uv sync failed" >&2; tail -20 "$LOGS/uv_sync.log" >&2; finish 6; }
INSTALL_S=$(( $(date +%s) - t0 ))
echo "[lab] install ${INSTALL_S}s"
python3 - "$INSTALL_S" "$CREATED_EPOCH" "$START" "$LAB_REF" > "$OUT/host.json" <<'PY'
import json, subprocess, sys
install_s, created, started, lab_ref = sys.argv[1:5]
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""
print(json.dumps({"job_started_epoch": int(started), "created_epoch": int(created),
                  "image_pull_and_boot_s": max(0, int(started) - int(created)), "lab_install_s": int(install_s),
                  "lab_ref": lab_ref,
                  "driver_and_gpu": sh("nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader"),
                  "disk_avail": sh("df -h /workspace | tail -1")}, indent=2))
PY
cat "$OUT/host.json"

# ---------------------------------------------------------------------------------------------
# promote the checkpoint the cut run left behind, then evaluate and film it
# ---------------------------------------------------------------------------------------------
SRC="$REMOTE/jobs/newton-20261116T061409Z/outputs/train/train/go2_newton_locomotion_v1_candidate"
stage fetch-run
mkdir -p "$OUT/run" "$OUT/candidate"
rclone copy "$SRC" "$OUT/run" --no-check-dest --include "model_1250.pt" --include "events.*"     --include "params/*.yaml" >/dev/null 2>&1 || true
ls -la "$OUT/run" | head
[ -f "$OUT/run/model_1250.pt" ] || { echo "[stage] model_1250.pt missing from R2" >&2; finish 7; }

stage export
timeout --signal=KILL 900 "$VENV/bin/python" -m spikes.newton.export_candidate \
    --checkpoint "$OUT/run/model_1250.pt" --run-dir "$OUT/run" --out "$OUT/candidate" \
    --iterations 1250 --num_envs 64 --viz none --device cuda:0 > "$LOGS/export.log" 2>&1
echo "[export] rc=$?"
tail -8 "$LOGS/export.log"
sync_outputs
NEW_CKPT="$OUT/candidate/go2_newton_locomotion_v1_candidate.pt"
rclone copy "$OUT/candidate" "$BASE/checkpoints" --no-check-dest >/dev/null 2>&1 || true

stage eval-newton
[ -f "$NEW_CKPT" ] || { echo "[eval] no candidate" >&2; finish 7; }
timeout --signal=KILL 900 "$VENV/bin/python" -m spikes.newton.transfer_eval \
    --policy "$NEW_CKPT" --out "$OUT/eval_newton" --joint_ordering none \
    --levels 0,1,2,3,4,5 --envs_per_level 64 --steps 1000 --vx 0.8 --viz none --device cuda:0 \
    --use_newton_actuators true --global_world_only true > "$LOGS/eval_newton.log" 2>&1
echo "[eval] rc=$?"
tail -10 "$LOGS/eval_newton.log"
sync_outputs

stage eval-push-sweep
if [ "${CERBERUS_SKIP_PUSH:-0}" = "1" ]; then
    echo "[push] skipped (CERBERUS_SKIP_PUSH=1)"
else
timeout --signal=KILL 600 "$VENV/bin/python" -m spikes.newton.run_push_recovery \
    --task "Isaac-Velocity-Rough-Unitree-Go2-Cerberus-Newton-Play-v0" --platform go2 --terrain rough \
    --headless --checkpoint "$NEW_CKPT" --magnitudes 0,80 --terrain_levels 0,4 --trials_per_cell 20 \
    --output "$OUT/push_sweep/eval.csv" > "$LOGS/push_sweep.log" 2>&1
echo "[push] rc=$?"
tail -12 "$LOGS/push_sweep.log"
echo "$?" > "$OUT/push_sweep_rc.txt" 2>/dev/null || true
fi
sync_outputs

stage walk-clips
for LVL in 0 3; do
    [ "$(remaining_min)" -ge 7 ] || { echo "[walk] out of time before level $LVL"; break; }
    timeout --signal=KILL 900 "$VENV/bin/python" -m spikes.newton.walk_clip \
        --policy "$NEW_CKPT" --level "$LVL" --out "$OUT/walk_level$LVL" --steps 1000 --vx 0.6 \
        --viz newton_gl --device cuda:0 > "$LOGS/walk_level$LVL.log" 2>&1
    echo "[walk] level=$LVL rc=$?"
    tail -6 "$LOGS/walk_level$LVL.log"
    CLIP=$(ls "$OUT"/walk_level"$LVL"/videos/*.mp4 2>/dev/null | head -1)
    if [ -n "${CLIP:-}" ]; then
        timeout --signal=KILL 300 "$VENV/bin/python" -m spikes.newton.clip_stills \
            --clip "$CLIP" --report "$OUT/walk_level$LVL/walk_clip.json" \
            --out "$OUT/walk_level$LVL/stills" >> "$LOGS/walk_level$LVL.log" 2>&1
        echo "[walk] stills level=$LVL rc=$?"
    fi
    sync_outputs
done

stage done
finish 0
