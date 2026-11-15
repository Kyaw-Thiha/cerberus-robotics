#!/usr/bin/env bash
# Kit-less Newton spike, phase 2: rest pose, actuator A/B, height-scan convention, scene asset sizes.
#
# Same job pattern as spikes/newton/newton_job.sh (no SSH; heartbeat to R2; self-stop), different stages and a
# bigger container disk. Nothing outside spikes/newton/ is touched.
#
# Env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, CERBERUS_CODE_BUNDLE,
#      CERBERUS_JOB_ID, CERBERUS_DEADLINE_MINUTES, CERBERUS_IMAGE_TAG, CERBERUS_CREATED_EPOCH,
#      RUNPOD_POD_ID, RUNPOD_API_KEY.
set -uo pipefail

REPO=/workspace/cerberus
LAB=/workspace/isaaclab
VENV=/opt/labenv
JOB_ID="${CERBERUS_JOB_ID:-newton2-$(date -u +%Y%m%dT%H%M%SZ)}"
DEADLINE_MIN="${CERBERUS_DEADLINE_MINUTES:-75}"
IMG="${CERBERUS_IMAGE_TAG:-unknown}"
LAB_REF="${CERBERUS_LAB_REF:-v3.0.0}"
START=$(date +%s)
CREATED_EPOCH="${CERBERUS_CREATED_EPOCH:-$START}"
DEADLINE=$((START + DEADLINE_MIN * 60))
STAGE_FILE=/workspace/job_stage
LOG_FILE=/workspace/job.log
OUT=/workspace/newton_out
LOGS=/workspace/spike_logs
SCENE_ID="${CERBERUS_SCENE_ID:-MV7J6NIKTKJZ2AABAAAAADQ8_usd}"
SCENE_DEST=/workspace/scene_lite
CKPT="$REPO/policy/locomotion/checkpoints/unitree_go2_rough/2026-11-07_07-29-40/model_1499.pt"
export CERBERUS_ON_POD=1 PYTHONUNBUFFERED=1

mkdir -p /workspace "$OUT" "$LOGS" /root
echo bootstrap > "$STAGE_FILE"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

stage() { echo "$1" > "$STAGE_FILE"; cd "$REPO" 2>/dev/null || true; echo "[stage] $1 $(date -u +%H:%M:%SZ) remaining=$(( (DEADLINE - $(date +%s)) / 60 ))min"; }
remaining() { echo $(( DEADLINE - $(date +%s) )); }
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
    local stage_name elapsed rem detail
    stage_name=$(cat "$STAGE_FILE" 2>/dev/null || echo unknown)
    elapsed=$(( $(date +%s) - START ))
    rem=$(remaining)
    detail=$(cat "$OUT/detail.txt" 2>/dev/null | tail -1)
    python3 - "$JOB_ID" "$stage_name" "$elapsed" "$rem" "$IMG" "$detail" <<'PY'
import json, os, sys, time
job, stage_name, elapsed, rem, img, detail = sys.argv[1:7]
status = {"job_id": job, "stage": stage_name, "elapsed_s": int(elapsed), "deadline_remaining_s": int(rem),
          "image": img, "provider": "runpod", "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "self_stop_ok": (open("/workspace/self_stop_ok").read().strip() == "true"
                           if os.path.exists("/workspace/self_stop_ok") else None), "pid": os.getpid()}
if detail:
    status["detail"] = detail
print(json.dumps(status))
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

self_test() {
    local status=""
    if [ -n "${RUNPOD_POD_ID:-}" ] && command -v curl >/dev/null 2>&1 && [ -n "${RUNPOD_API_KEY:-}" ]; then
        status=$(curl -s -o /dev/null -w '%{http_code}' \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Accept: application/json" \
            -H "User-Agent: cerberus-pod-tools/1.0" \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" || true)
    fi
    case "$status" in
        2*) echo "[self_test] authenticated read -> HTTP $status"; echo true > /workspace/self_stop_ok ;;
        *) echo "[self_test] authenticated read -> HTTP ${status:-none}" >&2; echo false > /workspace/self_stop_ok ;;
    esac
}

self_stop() {
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        echo "[self_stop] no RUNPOD_POD_ID; leaving the pod to the operator"
        return 0
    fi
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
    echo "false" > /workspace/self_stop_ok
    echo "STOP_FAILED" > "$OUT/detail.txt"
    status_json > /workspace/job_status.json
    rclone copyto /workspace/job_status.json "$BASE/status.json" >/dev/null 2>&1
    return 1
}

sync_outputs() {
    rclone copy "$OUT" "$BASE/outputs" --no-check-dest >/dev/null 2>&1 || true
    rclone copy "$LOGS" "$BASE/logs" --no-check-dest >/dev/null 2>&1 || true
    [ -d "$REPO/logs/rsl_rl" ] && rclone copy "$REPO/logs/rsl_rl" "$BASE/rl_logs" --no-check-dest \
        --include "**/model_*.pt" >/dev/null 2>&1
    rclone copyto /workspace/job.log "$BASE/job.log" >/dev/null 2>&1 || true
}

finish() {
    local rc=$1
    stage finishing
    sync_outputs
    echo "rc=$rc" > "$OUT/detail.txt"
    status_json > /workspace/job_status.json
    rclone copyto /workspace/job_status.json "$BASE/status.json" >/dev/null 2>&1
    self_stop || true
    exit "$rc"
}
trap 'finish 130' INT TERM

stage bootstrap
echo "[job] id=$JOB_ID image=$IMG lab_ref=$LAB_REF deadline=${DEADLINE_MIN}min"
for v in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET CERBERUS_CODE_BUNDLE CERBERUS_JOB_ID; do
    [ -n "${!v:-}" ] || { echo "[job] MISSING ENV $v" >&2; finish 3; }
done

stage deps
t0=$(date +%s)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends build-essential ca-certificates git curl unzip cmake \
    libglib2.0-0 libegl1 libgl1 libopengl0 libgomp1 libvulkan1 libx11-6 libxext6 libxrender1 \
    python3 python3-dev >/dev/null 2>&1
echo "[deps] apt done in $(( $(date +%s) - t0 ))s"
self_test
heartbeat & HEARTBEAT_PID=$!
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
fi
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --break-system-packages -q uv >/dev/null 2>&1 || true
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv || { echo "[deps] uv missing" >&2; finish 6; }
# the image's rclone 1.58 rejects the Cloudflare provider; install the pinned build
if ! rclone version 2>/dev/null | head -1 | grep -q 'v1.7'; then
    curl -fsSL -o /tmp/rclone.zip https://downloads.rclone.org/v1.75.1/rclone-v1.75.1-linux-amd64.zip \
        && unzip -q -o /tmp/rclone.zip -d /tmp/rclone \
        && install -m 0755 /tmp/rclone/rclone-v1.75.1-linux-amd64/rclone /usr/local/bin/rclone
fi
command -v rclone || { echo "[deps] rclone missing" >&2; finish 6; }

stage fetch-code
if [ ! -d "$REPO/.git" ]; then
    rclone copyto "$REMOTE/$CERBERUS_CODE_BUNDLE" /root/code.bundle || { echo "[job] bundle fetch failed" >&2; finish 4; }
    GIT_LFS_SKIP_SMUDGE=1 git clone /root/code.bundle "$REPO" >/dev/null 2>&1 || { echo "[job] clone failed" >&2; finish 4; }
    cd "$REPO" && git checkout newton-spike >/dev/null 2>&1
else
    rclone copyto "$REMOTE/$CERBERUS_CODE_BUNDLE" /root/code.bundle || { echo "[job] bundle re-fetch failed" >&2; finish 4; }
    git -C "$REPO" fetch -f /root/code.bundle newton-spike:refs/remotes/bundle/newton-spike >/dev/null 2>&1
    git -C "$REPO" reset --hard bundle/newton-spike >/dev/null 2>&1
fi
cd "$REPO" || finish 4
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "[job] code at $(git log --oneline -1)"
INPUT_BASE="${CERBERUS_CODE_BUNDLE%/code/*}"
rclone copy "$REMOTE/$INPUT_BASE/inputs/checkpoint" "$(dirname "$CKPT")" --no-check-dest >/dev/null 2>&1
[ -f "$CKPT" ] || { echo "[job] checkpoint missing at $CKPT" >&2; finish 5; }

stage install-lab
t0=$(date +%s)
if [ ! -d "$LAB/.git" ]; then
    git clone --depth 1 --branch "$LAB_REF" https://github.com/isaac-sim/IsaacLab.git "$LAB" \
        || { echo "[job] Isaac Lab clone failed" >&2; finish 6; }
fi
export VIRTUAL_ENV="$VENV" UV_PROJECT_ENVIRONMENT="$VENV" UV_PYTHON=/usr/bin/python3
export UV_PYTHON_PREFERENCE=only-system UV_NO_CACHE=1 PIP_RETRIES=12 UV_HTTP_RETRIES=12
cd "$LAB" || finish 6
uv venv --seed "$VENV" >/dev/null 2>&1 || true
if ! timeout --signal=KILL 1200 uv sync --frozen --inexact --extra video --extra importers > "$LOGS/uv_sync.log" 2>&1; then
    echo "[lab] uv sync failed after $(( $(date +%s) - t0 ))s" >&2
    tail -30 "$LOGS/uv_sync.log" >&2
    finish 6
fi
INSTALL_S=$(( $(date +%s) - t0 ))
echo "[lab] uv sync done in ${INSTALL_S}s"
"$VENV/bin/python" -c "import isaaclab, isaaclab_newton, newton, warp; print('imports ok')" \
    > "$LOGS/import_probe.log" 2>&1 || { echo "[lab] import probe failed" >&2; tail -20 "$LOGS/import_probe.log" >&2; finish 6; }
cat "$LOGS/import_probe.log"
python3 - "$INSTALL_S" "$CREATED_EPOCH" "$START" "$LAB_REF" > "$OUT/host.json" <<'PY'
import json, subprocess, sys
install_s, created, started, lab_ref = sys.argv[1:5]
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""
print(json.dumps({
    "job_started_epoch": int(started), "created_epoch": int(created),
    "image_pull_and_boot_s": max(0, int(started) - int(created)), "lab_install_s": int(install_s),
    "lab_ref": lab_ref,
    "driver_and_gpu": sh("nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader"),
    "hostname": sh("hostname"), "disk_avail": sh("df -h /workspace | tail -1"),
}, indent=2))
PY
cat "$OUT/host.json"

# ---------------------------------------------------------------------------------------------
# 1. rest pose: zero action (default joint targets) for 200 steps, both actuator paths
# ---------------------------------------------------------------------------------------------
stage rest-pose
for ACT in true false; do
    timeout --signal=KILL 600 "$VENV/bin/python" -m spikes.newton.rest_pose \
        --out "$OUT/rest_pose_$ACT" --num_envs 16 --steps 200 --viz none --device cuda:0 \
        --use_newton_actuators "$ACT" > "$LOGS/rest_pose_$ACT.log" 2>&1
    echo "[rest] actuators=$ACT rc=$?"
    tail -18 "$LOGS/rest_pose_$ACT.log"
    sync_outputs
done
echo "rest_pose done" > "$OUT/detail.txt"

# ---------------------------------------------------------------------------------------------
# 2. transfer A/B: the same rollout with and without the backend-native actuator path
# ---------------------------------------------------------------------------------------------
stage transfer-ab
"$VENV/bin/python" - "$REPO" > "$OUT/solver_params.json" <<'PY' || true
import json, sys
repo = sys.argv[1]
sys.path.insert(0, repo)
from policy.locomotion.core.platforms.go2.env_cfg import UnitreeGo2RoughCerberusEnvCfg_PLAY
cfg = UnitreeGo2RoughCerberusEnvCfg_PLAY()
preset = cfg.sim.physics
newton = getattr(preset, "newton_mjwarp", None)
solver = getattr(newton, "solver_cfg", None)
def flat(obj, prefix=""):
    out = {}
    for field in ("njmax", "nconmax", "cone", "impratio", "integrator", "use_mujoco_contacts", "max_triangle_pairs",
                  "num_substeps", "debug_mode"):
        value = getattr(obj, field, None) if obj is not None else None
        if value is not None:
            out[field] = value
    shape = getattr(newton, "default_shape_cfg", None) if newton is not None else None
    for field in ("margin", "ke", "kd"):
        value = getattr(shape, field, None) if shape is not None else None
        if value is not None:
            out[f"shape.{field}"] = value
    return out
print(json.dumps({"preset_container": type(preset).__name__, "solver_class": type(solver).__name__,
                  "stock_values": flat(solver), "note": "the stock Go2 Newton preset; nothing tuned"}, indent=2))
PY
cat "$OUT/solver_params.json" 2>/dev/null || true
for ACT in true false; do
    [ "$(remaining)" -gt 300 ] || { echo "[transfer] out of time before actuators=$ACT"; break; }
    timeout --signal=KILL 900 "$VENV/bin/python" -m spikes.newton.transfer_eval \
        --policy "$CKPT" --out "$OUT/transfer_$ACT" --joint_ordering none \
        --levels 0,1,2,3,4,5 --envs_per_level 64 --steps 1000 --vx 0.8 --viz none --device cuda:0 \
        --use_newton_actuators "$ACT" > "$LOGS/transfer_$ACT.log" 2>&1
    echo "[transfer] actuators=$ACT rc=$?"
    tail -12 "$LOGS/transfer_$ACT.log"
    sync_outputs
done
echo "transfer-ab done" > "$OUT/detail.txt"

# ---------------------------------------------------------------------------------------------
# 3. height-scan convention: installed source + flat-ground measurement
# ---------------------------------------------------------------------------------------------
stage height-scan
timeout --signal=KILL 600 "$VENV/bin/python" -m spikes.newton.height_scan_check \
    --out "$OUT/height_scan" --num_envs 4 --steps 30 --viz none --device cuda:0 \
    > "$LOGS/height_scan.log" 2>&1
echo "[scan] rc=$?"
tail -12 "$LOGS/height_scan.log"
sync_outputs

# ---------------------------------------------------------------------------------------------
# 4. scene payload sizes (R2 side, no download) and a geometry-only fetch
# ---------------------------------------------------------------------------------------------
stage scene-sizes
timeout --signal=KILL 600 "$VENV/bin/python" -m spikes.newton.scene_assets --list --out "$OUT/scene_assets" \
    > "$LOGS/scene_sizes.log" 2>&1
echo "[disk] list rc=$?"
tail -25 "$LOGS/scene_sizes.log"
sync_outputs

stage scene-fetch-lite
if [ "$(remaining_min)" -ge 12 ]; then
    timeout --signal=KILL 1200 "$VENV/bin/python" -m spikes.newton.scene_assets --fetch-lite \
        --scene-id "$SCENE_ID" --dest "$SCENE_DEST" --out "$OUT/scene_assets" \
        > "$LOGS/scene_fetch_lite.log" 2>&1
    echo "[disk] lite fetch rc=$?"
    tail -20 "$LOGS/scene_fetch_lite.log"
    sync_outputs
    USD=$(ls "$SCENE_DEST"/scenes/"$SCENE_ID"/start_result_navigation.usd 2>/dev/null || \
          ls "$SCENE_DEST"/scenes/"$SCENE_ID"/*.usd 2>/dev/null | head -1)
    if [ -n "${USD:-}" ] && [ -f "$USD" ]; then
        stage scene-parse
        timeout --signal=KILL 600 "$VENV/bin/python" -m spikes.newton.scene_load --probe-only \
            --usd "$USD" --out "$OUT/scene_probe" > "$LOGS/scene_parse.log" 2>&1
        echo "[scene] parse rc=$?"
        tail -20 "$LOGS/scene_parse.log"
        echo "$?" > "$OUT/scene_probe/parse_rc.txt"
        sync_outputs
        if [ "$(remaining_min)" -ge 8 ]; then
            stage scene-load
            timeout --signal=KILL 600 "$VENV/bin/python" -m spikes.newton.scene_load --usd "$USD" \
                --out "$OUT/scene_probe" --steps 200 --num_envs 1 --mdl-path "$SCENE_DEST/Materials" \
                --viz none --device cuda:0 > "$LOGS/scene_load.log" 2>&1
            echo "[scene] load rc=$?"
            tail -25 "$LOGS/scene_load.log"
            echo "$?" > "$OUT/scene_probe/load_rc.txt"
            sync_outputs
        fi
    else
        echo "[scene] no scene usd after the lite fetch" >&2
    fi
else
    echo "[scene] skipped the lite fetch: $(remaining_min)min left"
fi

stage done
finish 0
