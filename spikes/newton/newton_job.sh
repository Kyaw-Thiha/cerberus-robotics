#!/usr/bin/env bash
# Kit-less Newton spike: the pod-side job. Runs with no SSH, heartbeats to R2, and stops its own pod.
#
# This is the job pattern from scripts/job_entrypoint.sh, copied into spikes/newton/ (the existing
# entrypoint is untouched). What differs is everything after boot: a small CUDA base image, a kit-less
# Isaac Lab install from a pinned tag, and five stages that answer the spike's questions.
#
# Env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, CERBERUS_CODE_BUNDLE,
#      CERBERUS_JOB_ID, CERBERUS_DEADLINE_MINUTES, CERBERUS_IMAGE_TAG, CERBERUS_CREATED_EPOCH,
#      RUNPOD_POD_ID, RUNPOD_API_KEY. Nothing else is passed.
set -uo pipefail

REPO=/workspace/cerberus
LAB=/workspace/isaaclab
VENV=/opt/labenv
JOB_ID="${CERBERUS_JOB_ID:-newton-$(date -u +%Y%m%dT%H%M%SZ)}"
DEADLINE_MIN="${CERBERUS_DEADLINE_MINUTES:-105}"
IMG="${CERBERUS_IMAGE_TAG:-unknown}"
LAB_REF="${CERBERUS_LAB_REF:-v3.0.0}"
START=$(date +%s)
CONTAINER_START=$(date -u -d "@${CERBERUS_CREATED_EPOCH:-$START}" +%s 2>/dev/null || echo "$START")
DEADLINE=$((START + DEADLINE_MIN * 60))
STAGE_FILE=/workspace/job_stage
LOG_FILE=/workspace/job.log
OUT=/workspace/newton_out
LOGS=/workspace/spike_logs
STOCK_TASK="Isaac-Velocity-Rough-Unitree-Go2"
SCENE_ID="${CERBERUS_SCENE_ID:-MV7J6NIKTKJZ2AABAAAAADQ8_usd}"
SCENE_DEST="$REPO/data/scenes/grscenes/home"
CKPT="$REPO/policy/locomotion/checkpoints/unitree_go2_rough/2026-11-07_07-29-40/model_1499.pt"
export CERBERUS_ON_POD=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

mkdir -p /workspace "$OUT" "$LOGS" /root "$OUT/stage1" "$OUT/stage2"
echo bootstrap > "$STAGE_FILE"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

stage() { echo "$1" > "$STAGE_FILE"; echo "[stage] $1 $(date -u +%H:%M:%SZ) remaining=$(( (DEADLINE - $(date +%s)) / 60 ))min"; }
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
                           if os.path.exists("/workspace/self_stop_ok") else None),
          "pid": os.getpid()}
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
    if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
        runpodctl get pod "$RUNPOD_POD_ID" >/dev/null 2>&1 && status=200
    fi
    if [ -z "$status" ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
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
        if command -v runpodctl >/dev/null 2>&1 && runpodctl stop pod "$RUNPOD_POD_ID" >/dev/null 2>&1; then
            echo "[self_stop] runpodctl stop ok"; return 0
        fi
        status=$(curl -s -o /tmp/stop_body -w '%{http_code}' -X POST \
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
        --include "**/model_*.pt" --include "**/*.txt" --include "**/*.log" --include "**/*.mp4" \
        --include "**/events.*" >/dev/null 2>&1
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

# ---------------------------------------------------------------------------------------------
# Boot: deps, Isaac Lab kit-less, Cerberus code
# ---------------------------------------------------------------------------------------------
stage bootstrap
echo "[job] id=$JOB_ID image=$IMG lab_ref=$LAB_REF deadline=${DEADLINE_MIN}min"
for v in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET CERBERUS_CODE_BUNDLE CERBERUS_JOB_ID; do
    [ -n "${!v:-}" ] || { echo "[job] MISSING ENV $v" >&2; finish 3; }
done
self_test
heartbeat & HEARTBEAT_PID=$!

stage deps
t0=$(date +%s)
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends build-essential ca-certificates git curl unzip cmake \
    libglib2.0-0 libegl1 libgl1 libopengl0 libgomp1 libvulkan1 libx11-6 libxext6 libxrender1 \
    python3 python3-dev >/dev/null 2>&1
echo "[deps] apt done in $(( $(date +%s) - t0 ))s"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh >/dev/null 2>&1 || true
fi
export PATH="$HOME/.local/bin:$PATH"
command -v uv || { echo "[deps] uv missing" >&2; finish 6; }
if ! command -v rclone >/dev/null 2>&1; then
    curl -fsSL -o /tmp/rclone.zip https://downloads.rclone.org/v1.75.1/rclone-v1.75.1-linux-amd64.zip \
        && unzip -q /tmp/rclone.zip -d /tmp/rclone \
        && install -m 0755 /tmp/rclone/rclone-v1.75.1-linux-amd64/rclone /usr/local/bin/rclone
fi
command -v rclone || { echo "[deps] rclone missing" >&2; finish 6; }
python3 --version
uv --version

stage fetch-code
if [ ! -d "$REPO/.git" ]; then
    rclone copyto "$REMOTE/$CERBERUS_CODE_BUNDLE" /root/code.bundle || { echo "[job] bundle fetch failed" >&2; finish 4; }
    GIT_LFS_SKIP_SMUDGE=1 git clone /root/code.bundle "$REPO" >/dev/null 2>&1 || { echo "[job] clone failed" >&2; finish 4; }
    cd "$REPO" && git checkout newton-spike >/dev/null 2>&1
fi
cd "$REPO" || finish 4
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "[job] code at $(git log --oneline -1)"
INPUT_BASE="${CERBERUS_CODE_BUNDLE%/code/*}"
rclone copy "$REMOTE/$INPUT_BASE/inputs/checkpoint" "$(dirname "$CKPT")" --no-check-dest >/dev/null 2>&1
rclone copyto "$REMOTE/$INPUT_BASE/inputs/admission.json" "$REPO/data/scenes/grscenes/admission.json" >/dev/null 2>&1
[ -f "$CKPT" ] || { echo "[job] checkpoint missing at $CKPT" >&2; finish 5; }

stage install-lab
t0=$(date +%s)
if [ ! -d "$LAB/.git" ]; then
    git clone --depth 1 --branch "$LAB_REF" https://github.com/isaac-sim/IsaacLab.git "$LAB" \
        || { echo "[job] Isaac Lab clone failed" >&2; finish 6; }
fi
echo "[lab] $(git -C "$LAB" log --oneline -1)"
export VIRTUAL_ENV="$VENV" UV_PROJECT_ENVIRONMENT="$VENV" UV_PYTHON=/usr/bin/python3
export UV_PYTHON_PREFERENCE=only-system UV_NO_CACHE=1 PIP_RETRIES=12 UV_HTTP_RETRIES=12
cd "$LAB" || finish 6
uv venv --seed "$VENV" >/dev/null 2>&1 || true
# `--frozen` is load-bearing (uv would re-resolve a stale lock); `--inexact` keeps the seeded pip.
if ! timeout --signal=KILL 1500 uv sync --frozen --inexact --extra video --extra importers > "$LOGS/uv_sync.log" 2>&1; then
    echo "[lab] uv sync failed after $(( $(date +%s) - t0 ))s (see uv_sync.log)" >&2
    tail -40 "$LOGS/uv_sync.log" >&2
    finish 6
fi
INSTALL_S=$(( $(date +%s) - t0 ))
echo "[lab] uv sync done in ${INSTALL_S}s"
"$VENV/bin/python" - <<PY > "$OUT/versions.json"
import json, importlib.metadata as md, sys
out = {"python": sys.version.split()[0]}
for name in ("isaaclab", "isaaclab_newton", "newton", "warp-lang", "torch", "usd-exchange", "rsl-rl-lib"):
    try:
        out[name] = md.version(name)
    except md.PackageNotFoundError:
        out[name] = None
try:
    from pxr import Usd
    out["openusd"] = ".".join(str(v) for v in Usd.GetVersion())
except Exception as exc:
    out["openusd"] = f"unavailable: {exc}"
print(json.dumps(out, indent=2))
PY
cat "$OUT/versions.json"
python3 - "$OUT/versions.json" "$INSTALL_S" "$CONTAINER_START" "$START" "$LAB_REF" > "$OUT/host.json" <<'PY'
import json, sys, subprocess, time
versions, install_s, container_start, job_start, lab_ref = sys.argv[1:6]
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""
host = {"job_started_epoch": int(job_start), "container_started_epoch": int(container_start),
        "image_pull_and_boot_s": max(0, int(job_start) - int(container_start)),
        "lab_install_s": int(install_s), "lab_ref": lab_ref,
        "driver_and_gpu": sh("nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader"),
        "cuda_from_driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "nvidia_smi_head": sh("nvidia-smi | head -12"),
        "gpu_count": len(sh("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines()),
        "hostname": sh("hostname"), "uptime_s": sh("cat /proc/uptime | cut -d' ' -f1")}
host["versions"] = json.load(open(versions))
print(json.dumps(host, indent=2))
PY
cat "$OUT/host.json"
cp "$OUT/host.json" "$LOGS/host.json"

# ---------------------------------------------------------------------------------------------
# Stage 1: stock Go2 rough task on Newton, then 5 consecutive launches
# ---------------------------------------------------------------------------------------------
stage stage1-stock
"$VENV/bin/python" -m spikes.newton.gl_probe "$OUT/stage2/gl_probe.json" > "$LOGS/gl_probe.log" 2>&1 || true
t0=$(date +%s)
timeout --signal=KILL 1200 "$VENV/bin/isaaclab" train --rl_library rsl_rl --task "$STOCK_TASK" \
    --max_iterations 50 --viz none physics=newton_mjwarp env.scene.num_envs=128 \
    > "$LOGS/stage1_train.log" 2>&1
STAGE1_RC=$?
echo "[stage1] 50-iteration smoke rc=$STAGE1_RC in $(( $(date +%s) - t0 ))s"
tail -30 "$LOGS/stage1_train.log"
"$VENV/bin/python" - "$REPO" <<'PY' > "$OUT/stage1/smoke.json" || true
import glob, json, os, sys
import torch
repo = sys.argv[1]
runs = sorted(glob.glob(os.path.join(repo, "logs/rsl_rl/*/*/model_*.pt")))
data = {"checkpoints": runs[-3:]}
if runs:
    ckpt = torch.load(runs[-1], map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("actor_state_dict") or {}
    bad = [k for k, v in state.items() if hasattr(v, "isfinite") and not bool(torch.isfinite(v).all())]
    data["latest"] = runs[-1]
    data["params_finite"] = not bad
    data["nonfinite_keys"] = bad[:5]
    data["iteration"] = ckpt.get("iter")
print(json.dumps(data, indent=2))
PY
cp "$OUT/stage1/smoke.json" "$LOGS/" 2>/dev/null || true
python3 - "$LOGS/stage1_train.log" <<'PY' >> "$OUT/stage1/smoke.txt" || true
import sys
text = open(sys.argv[1], errors="replace").read()
print("nan_in_log", "nan" in text.lower())
print("lines", text.count("\n"))
print("last_lines")
print("\n".join(text.strip().splitlines()[-12:]))
PY
cat "$OUT/stage1/smoke.txt" 2>/dev/null

: > "$OUT/stage1/launches.csv"
echo "attempt,rc,seconds,started" >> "$OUT/stage1/launches.csv"
for i in 1 2 3 4 5; do
    [ "$(remaining)" -gt 600 ] || { echo "[stage1] out of time for launch $i"; break; }
    t0=$(date +%s)
    timeout --signal=KILL 900 "$VENV/bin/isaaclab" train --rl_library rsl_rl --task "$STOCK_TASK" \
        --max_iterations 2 --viz none physics=newton_mjwarp env.scene.num_envs=128 \
        > "$LOGS/stage1_launch_$i.log" 2>&1
    rc=$?
    started=0
    [ "$rc" -eq 0 ] && started=1
    echo "$i,$rc,$(( $(date +%s) - t0 )),$started" >> "$OUT/stage1/launches.csv"
    echo "[stage1] launch $i rc=$rc started=$started"
done
cat "$OUT/stage1/launches.csv"
echo "launches=$(awk -F, 'NR>1 && $4==1' "$OUT/stage1/launches.csv" | wc -l)/5" > "$OUT/detail.txt"
sync_outputs

# ---------------------------------------------------------------------------------------------
# Stage 2: --video through the Newton GL headless visualizer
# ---------------------------------------------------------------------------------------------
stage stage2-video
t0=$(date +%s)
timeout --signal=KILL 900 "$VENV/bin/isaaclab" train --rl_library rsl_rl --task "$STOCK_TASK" \
    --max_iterations 8 --viz newton_gl --video --video_length 80 --video_interval 0 \
    physics=newton_mjwarp env.scene.num_envs=16 > "$LOGS/stage2_video.log" 2>&1
VIDEO_RC=$?
echo "[stage2] --video rc=$VIDEO_RC in $(( $(date +%s) - t0 ))s"
tail -25 "$LOGS/stage2_video.log"
mkdir -p "$OUT/stage2"
find "$REPO/logs" -name '*.mp4' -newermt '-30 minutes' -print0 2>/dev/null | while IFS= read -r -d '' f; do
    cp "$f" "$OUT/stage2/" 2>/dev/null || true
done
"$VENV/bin/python" - "$OUT/stage2" <<'PY' > "$OUT/stage2/video.json" 2>&1 || true
import glob, json, os, sys
out = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.mp4"))):
    entry = {"bytes": os.path.getsize(path)}
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(path)
        entry["duration_s"] = round(float(clip.duration), 2)
        entry["size"] = list(clip.size)
        entry["fps"] = clip.fps
        entry["frames"] = int(clip.duration * clip.fps) if clip.fps else None
        first = clip.get_frame(0)
        mid = clip.get_frame(max(0.0, clip.duration / 2))
        entry["frame0_mean"] = [round(float(v), 1) for v in first.mean(axis=(0, 1))]
        entry["mid_mean"] = [round(float(v), 1) for v in mid.mean(axis=(0, 1))]
        entry["mid_std"] = [round(float(v), 1) for v in mid.std(axis=(0, 1))]
        clip.close()
    except Exception as exc:
        entry["probe_error"] = f"{type(exc).__name__}: {exc}"
    out[os.path.basename(path)] = entry
print(json.dumps(out, indent=2))
PY
cat "$OUT/stage2/video.json" 2>/dev/null
sync_outputs

# ---------------------------------------------------------------------------------------------
# Stage 3: Cerberus task on Newton -- observation/action parity
# ---------------------------------------------------------------------------------------------
stage stage3-parity
timeout --signal=KILL 1200 "$VENV/bin/python" -m spikes.newton.parity_check \
    --out "$OUT/stage3" --num_envs 64 --steps 500 --joint_ordering none --viz none --device cuda:0 \
    > "$LOGS/stage3_parity.log" 2>&1
echo "[stage3] rc=$?"
tail -40 "$LOGS/stage3_parity.log"
sync_outputs

# ---------------------------------------------------------------------------------------------
# Stage 4: the PhysX-trained policy on Newton, per terrain level
# ---------------------------------------------------------------------------------------------
stage stage4-transfer
for ORDER in none contract; do
    [ "$(remaining)" -gt 420 ] || { echo "[stage4] out of time before ordering=$ORDER"; break; }
    timeout --signal=KILL 1500 "$VENV/bin/python" -m spikes.newton.transfer_eval \
        --policy "$CKPT" --out "$OUT/stage4_$ORDER" --joint_ordering "$ORDER" \
        --levels 0,1,2,3,4,5 --envs_per_level 64 --steps 1000 --vx 0.8 --viz none --device cuda:0 \
        > "$LOGS/stage4_$ORDER.log" 2>&1
    echo "[stage4] ordering=$ORDER rc=$?"
    tail -20 "$LOGS/stage4_$ORDER.log"
    sync_outputs
    RATE=$(python3 - "$OUT/stage4_$ORDER/transfer.json" <<'PY' 2>/dev/null || echo ""
import json, sys
try:
    print(json.load(open(sys.argv[1]))["rollout"]["fall_rate"])
except Exception:
    print("")
PY
)
    echo "[stage4] ordering=$ORDER fall_rate=$RATE"
    echo "stage4 fall_rate=$RATE ($ORDER)" > "$OUT/detail.txt"
    # `none` is the backend-order baseline; stop early only when transfer already looks healthy.
    if [ "$ORDER" = "none" ] && [ -n "$RATE" ] && python3 -c "import sys; sys.exit(0 if float('$RATE') < 0.5 else 1)" 2>/dev/null; then
        echo "[stage4] fall rate below 0.5 with backend order; contract-order run skipped"
        break
    fi
done

# ---------------------------------------------------------------------------------------------
# Stage 5: one GRScenes home, time-boxed
# ---------------------------------------------------------------------------------------------
stage stage5-scene-fetch
if [ "$(remaining_min)" -lt 18 ]; then
    echo "[stage5] skipped: $(remaining_min)min left"
else
    timeout --signal=KILL 1500 python3 scripts/fetch_grscenes_from_r2.py --scene-id "$SCENE_ID" \
        --dest "$SCENE_DEST" > "$LOGS/stage5_fetch.log" 2>&1
    echo "[stage5] fetch rc=$?"
    tail -5 "$LOGS/stage5_fetch.log"
    USD="$SCENE_DEST/scenes/$SCENE_ID/start_result_navigation.usd"
    [ -f "$USD" ] || USD=$(ls "$SCENE_DEST"/scenes/"$SCENE_ID"/*.usd 2>/dev/null | head -1)
    if [ -n "${USD:-}" ] && [ -f "$USD" ]; then
        ls -la "$USD"
        stage stage5-usd-parse
        timeout --signal=KILL 600 "$VENV/bin/python" -m spikes.newton.scene_load --probe-only \
            --usd "$USD" --out "$OUT/stage5" --mdl-path "$SCENE_DEST/Materials" > "$LOGS/stage5_parse.log" 2>&1
        echo "[stage5] usd parse rc=$?"
        tail -15 "$LOGS/stage5_parse.log"
        sync_outputs
        if [ "$(remaining_min)" -ge 12 ]; then
            stage stage5-scene-load
            timeout --signal=KILL 900 "$VENV/bin/python" -m spikes.newton.scene_load --usd "$USD" \
                --out "$OUT/stage5" --steps 200 --num_envs 1 --mdl-path "$SCENE_DEST/Materials" \
                --viz none --device cuda:0 > "$LOGS/stage5_load.log" 2>&1
            LOAD_RC=$?
            echo "[stage5] scene load rc=$LOAD_RC"
            tail -40 "$LOGS/stage5_load.log"
            echo "$LOAD_RC" > "$OUT/stage5/load_rc.txt"
            [ "$LOAD_RC" -eq 0 ] || echo "stage5 load_rc=$LOAD_RC FAILED" > "$OUT/detail.txt"
            sync_outputs
        fi
    else
        echo "[stage5] scene usd not found after fetch" >&2
    fi
fi

stage done
finish 0
