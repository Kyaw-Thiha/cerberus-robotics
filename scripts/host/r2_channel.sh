#!/usr/bin/env bash
# R2 command channel, instance side. Started at boot by the Brev startup script (or a systemd unit), it turns a
# bucket prefix into a small job runner for hosts that have no SSH.
#
# Layout under jobs/<job>/ (only this prefix is ever read or written):
#   cmd/<seq>_<timeout_s>.cmd   commands, run in order, each with `timeout`
#   out/<seq>.out|.err|.rc      stdout, stderr, exit status
#   heartbeat.json              every 30 s: ts, job, current command, GPU summary, disk free
#   state/channel.json          written once at start: user, host, arch, uname, python
#
# A command runs with bash from $WORK_DIR, with the channel's environment. Long jobs should start themselves with
# nohup and log to a file; a later command uploads or tails that file.
#
# Requires: bash, coreutils, rclone, curl/unzip only at install time (the startup script installs rclone).
# Environment: JOB_ID, R2_BUCKET and the RCLONE_CONFIG_CERBERUS_R2_* variables (or R2_ENV pointing at a 0600 file).
set -uo pipefail

JOB_ID="${JOB_ID:?JOB_ID is required}"
R2_BUCKET="${R2_BUCKET:?R2_BUCKET is required}"
R2_ENV="${R2_ENV:-}"
if [[ -n "$R2_ENV" && -f "$R2_ENV" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$R2_ENV"
    set +a
fi
PREFIX="jobs/$JOB_ID"
WORK_DIR="${WORK_DIR:-/tmp/cerberus/work/$JOB_ID}"
STATE_DIR="${STATE_DIR:-/tmp/cerberus/state/$JOB_ID}"
POLL_S="${POLL_S:-5}"
HB_S="${HB_S:-30}"
# The loop's arithmetic is integer-only; a fractional heartbeat interval would break it.
case "$HB_S" in *[!0-9]*|"") HB_S=30 ;; esac
mkdir -p /tmp/cerberus "$WORK_DIR" "$STATE_DIR" "$STATE_DIR/cmd"
PROCESSED="$STATE_DIR/processed"
touch "$PROCESSED"

# One channel per host: a second copy would run commands twice.
exec 9>"$STATE_DIR/channel.lock"
if ! flock -n 9; then
    echo "r2_channel: another channel holds $STATE_DIR/channel.lock; exiting" >&2
    exit 0
fi

R() { rclone --s3-no-check-bucket --no-traverse "$@"; }
REMOTE="cerberus_r2:$R2_BUCKET/$PREFIX"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

json_escape() { printf '%s' "$1" | tr '\n' ' ' | sed 's/\\/\\\\/g; s/"/\\"/g' | cut -c1-300; }

write_heartbeat() {
    local current="$1" ts gpu disk tmp
    ts="$(date -u +%FT%TZ)"
    gpu="none"
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -3 | tr '\n' ';' | cut -c1-200)"
        [[ -z "$gpu" ]] && gpu="nvidia-smi-failed"
    fi
    disk="$(df -Pk "$WORK_DIR" 2>/dev/null | awk 'NR==2 {printf "%d MB free", $4/1024}')"
    tmp="/tmp/cerberus/heartbeat.json"
    printf '{"ts":"%s","job":"%s","current":"%s","user":"%s","host":"%s","arch":"%s","gpu":"%s","disk":"%s"}\n' \
        "$ts" "$JOB_ID" "$(json_escape "$current")" "$(id -un 2>/dev/null)" "$(hostname)" "$(uname -m)" \
        "$(json_escape "$gpu")" "$(json_escape "$disk")" > "$tmp"
    R copyto "$tmp" "$REMOTE/heartbeat.json" >/dev/null 2>&1
}

write_state() {
    local tmp="/tmp/cerberus/channel.json"
    printf '{"ts":"%s","job":"%s","user":"%s","host":"%s","arch":"%s","uname":"%s","python":"%s","cwd":"%s"}\n' \
        "$(date -u +%FT%TZ)" "$JOB_ID" "$(id -un 2>/dev/null)" "$(hostname)" "$(uname -m)" \
        "$(json_escape "$(uname -a)")" "$(command -v python3 || echo none)" "$WORK_DIR" > "$tmp"
    R copyto "$tmp" "$REMOTE/state/channel.json" >/dev/null 2>&1
}

install_rclone_if_missing() {
    command -v rclone >/dev/null 2>&1 && return 0
    local arch target
    case "$(uname -m)" in
        aarch64|arm64) arch=arm64 ;;
        x86_64|amd64) arch=amd64 ;;
        *) arch=amd64 ;;
    esac
    if [[ "$(id -u)" == "0" ]]; then target=/usr/local/bin; else target="$HOME/.local/bin"; fi
    mkdir -p "$target" /tmp/rclone-install
    curl -fsSL -o /tmp/rclone-install/rclone.zip "https://downloads.rclone.org/rclone-current-linux-$arch.zip" || return 1
    ( cd /tmp/rclone-install && unzip -q -o rclone.zip ) || return 1
    local bin
    bin="$(find /tmp/rclone-install -type f -name rclone | head -1)"
    [[ -n "$bin" ]] || return 1
    install -m755 "$bin" "$target/rclone"
    export PATH="$target:$PATH"
}

install_rclone_if_missing || { log "rclone install failed"; }
write_state
write_heartbeat "starting"
log "channel up: job=$JOB_ID user=$(id -un) arch=$(uname -m) rclone=$(command -v rclone || echo missing)"
last_hb=0
while true; do
    now="$(date +%s)"
    if (( now - last_hb >= HB_S )); then
        write_heartbeat ""
        last_hb=$now
    fi
    names="$(R lsf "$REMOTE/cmd/" --files-only --max-depth 1 2>/dev/null | sort)"
    for name in $names; do
        [[ "$name" == *.cmd ]] || continue
        if grep -qxF "$name" "$PROCESSED" 2>/dev/null; then continue; fi
        base="${name%.cmd}"
        seq_no="${base%%_*}"
        # <seq>_<timeout>.cmd -> timeout; default 1800 s
        limit="${base##*_}"
        [[ "$limit" =~ ^[0-9]+$ ]] || limit=1800
        local_cmd="$STATE_DIR/cmd/$name"
        if ! R copyto "$REMOTE/cmd/$name" "$local_cmd" >/dev/null 2>&1; then
            log "could not fetch $name"
            continue
        fi
        log "run $name timeout=${limit}s"
        write_heartbeat "running $name"
        : > "$STATE_DIR/out.tmp"
        : > "$STATE_DIR/err.tmp"
        ( cd "$WORK_DIR" && timeout "$limit" bash "$local_cmd" ) > "$STATE_DIR/out.tmp" 2> "$STATE_DIR/err.tmp"
        rc=$?
        R copyto "$STATE_DIR/out.tmp" "$REMOTE/out/$seq_no.out" >/dev/null 2>&1
        R copyto "$STATE_DIR/err.tmp" "$REMOTE/out/$seq_no.err" >/dev/null 2>&1
        printf '%s\n' "$rc" > "$STATE_DIR/rc.tmp"
        R copyto "$STATE_DIR/rc.tmp" "$REMOTE/out/$seq_no.rc" >/dev/null 2>&1
        echo "$name" >> "$PROCESSED"
        log "done $name rc=$rc"
        write_heartbeat ""
        last_hb="$(date +%s)"
    done
    sleep "$POLL_S"
done
