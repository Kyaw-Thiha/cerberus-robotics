#!/usr/bin/env bash
# Shared helpers for the provider-neutral host drivers. Sourced, never executed.
#
# The drivers used to assume a pod (fixed /workspace paths, a dead-man switch and a provider delete at the end).
# This library keeps only what is true on any host that meets docs/host_contract.md:
#
#   cerberus_defaults   paths from the environment, with the image's layout as the default
#   cerberus_load_env   source a credentials file into the driver's environment (never printed)
#   cerberus_gate       driver gate, then the launch probe (each can be skipped by an env var)
#   cerberus_say/record/elapsed   one-line progress and a PASS/FAIL/SKIP summary line
#   run_isaac           run a Kit command under a hard cap, a crash-marker detector and a log-growth stall detector;
#                       retries, which is what makes a flaky launch survivable
#
# The caller sets REPO_ROOT before sourcing this file.

# Inputs (all optional): CERBERUS_DATA_DIR, CERBERUS_RESULTS_DIR, ISAACLAB_PATH, CERBERUS_ENV_FILE.
cerberus_defaults() {
    : "${REPO_ROOT:?REPO_ROOT must be set before sourcing session_lib.sh}"
    DATA_DIR="${CERBERUS_DATA_DIR:-$HOME/cerberus-data}"
    RESULTS_DIR="${CERBERUS_RESULTS_DIR:-$HOME/cerberus-results}"
    SCENES_DIR="${CERBERUS_SCENES_DIR:-$DATA_DIR/scenes/grscenes/home}"
    # discovery expects <root>/<source>/.../scenes/<id>; the fetch destination is the mirrored home
    export CERBERUS_SCENES_DIR="${CERBERUS_SCENES_ROOT:-$DATA_DIR/scenes}"
    ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
    LAB="$ISAACLAB_PATH/isaaclab.sh"
    # Isaac's interpreter has isaaclab/rsl-rl/torch; the system python3 has none of them.
    PY="$ISAACLAB_PATH/_isaac_sim/python.sh"
    [[ -x "$PY" ]] || PY="python3"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export TERM="${TERM:-xterm}"
}

# Values are exported to children only; nothing here echoes them.
cerberus_load_env() {
    local file="${1:-${CERBERUS_ENV_FILE:-$REPO_ROOT/.env}}"
    if [[ -f "$file" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$file"
        set +a
        return 0
    fi
    return 1
}

cerberus_say() { echo "[$(basename "$0" .sh) $(date -u +%H:%M:%S)] $*"; }

# cerberus_record <name> <PASS|FAIL|SKIP|WARN> <detail>; needs SUMMARY and FAILS in the caller.
cerberus_record() {
    echo "$1 $2 $3" | tee -a "$SUMMARY"
    [[ "$2" == PASS || "$2" == SKIP || "$2" == WARN ]] || FAILS=$((FAILS + 1))
}

# cerberus_gate [--skip-driver] [--skip-probe]; both run before any download or Kit launch.
cerberus_gate() {
    local skip_driver=0 skip_probe=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-driver) skip_driver=1 ;;
            --skip-probe) skip_probe=1 ;;
            *) cerberus_say "cerberus_gate: unknown option $1" >&2; return 2 ;;
        esac
        shift
    done
    export CERBERUS_HOST_LOCATION="${CERBERUS_HOST_LOCATION:-$(hostname)}"
    if [[ $skip_driver -eq 0 && "${CERBERUS_SKIP_DRIVER_GATE:-0}" != "1" ]]; then
        bash "$REPO_ROOT/scripts/host/driver_gate.sh" || { cerberus_say "driver gate rejected this host; stopping" >&2; return 2; }
    fi
    if [[ $skip_probe -eq 0 && "${CERBERUS_SKIP_PROBE:-0}" != "1" ]]; then
        bash "$REPO_ROOT/scripts/host/launch_probe.sh" || { cerberus_say "launch probe failed on this host; stopping" >&2; return 3; }
    fi
    return 0
}

# run_isaac <name> <timeout_s> <attempts> -- cmd...
# A Kit launch can fail without exiting (it writes a crash report, then hangs) and a stuck launch shows no log
# growth. Both are launch failures to retry, not job verdicts; the cap still bounds the wall clock.
run_isaac() {
    local name="$1" tmo="$2" tries="$3" i rc=1 pid cap_pid watch_pid
    shift 4
    for ((i = 1; i <= tries; i++)); do
        local log="$REPORT_DIR/$name.attempt$i.log"
        local last=-1 size stalls=0
        cerberus_say "$name attempt $i/$tries"
        "$@" > "$log" 2>&1 &
        pid=$!
        ( sleep "$tmo"; kill -9 "$pid" 2>/dev/null ) >/dev/null 2>&1 &
        cap_pid=$!
        (
            for _ in $(seq 1 "$tmo"); do
                sleep 1
                grep -q "crash has occurred" "$log" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; break; }
                kill -0 "$pid" 2>/dev/null || break
                size=$(stat -c %s "$log" 2>/dev/null || echo 0)
                if [[ "$size" == "$last" ]]; then
                    stalls=$((stalls + 1))
                    if ((stalls >= ${STALL_S:-90})); then
                        echo "[watchdog] no log growth for ${STALL_S:-90}s: killing stalled launch" >> "$log"
                        kill -9 "$pid" 2>/dev/null
                        break
                    fi
                else
                    stalls=0
                    last=$size
                fi
            done
        ) >/dev/null 2>&1 &
        watch_pid=$!
        wait "$pid"
        rc=$?
        kill "$cap_pid" "$watch_pid" 2>/dev/null
        wait "$cap_pid" 2>/dev/null
        wait "$watch_pid" 2>/dev/null
        cp "$log" "$REPORT_DIR/$name.log"
        if grep -qE "crash has occurred|no log growth for" "$log" 2>/dev/null; then
            cerberus_say "$name attempt $i: launch failed (crash or stall, rc=$rc); retrying"
            rc=1
            continue
        fi
        cerberus_say "$name attempt $i exit=$rc (124/137 = cap)"
        [[ $rc -eq 0 ]] && return 0
    done
    return "$rc"
}

# rclone with the bucket remote defined entirely from R2_* in this process (no config file, no values printed).
cerberus_r2() {
    local missing=() v
    for v in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET; do
        [[ -n "${!v:-}" ]] || missing+=("$v")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        cerberus_say "missing ${missing[*]} (put them in the env file); nothing uploaded" >&2
        return 1
    fi
    RCLONE_CONFIG_CERBERUS_R2_TYPE=s3 RCLONE_CONFIG_CERBERUS_R2_PROVIDER=Cloudflare \
    RCLONE_CONFIG_CERBERUS_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
    RCLONE_CONFIG_CERBERUS_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" \
    RCLONE_CONFIG_CERBERUS_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    RCLONE_CONFIG_CERBERUS_R2_ACL=private RCLONE_CONFIG_CERBERUS_R2_NO_CHECK_BUCKET=true \
        rclone "$@"
}

# cerberus_sync <local-dir> <r2-prefix>: copy (never delete, never move) a results directory to the bucket.
cerberus_sync() {
    local dir="$1" prefix="$2"
    [[ -d "$dir" ]] || { cerberus_say "nothing to sync: $dir" >&2; return 1; }
    cerberus_r2 copy "$dir" "cerberus_r2:${R2_BUCKET}/${prefix}" --min-size 1B --transfers 4
}
