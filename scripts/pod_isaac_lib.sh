#!/usr/bin/env bash
# Shared pod-side helpers for the Isaac drivers. Sourced, not executed.
#
# run_isaac exists because on driver 595 a failed Kit launch does not exit: it writes a crash report
# and then hangs (docs/runpod.md). A plain `timeout` therefore burns the whole per-step cap on a dead
# process. This runs the command under two watchdogs -- a hard cap and a crash-marker detector -- and
# retries, which is what the repo's own smoke script does with a much blunter timeout.
#
# Requires the sourcing script to define: REPORT_DIR, say().

run_isaac() {  # run_isaac <name> <timeout_s> <attempts> -- cmd...
    local name="$1" tmo="$2" tries="$3" rc=1 i pid cap_pid crash_pid; shift 4
    for ((i = 1; i <= tries; i++)); do
        local log="$REPORT_DIR/$name.attempt$i.log"
        say "$name attempt $i/$tries"
        "$@" > "$log" 2>&1 &
        pid=$!
        ( sleep "$tmo"; kill -9 "$pid" 2>/dev/null ) >/dev/null 2>&1 & cap_pid=$!
        ( for _ in $(seq 1 300); do
              sleep 1
              grep -q "crash has occurred" "$log" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; break; }
              kill -0 "$pid" 2>/dev/null || break
          done ) >/dev/null 2>&1 & crash_pid=$!
        wait "$pid"; rc=$?
        kill "$cap_pid" "$crash_pid" 2>/dev/null
        wait "$cap_pid" 2>/dev/null; wait "$crash_pid" 2>/dev/null
        cp "$log" "$REPORT_DIR/$name.log"
        if grep -q "crash has occurred" "$log" 2>/dev/null; then
            say "$name attempt $i: Kit crashed at launch (rc=$rc); retrying"
            rc=1
            continue
        fi
        say "$name attempt $i exit=$rc (124/137 = cap)"
        [[ $rc -eq 0 ]] && return 0
    done
    return "$rc"
}
