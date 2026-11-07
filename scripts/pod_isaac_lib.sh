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
        ( local last=-1 size stalls=0
          for _ in $(seq 1 "$tmo"); do
              sleep 1
              grep -q "crash has occurred" "$log" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; break; }
              kill -0 "$pid" 2>/dev/null || break
              size=$(stat -c %s "$log" 2>/dev/null || echo 0)
              if [[ "$size" == "$last" ]]; then
                  stalls=$((stalls + 1))
                  # A hung Kit launch keeps the process alive with no new output; driver 595 does
                  # this instead of crashing outright. STALL_S of silence means dead, not slow.
                  if (( stalls >= ${STALL_S:-90} )); then
                      echo "[watchdog] no log growth for ${STALL_S:-90}s: killing stalled launch" >> "$log"
                      kill -9 "$pid" 2>/dev/null
                      break
                  fi
              else
                  stalls=0; last=$size
              fi
          done ) >/dev/null 2>&1 & crash_pid=$!
        wait "$pid"; rc=$?
        kill "$cap_pid" "$crash_pid" 2>/dev/null
        wait "$cap_pid" 2>/dev/null; wait "$crash_pid" 2>/dev/null
        cp "$log" "$REPORT_DIR/$name.log"
        if grep -qE "crash has occurred|no log growth for" "$log" 2>/dev/null; then
            say "$name attempt $i: launch failed (crash or stall, rc=$rc); retrying"
            rc=1
            continue
        fi
        say "$name attempt $i exit=$rc (124/137 = cap)"
        [[ $rc -eq 0 ]] && return 0
    done
    return "$rc"
}
