#!/usr/bin/env bash
# Rehearsal replacement for isaaclab.sh: never touches Isaac, prints the startup marker kit_retry waits for, sleeps the
# requested startup delay, then runs the stub app for the requested stage.
echo "[stub] Simulation App Startup Complete"
sleep "${CERBERUS_REHEARSAL_STARTUP_S:-2}"
exec python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stub_app.py" "$@"
