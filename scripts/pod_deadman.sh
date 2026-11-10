#!/usr/bin/env bash
# Dead-man switch: stop this pod (not delete it) after CERBERUS_POD_MAX_HOURS, default 3.
#
# Run by every pod driver right after the launch probe passes. It is the safety net for a hung job, a lost shell or an
# exhausted balance: the timer lives in the pod's own process tree and stops the pod through the REST API, so nothing
# keeps billing after the cap. Override the cap with CERBERUS_POD_MAX_HOURS (fractional hours allowed).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_ID="${RUNPOD_POD_ID:-}"
HOURS="${CERBERUS_POD_MAX_HOURS:-3}"

if [[ -z "$POD_ID" ]]; then
    echo "[deadman] RUNPOD_POD_ID is unset: no auto-stop armed (pod not created by Runpod, or env not passed?)"
    exit 0
fi

MARKER="/tmp/pod_deadman_${POD_ID}.pid"
if [[ -f "$MARKER" ]] && kill -0 "$(cat "$MARKER")" 2>/dev/null; then
    echo "[deadman] already armed for $POD_ID (pid $(cat "$MARKER"))"
    exit 0
fi

NOW=$(date +%s)
STOP_AT=$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT/scripts'); import pod_stop; print(pod_stop.deadline_from_now(now=int('$NOW'), hours=float('$HOURS')))") || exit 1
STOP_UTC=$(date -u -d "@$STOP_AT" +"%Y-%m-%dT%H:%M:%SZ")
echo "[deadman] pod $POD_ID will be STOPPED at $STOP_UTC (in $((STOP_AT - NOW)) s; CERBERUS_POD_MAX_HOURS=$HOURS)"

nohup bash -c "sleep $((STOP_AT - NOW)); python3 '$REPO_ROOT/scripts/pod_stop.py' --pod-id '$POD_ID' >> /tmp/pod_deadman.log 2>&1" >/dev/null 2>&1 &
echo $! > "$MARKER"
exit 0
