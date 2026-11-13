#!/usr/bin/env bash
# Relaunch a long Isaac job after a post-startup failure. The driver must be resumable: relaunch skips finished work.
# usage: run_resilient.sh --name stand --max-restarts 5 -- <cmd...>
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="job"; MAX=5; CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --max-restarts) MAX="$2"; shift 2 ;;
    --) shift; CMD=("$@"); break ;;
    *) echo "run_resilient: unknown option $1" >&2; exit 2 ;;
  esac
done
if [ ${#CMD[@]} -eq 0 ]; then echo "run_resilient: no command" >&2; exit 2; fi
for attempt in $(seq 1 $((MAX + 1))); do
  START=$(date +%s)
  "${CMD[@]}"
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  python3 "$REPO_ROOT/scripts/job_attempt_log.py" "$NAME" "$attempt" "$RC" "$ELAPSED" >/dev/null 2>&1 || true
  echo "[resilient] $NAME attempt $attempt exit=$RC elapsed=${ELAPSED}s" >&2
  if [ $RC -eq 0 ]; then exit 0; fi
  if [ $attempt -gt $MAX ]; then echo "[resilient] $NAME gave up after $MAX restarts" >&2; exit $RC; fi
  sleep 5
done
