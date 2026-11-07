#!/usr/bin/env bash
# Pull the run artifacts off the pod (the pod has no rsync; tar over ssh does the job).
#
# Used because this session had no R2 credentials on either side, so "sync before you stop" means
# copying down rather than pushing up. The pattern is the same either way: get the artifacts off the
# disposable instance before it is deleted.
#
# Usage: scripts/pull_pod_artifacts.sh <host> <port> [dest]
set -euo pipefail

HOST="${1:?usage: pull_pod_artifacts.sh <host> <port> [dest]}"
PORT="${2:?usage: pull_pod_artifacts.sh <host> <port> [dest]}"
DEST="${3:-data/pulled}"
KEY="${CERBERUS_SSH_KEY:-$HOME/.ssh/cerberus_runpod_ed25519}"

mkdir -p "$DEST"
ssh -i "$KEY" -o StrictHostKeyChecking=no -o ServerAliveInterval=10 -p "$PORT" "root@$HOST" \
    'cd /workspace/cerberus && tar czf - data/reports docs/reports 2>/dev/null; ' \
    | tar xzf - -C "$DEST"
echo "[pull] artifacts under $DEST"
