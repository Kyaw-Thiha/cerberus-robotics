#!/usr/bin/env bash
# Launch probe: run only after pod_driver_gate.sh passes. Proves Kit starts on this host (5-minute cap).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$REPO_ROOT/scripts/pod_launch_probe.py" "$@"
