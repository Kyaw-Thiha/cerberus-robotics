#!/usr/bin/env bash
# Host-driver gate: run FIRST in every pod driver, before any download or Kit launch.
# Refuses drivers that segfault Isaac Sim 6.x; see scripts/pod_driver_allowlist.json and docs/runpod.md "Pod hosts".
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$REPO_ROOT/scripts/pod_driver_gate.py" "$@"
