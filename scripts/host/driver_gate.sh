#!/usr/bin/env bash
# Host-driver gate: run FIRST in every host session, before any download or Kit launch.
# Refuses drivers that segfault Isaac Sim 6.x; see scripts/host/driver_allowlist.json and docs/archive.md "Pod hosts".
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$REPO_ROOT/scripts/host/driver_gate.py" "$@"
