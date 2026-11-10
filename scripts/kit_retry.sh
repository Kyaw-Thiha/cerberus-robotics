#!/usr/bin/env bash
# Retry a Kit launch on the startup SIGSEGV (exit 139 within 180 s), at most 3 launches. See scripts/kit_retry.py.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$REPO_ROOT/scripts/kit_retry.py" "$@"
