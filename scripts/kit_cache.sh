#!/usr/bin/env bash
# Kit startup-cache bundle: discover | save | restore. See scripts/kit_cache.py.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$REPO_ROOT/scripts/kit_cache.py" "$@"
