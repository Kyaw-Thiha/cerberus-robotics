#!/usr/bin/env bash
# Copy files to/from an instance THIS RUN created. Source or destination must name that instance (name:/path).
# Usage: scripts/host/brev_copy.sh NAME:/remote/path ./local   |   scripts/host/brev_copy.sh ./local NAME:/remote
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=brev_lib.sh
source "$HERE/brev_lib.sh"
[[ $# -ge 2 ]] || { echo "usage: brev_copy.sh <src> <dst>  (one side must be NAME:/path)" >&2; exit 2; }
managed=0
for arg in "$@"; do
    case "$arg" in *:*) name="${arg%%:*}"
        if brev_managed "$name"; then managed=1; fi ;;
    esac
done
[[ $managed -eq 1 ]] || { brev_say "refusing: neither side names an instance this run created" >&2; exit 4; }
exec brev copy "$@"
