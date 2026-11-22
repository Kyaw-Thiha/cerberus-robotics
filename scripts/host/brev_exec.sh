#!/usr/bin/env bash
# Run a command non-interactively on an instance THIS RUN created (ownership guard: other instances are not ours).
# Usage: scripts/host/brev_exec.sh NAME '<command>'  |  scripts/host/brev_exec.sh NAME @local-script.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=brev_lib.sh
source "$HERE/brev_lib.sh"
[[ $# -ge 2 ]] || { echo "usage: brev_exec.sh NAME '<command>'" >&2; exit 2; }
NAME="$1"; shift
brev_managed "$NAME" || { brev_say "refusing: $NAME was not created by this run (see $BREV_LEDGER)" >&2; exit 4; }
exec brev exec "$NAME" "$@"
