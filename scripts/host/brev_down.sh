#!/usr/bin/env bash
# Stop, then delete an instance THIS RUN created, and cancel its kill timer. Idempotent: a missing instance is not
# an error, and a failed stop still tries the delete (leave nothing running).
# Usage: scripts/host/brev_down.sh NAME...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=brev_lib.sh
source "$HERE/brev_lib.sh"
[[ $# -ge 1 ]] || { echo "usage: brev_down.sh NAME..." >&2; exit 2; }
rc=0
for NAME in "$@"; do
    brev_managed "$NAME" || { brev_say "refusing: $NAME was not created by this run" >&2; rc=4; continue; }
    brev_say "stopping $NAME"
    timeout 300 brev stop "$NAME" || brev_say "WARNING: stop failed or timed out; deleting anyway" >&2
    sleep 5
    if timeout 300 brev delete "$NAME"; then
        brev_append_ledger delete "$NAME"
        brev_cancel_kill "$NAME"
        brev_say "deleted $NAME; spend estimate \$$(brev_spend_estimate)"
    else
        brev_say "ERROR: delete failed for $NAME; the kill timer stays armed as the backstop" >&2
        rc=1
    fi
done
exit "$rc"
