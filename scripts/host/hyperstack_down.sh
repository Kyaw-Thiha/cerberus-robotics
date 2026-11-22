#!/usr/bin/env bash
# Delete a VM this run created, verify it is gone, cancel its kill timers and close the ledger entry.
#
# Usage: scripts/host/hyperstack_down.sh NAME|--id ID
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=hyperstack_lib.sh
source "$HERE/hyperstack_lib.sh"

TARGET=""
if [[ "${1:-}" == "--id" ]]; then TARGET="$2"; else TARGET="${1:-}"; fi
[[ -n "$TARGET" ]] || { echo "usage: hyperstack_down.sh NAME|--id ID" >&2; exit 2; }

if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
    hs_managed_id "$TARGET" || { hs_say "VM id $TARGET was not created by this run; refusing" >&2; exit 3; }
    NAME="$(hs_resolve_vm "$TARGET" | hs_json_field name)"
else
    hs_managed "$TARGET" || { hs_say "VM $TARGET was not created by this run; refusing" >&2; exit 3; }
    NAME="$TARGET"
fi

"$HERE/hyperstack_kill.sh" "$TARGET" || exit $?
hs_append_ledger delete "$NAME" "\"target\":\"$TARGET\""
hs_say "deleted $NAME; spend ~\$$(hs_usage | cut -d' ' -f1); remaining timers: $(systemctl --user list-timers 'cerberus-kill-*' --no-pager 2>/dev/null | grep -c 'cerberus-kill-' || true)"
