#!/usr/bin/env bash
# SSH to a VM this run created, resolving the floating IP from the API each time (so a changed IP is picked up).
#
# Usage:
#   scripts/host/hyperstack_ssh.sh NAME [command...]
#   scripts/host/hyperstack_ssh.sh --id 123 [command...]
#   scripts/host/hyperstack_ssh.sh NAME            # interactive shell
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=hyperstack_lib.sh
source "$HERE/hyperstack_lib.sh"

TARGET=""
if [[ "${1:-}" == "--id" ]]; then TARGET="$2"; shift 2; else TARGET="${1:-}"; shift || true; fi
[[ -n "$TARGET" ]] || { echo "usage: hyperstack_ssh.sh NAME|--id ID [command...]" >&2; exit 2; }

if [[ "$TARGET" =~ ^[0-9]+$ ]]; then
    hs_managed_id "$TARGET" || { hs_say "VM id $TARGET was not created by this run; refusing" >&2; exit 3; }
else
    hs_managed "$TARGET" || { hs_say "VM $TARGET was not created by this run; refusing" >&2; exit 3; }
fi

VM="$(hs_resolve_vm "$TARGET")" || { hs_say "cannot look up $TARGET" >&2; exit 3; }
IP="$(printf '%s' "$VM" | hs_json_field floating_ip)"
STATUS="$(printf '%s' "$VM" | hs_json_field status)"
[[ "$STATUS" == "ACTIVE" ]] || { hs_say "$TARGET is $STATUS, not ACTIVE" >&2; exit 4; }
[[ -n "$IP" ]] || { hs_say "$TARGET has no floating IP yet" >&2; exit 4; }
hs_ssh_cmd "$IP" "$@"
