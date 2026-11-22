#!/usr/bin/env bash
# The kill switch: DELETE one VM, verify it is gone, and cancel every timer armed for it. Safe to run from systemd.
#
# Usage:
#   scripts/host/hyperstack_kill.sh <vm_id>
#   scripts/host/hyperstack_kill.sh --name NAME
#   scripts/host/hyperstack_kill.sh --check <vm_id|--name NAME>   # harmless GET only
#
# Safety: only a VM named cerberus-* is touched (override with HYPERSTACK_ALLOW_ANY=1). A VM that is already gone is
# not an error: the timer may fire after a normal delete. Never prints a secret.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=hyperstack_lib.sh
source "$HERE/hyperstack_lib.sh"

CHECK=0; TARGET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=1; shift ;;
        --name) TARGET="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) TARGET="$1"; shift ;;
    esac
done
[[ -n "$TARGET" ]] || { echo "usage: hyperstack_kill.sh <vm_id>|--name NAME [--check]" >&2; exit 2; }

VM="$(hs_resolve_vm "$TARGET" 2>/dev/null)" || VM=""
if [[ -z "$VM" ]]; then
    hs_say "$TARGET is already gone (or not visible)"
    [[ $CHECK -eq 0 ]] && systemctl --user list-timers 'cerberus-kill-*' --no-pager 2>/dev/null | grep -q "cerberus-kill-$TARGET" \
        && hs_cancel_kill "$TARGET"
    exit 0
fi
ID="$(printf '%s' "$VM" | hs_json_field id)"
NAME="$(printf '%s' "$VM" | hs_json_field name)"
STATE="$(printf '%s' "$VM" | hs_json_field status)"
if [[ "${HYPERSTACK_ALLOW_ANY:-0}" != "1" && "$NAME" != cerberus-* ]]; then
    hs_say "refusing to touch $NAME (id $ID): not a cerberus-* VM; set HYPERSTACK_ALLOW_ANY=1 to override" >&2
    exit 3
fi
if [[ $CHECK -eq 1 ]]; then
    hs_say "check: id=$ID name=$NAME status=$STATE ip=$(printf '%s' "$VM" | hs_json_field floating_ip)"
    exit 0
fi

hs_say "deleting $NAME (id $ID, $STATE)"
hs_api delete-vm --id "$ID" --timeout 300 >/dev/null || {
    hs_say "DELETE failed or the VM did not disappear within 300 s; timer stays armed" >&2
    exit 4
}
hs_cancel_all_kill "$ID" "$NAME"
hs_append_ledger delete "$NAME" "\"id\":$ID,\"reason\":\"kill\""
if [[ -n "$(hs_resolve_vm "$ID" 2>/dev/null)" ]]; then
    hs_say "VM $ID is still visible after DELETE" >&2
    exit 5
fi
hs_say "gone: $NAME (id $ID); timers for it cancelled"
