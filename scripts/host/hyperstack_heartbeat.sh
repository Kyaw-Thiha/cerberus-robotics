#!/usr/bin/env bash
# One heartbeat check, run every minute by a laptop loop (or a systemd timer).
#
# Order matters: the SSH /32 ingress rule is refreshed to the laptop's *current* public address FIRST, and only then
# is the VM's heartbeat file touched. Two session-3 VMs were deleted by their own dead-man because a carrier-grade-NAT
# address change made the touch impossible while the rule still pointed at the old address.
#
# Usage: hyperstack_heartbeat.sh --id ID
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=hyperstack_lib.sh
source "$HERE/hyperstack_lib.sh"

VM_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --id) VM_ID="${2:-}"; shift 2 ;;
        -h|--help) echo "usage: hyperstack_heartbeat.sh --id ID" >&2; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$VM_ID" =~ ^[0-9]+$ ]] || { echo "usage: hyperstack_heartbeat.sh --id ID" >&2; exit 2; }

IP="$(hs_current_ip)" || exit 3
[[ -n "$IP" ]] || { hs_say "no public address; not touching the heartbeat" >&2; exit 3; }

VM="$(hs_resolve_vm "$VM_ID")" || { hs_say "cannot look up VM $VM_ID" >&2; exit 3; }
[[ -n "$VM" ]] || { hs_say "VM $VM_ID is gone; stopping" >&2; exit 3; }
FLOATING="$(printf '%s' "$VM" | hs_json_field floating_ip)"
[[ -n "$FLOATING" ]] || { hs_say "VM $VM_ID has no floating IP; not touching the heartbeat" >&2; exit 3; }

if printf '%s' "$VM" | grep -q "$IP/32"; then
    hs_say "ssh rule already allows $IP/32"
else
    hs_say "public address changed; adding $IP/32 before the heartbeat"
    hs_api post "/core/virtual-machines/$VM_ID/sg-rules" \
        "{\"direction\":\"ingress\",\"protocol\":\"tcp\",\"port_range_min\":22,\"port_range_max\":22,\"ethertype\":\"IPv4\",\"remote_ip_prefix\":\"$IP/32\"}" \
        >/dev/null || { hs_say "could not add the ssh rule; heartbeat NOT touched" >&2; exit 4; }
    sleep "${HYPERSTACK_RULE_SETTLE:-10}"
fi

hs_ssh_cmd "$FLOATING" 'mkdir -p /var/lib/cerberus && touch /var/lib/cerberus/heartbeat' || {
    hs_say "ssh touch failed; the dead-man will see a stale heartbeat" >&2
    exit 5
}
hs_say "heartbeat touched"
