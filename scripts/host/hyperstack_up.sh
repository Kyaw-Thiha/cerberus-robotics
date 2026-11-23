#!/usr/bin/env bash
# Create one Hyperstack VM inside this run's caps. The kill timer is armed before the POST and re-armed on the real VM
# id immediately after it, so a lost response cannot leave an untracked VM billing.
#
# Usage:
#   scripts/host/hyperstack_up.sh --name NAME --flavor n3-L40x1-bigroot [--region CANADA-1]
#                                 [--image "Ubuntu Server 24.04 LTS R595 CUDA 13.2 with Docker"]
#                                 [--hours 3] [--cidr 203.0.113.7/32] [--key-name cerberus-hyperstack]
#                                 [--ssh-key ~/.ssh/cerberus_hyperstack] [--dry-run]
#
# The price is computed from the live pricebook (GPU + disk + public IP) and checked against the cap; the SSH ingress
# rule is TCP 22 restricted to the current public IP as /32 unless --cidr is given.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=hyperstack_lib.sh
source "$HERE/hyperstack_lib.sh"

NAME=""; REGION="CANADA-1"; FLAVOR=""; IMAGE=""; HOURS="$HS_MAX_HOURS"; CIDR=""
KEY_NAME="cerberus-hyperstack"; DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --flavor) FLAVOR="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --hours) HOURS="$2"; shift 2 ;;
        --cidr) CIDR="$2"; shift 2 ;;
        --key-name) KEY_NAME="$2"; shift 2 ;;
        --ssh-key) HS_SSH_KEY="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "hyperstack_up: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ -n "$NAME" && -n "$FLAVOR" ]] || { echo "usage: hyperstack_up.sh --name N --flavor F [--region R]" >&2; exit 2; }
hs_need_key || exit 2

if [[ -z "$CIDR" ]]; then
    IP="$(hs_current_ip)"
    CIDR="$(hs_api validate-cidr "${IP}/32" 2>/dev/null | hs_json_field cidr)"
    [[ -n "$CIDR" ]] || { hs_say "could not determine the current public IP; pass --cidr explicitly" >&2; exit 3; }
fi

PRICE_JSON="$(hs_api price --flavor "$FLAVOR")" || { hs_say "cannot price $FLAVOR" >&2; exit 3; }
PRICE="$(printf '%s' "$PRICE_JSON" | hs_json_field total_per_hour)"
[[ -n "$PRICE" ]] || { hs_say "no price for $FLAVOR: $PRICE_JSON" >&2; exit 3; }
hs_say "$FLAVOR: \$$PRICE/h all-in ($(printf '%s' "$PRICE_JSON" | hs_json_field gpu), $(printf '%s' "$PRICE_JSON" | hs_json_field disk_gb) GB disk, IP)"
hs_guard_create "$NAME" "$PRICE" "$HOURS" || exit $?

ENV_JSON="$(hs_api ensure-environment --region "$REGION")" || { hs_say "cannot resolve environment" >&2; exit 3; }
ENVIRONMENT="$(printf '%s' "$ENV_JSON" | hs_json_field name)"
[[ -n "$ENVIRONMENT" ]] || { hs_say "no environment for $REGION" >&2; exit 3; }

[[ -f "$HS_SSH_KEY.pub" ]] || { hs_say "public key $HS_SSH_KEY.pub is missing; run ssh-keygen first" >&2; exit 3; }
KEY_JSON="$(hs_api ensure-keypair --region "$REGION" --name "$KEY_NAME" --public-key-file "$HS_SSH_KEY.pub")" \
    || { hs_say "cannot ensure key pair $KEY_NAME" >&2; exit 3; }
hs_say "key pair $KEY_NAME (created: $(printf '%s' "$KEY_JSON" | hs_json_field created))"

if [[ -z "$IMAGE" ]]; then
    IMAGE_JSON="$(hs_api pick-image --region "$REGION" --flavor "$FLAVOR")" || { hs_say "cannot pick an image" >&2; exit 3; }
    IMAGE="$(printf '%s' "$IMAGE_JSON" | hs_json_field name)"
fi
[[ -n "$IMAGE" ]] || { hs_say "no image selected" >&2; exit 3; }
hs_say "image: $IMAGE"

PAYLOAD="$(python3 - "$NAME" "$ENVIRONMENT" "$IMAGE" "$FLAVOR" "$KEY_NAME" "$CIDR" <<'PY'
import json, sys
name, env, image, flavor, key, cidr = sys.argv[1:7]
print(json.dumps({
    "name": name, "environment_name": env, "image_name": image, "flavor_name": flavor,
    "key_name": key, "count": 1, "assign_floating_ip": True, "enable_port_randomization": False,
    "security_rules": [{
        "direction": "ingress", "protocol": "tcp", "port_range_min": 22, "port_range_max": 22,
        "ethertype": "IPv4", "remote_ip_prefix": cidr,
    }],
}))
PY
)"

if [[ $DRY -eq 1 ]]; then
    hs_say "dry run: would arm cerberus-kill-name-$NAME ($HOURS h), then POST:"
    printf '%s\n' "$PAYLOAD" | python3 -m json.tool
    exit 0
fi

hs_arm_kill "name-$NAME" "$HOURS" --name "$NAME" || exit $?
hs_say "creating $NAME ($FLAVOR, \$$PRICE/h, SSH from $CIDR)"
CREATED="$(hs_api create-vm --payload "$PAYLOAD")" || {
    hs_say "create failed; cancelling the name timer (nothing to kill)" >&2
    hs_cancel_kill "name-$NAME"
    exit 8
}
VM_ID="$(printf '%s' "$CREATED" | hs_json_field id)"
[[ -n "$VM_ID" ]] || {
    hs_say "create returned no id; keeping the name timer armed so the VM cannot outlive 3 h" >&2
    exit 8
}
hs_arm_kill "$VM_ID" "$HOURS" "$VM_ID" || { hs_say "id timer failed; the name timer remains armed" >&2; }
hs_cancel_kill "name-$NAME"
hs_append_ledger create "$NAME" "\"id\":$VM_ID,\"flavor\":\"$FLAVOR\",\"price\":$PRICE,\"hours_cap\":$HOURS,\"image\":\"$IMAGE\",\"environment\":\"$ENVIRONMENT\",\"cidr\":\"$CIDR\""
hs_say "VM $VM_ID created; waiting for ACTIVE and a floating IP"

if ! VM_JSON="$(hs_api wait-vm --id "$VM_ID" --timeout 900)"; then
    hs_say "VM $VM_ID did not become usable; deleting it (the timer would do the same at $HOURS h)" >&2
    "$HERE/hyperstack_kill.sh" "$VM_ID" >/dev/null 2>&1
    exit 9
fi
VM_IP="$(printf '%s' "$VM_JSON" | hs_json_field floating_ip)"
hs_append_ledger ready "$NAME" "\"id\":$VM_ID,\"ip\":\"$VM_IP\""
hs_say "ACTIVE: $NAME id=$VM_ID ip=$VM_IP"
hs_say "ssh -i $HS_SSH_KEY -o IdentitiesOnly=yes ubuntu@$VM_IP"
hs_say "spend so far ~\$$(hs_usage | cut -d' ' -f1); delete with hyperstack_down.sh $NAME"
printf '%s\n' "$VM_IP"
