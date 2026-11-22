#!/usr/bin/env bash
# Create one Brev instance inside this run's caps. The kill timer is armed BEFORE the instance exists, and if it
# cannot be armed nothing is created. The price is supplied by the caller (from `brev search gpu --json`) and is
# checked against the 1.50 USD/h cap here, because the create path itself does not enforce a price.
#
# Usage:
#   scripts/host/brev_up.sh --name NAME --type TYPE --price 0.828 [--hours 3.5] [--min-disk 200] [--dry-run]
#
# After it returns: scripts/host/brev_exec.sh NAME '<cmd>' to run the setup, brev_status.sh for the state and the
# spend estimate, brev_down.sh NAME to stop, delete and cancel the timer.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=brev_lib.sh
source "$HERE/brev_lib.sh"
NAME=""; TYPE=""; PRICE=""; HOURS="$BREV_MAX_HOURS"; MIN_DISK="200"; DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --type) TYPE="$2"; shift 2 ;;
        --price) PRICE="$2"; shift 2 ;;
        --hours) HOURS="$2"; shift 2 ;;
        --min-disk) MIN_DISK="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "brev_up: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ -n "$NAME" && -n "$TYPE" && -n "$PRICE" ]] || { echo "usage: brev_up.sh --name N --type T --price P [--hours H]" >&2; exit 2; }
if ! python3 -c "import sys; sys.exit(0 if float('$HOURS') <= float('$BREV_MAX_HOURS') else 1)"; then
    brev_say "requested $HOURS h exceeds the per-instance cap $BREV_MAX_HOURS h; refusing" >&2; exit 3
fi
brev_guard_create "$NAME" "$PRICE" || exit $?

if [[ $DRY -eq 1 ]]; then
    brev_say "dry run: would arm cerberus-kill-$NAME ($HOURS h) and run:"
    echo "  brev create $NAME --type $TYPE --detached --min-disk $MIN_DISK --stoppable"
    exit 0
fi

brev_arm_kill "$NAME" "$HOURS" || exit $?

brev_say "creating $NAME ($TYPE, \$$PRICE/h)"
if ! brev create "$NAME" --type "$TYPE" --detached --min-disk "$MIN_DISK"; then
    brev_say "create failed; cancelling the kill timer (nothing to kill)" >&2
    brev_cancel_kill "$NAME"
    exit 8
fi
brev_append_ledger create "$NAME" "\"type\":\"$TYPE\",\"price\":$PRICE,\"hours_cap\":$HOURS"
brev_say "created; spend so far ~\$$(brev_spend_estimate). Run scripts/host/brev_status.sh $NAME"
