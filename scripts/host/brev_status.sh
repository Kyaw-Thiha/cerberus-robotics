#!/usr/bin/env bash
# State of this run's instances plus the spend estimate from the ledger, and the kill timers that are armed.
# Usage: scripts/host/brev_status.sh [NAME]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=brev_lib.sh
source "$HERE/brev_lib.sh"
echo "== brev ls"
timeout 30 brev ls 2>&1 | head -20 || echo "(brev ls failed: not logged in?)"
echo "== this run"
echo "created-but-not-deleted: $(brev_creations) of $BREV_MAX_CREATIONS"
echo "spend estimate: \$$(brev_spend_estimate) of the run budget"
echo "== kill timers"
systemctl --user list-timers 'cerberus-kill-*' --no-pager 2>/dev/null | head -10 || echo "(no user timers visible)"
echo "== ledger"
[[ -f "$BREV_LEDGER" ]] && tail -5 "$BREV_LEDGER" || echo "(no ledger at $BREV_LEDGER)"
