#!/usr/bin/env bash
# Shared guards for the Hyperstack wrappers. Sourced, never executed.
#
# The caps are enforced here, before anything is created:
#   * price       1.20 USD/hour all-in (GPU + root disk + public IP + any volume)
#   * wall clock  3 hours per VM, enforced by a systemd user timer armed before creation
#   * creations   3 per run, from the local ledger
#   * one live    creation is refused while any VM exists in the organisation
#   * total       6.00 USD and 5 hours across the run, from the ledger
#   * ownership   down/kill/ssh refuse a VM this run did not create (name prefix plus ledger)
#
# The API key is read from .env inside a subshell for each call; it is never printed. Everything the ledger records
# is non-secret: name, id, flavor, price, image and timestamps.

HS_MAX_PRICE="${HYPERSTACK_MAX_PRICE:-1.20}"
HS_MAX_HOURS="${HYPERSTACK_MAX_HOURS:-3}"
HS_MAX_CREATIONS="${HYPERSTACK_MAX_CREATIONS:-3}"
HS_TOTAL_SPEND="${HYPERSTACK_TOTAL_SPEND:-6.00}"
HS_TOTAL_HOURS="${HYPERSTACK_TOTAL_HOURS:-5}"
HS_DIR="${CERBERUS_HYPERSTACK_DIR:-$HOME/cerberus-results/hyperstack}"
HS_LEDGER="$HS_DIR/ledger.jsonl"
HS_ENV_FILE="${HYPERSTACK_ENV_FILE:-$REPO_ROOT/.env}"
HS_SSH_KEY="${HYPERSTACK_SSH_KEY:-$HOME/.ssh/cerberus_hyperstack}"
HS_API="$HERE/hyperstack_api.py"

hs_say() { echo "[hyperstack $(date -u +%H:%M:%S)] $*"; }

hs_need_key() {
    if ! grep -q '^HYPERSTACK_API_KEY=.' "$HS_ENV_FILE" 2>/dev/null; then
        hs_say "HYPERSTACK_API_KEY is empty or missing in $HS_ENV_FILE; refusing (no value printed)" >&2
        return 2
    fi
}

# One API call: the key exists only inside this subshell, for this process.
hs_api() {
    (
        set -a
        # shellcheck disable=SC1090
        . "$HS_ENV_FILE" >/dev/null 2>&1
        set +a
        exec python3 "$HS_API" "$@"
    )
}

hs_json_field() {  # hs_json_field <field>  (reads JSON on stdin)
    python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except ValueError: raise SystemExit("not JSON")
value=d
for part in sys.argv[1].split("."):
    if not isinstance(value, dict): value=None; break
    value=value.get(part)
if value is None: print("")
else: print(value)' "$1"
}

hs_append_ledger() {  # hs_append_ledger <action> <name> <json-extra>
    mkdir -p "$HS_DIR"
    local ts; ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"ts":"%s","action":"%s","name":"%s"%s}\n' "$ts" "$1" "$2" "${3:+,$3}" >> "$HS_LEDGER"
}

hs_creations() {
    [[ -f "$HS_LEDGER" ]] || { echo 0; return; }
    python3 - "$HS_LEDGER" <<'PY'
import json, sys
created, deleted = [], []
for line in open(sys.argv[1]):
    try: row = json.loads(line)
    except ValueError: continue
    if row.get("action") == "create": created.append(row.get("name"))
    elif row.get("action") == "delete": deleted.append(row.get("name"))
print(len([n for n in created if n not in deleted]))
PY
}

hs_managed() {  # 0 when this run created <name>
    [[ -f "$HS_LEDGER" ]] || return 1
    python3 - "$HS_LEDGER" "$1" <<'PY'
import json, sys
name = sys.argv[2]
for line in open(sys.argv[1]):
    try: row = json.loads(line)
    except ValueError: continue
    if row.get("action") == "create" and row.get("name") == name:
        sys.exit(0)
sys.exit(1)
PY
}

hs_managed_id() {  # 0 when this run created vm id <id>
    [[ -f "$HS_LEDGER" ]] || return 1
    python3 - "$HS_LEDGER" "$1" <<'PY'
import json, sys
want = str(sys.argv[2])
for line in open(sys.argv[1]):
    try: row = json.loads(line)
    except ValueError: continue
    if row.get("action") == "create" and str(row.get("id")) == want:
        sys.exit(0)
sys.exit(1)
PY
}

# Spend and hours are price x time-alive for every create that has no matching delete (still-open VMs included).
hs_usage() {  # prints "<usd> <hours>"
    [[ -f "$HS_LEDGER" ]] || { echo "0.00 0.00"; return; }
    python3 - "$HS_LEDGER" <<'PY'
import json, sys
from datetime import datetime, timezone
rows = []
for line in open(sys.argv[1]):
    try: rows.append(json.loads(line))
    except ValueError: pass
open_since, spend, hours = {}, 0.0, 0.0
for row in rows:
    try: ts = datetime.strptime(row["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (KeyError, ValueError): continue
    if row.get("action") == "create":
        open_since[row["name"]] = (ts, float(row.get("price", 0) or 0))
    elif row.get("action") == "delete" and row["name"] in open_since:
        start, price = open_since.pop(row["name"])
        secs = (ts - start).total_seconds()
        spend += price * secs / 3600
        hours += secs / 3600
now = datetime.now(timezone.utc)
for start, price in open_since.values():
    secs = (now - start).total_seconds()
    spend += price * secs / 3600
    hours += secs / 3600
print(f"{spend:.2f} {hours:.2f}")
PY
}

hs_live_count() {  # any VM in the org counts as live; unknown refuses creation
    local out
    if ! out="$(hs_api list-vms 2>/dev/null)"; then echo unknown; return; fi
    python3 - "$out" <<'PY'
import json, sys
try: rows = json.loads(sys.argv[1]).get("instances", [])
except ValueError: print("unknown"); raise SystemExit
print(sum(1 for row in rows if str(row.get("status", "")).upper() not in {"DELETED"}))
PY
}

hs_lte() { python3 -c "import sys; sys.exit(0 if float('$1') <= float('$2') else 1)"; }

hs_guard_create() {  # hs_guard_create <name> <price-per-hour> <hours>
    local name="$1" price="$2" hours="$3"
    hs_need_key || return 2
    [[ "$name" == cerberus-* ]] || { hs_say "name must start with 'cerberus-' (got $name); refusing" >&2; return 3; }
    hs_lte "$hours" "$HS_MAX_HOURS" || { hs_say "$hours h exceeds the per-VM cap $HS_MAX_HOURS h; refusing" >&2; return 3; }
    hs_lte "$price" "$HS_MAX_PRICE" || { hs_say "price \$$price/h exceeds the cap \$$HS_MAX_PRICE/h; refusing" >&2; return 3; }
    local made; made="$(hs_creations)"
    (( made >= HS_MAX_CREATIONS )) && { hs_say "already created $made VM(s) this run (cap $HS_MAX_CREATIONS); refusing" >&2; return 4; }
    read -r used_usd used_h < <(hs_usage)
    hs_lte "$used_usd" "$HS_TOTAL_SPEND" || { hs_say "spend estimate \$$used_usd already at the \$$HS_TOTAL_SPEND cap; refusing" >&2; return 4; }
    hs_lte "$used_h" "$HS_TOTAL_HOURS" || { hs_say "$used_h h already used of the $HS_TOTAL_HOURS h cap; refusing" >&2; return 4; }
    local live; live="$(hs_live_count)"
    [[ "$live" == "unknown" ]] && { hs_say "cannot read the VM list; refusing with an unknown live count" >&2; return 5; }
    (( live > 0 )) && { hs_say "$live live VM(s) exist; one live at a time, so delete yours first" >&2; return 6; }
    return 0
}

hs_now_plus() {  # absolute local timestamp, <seconds> from now (portable; no date -d)
    python3 - "$1" <<'PY'
import datetime, sys
print((datetime.datetime.now() + datetime.timedelta(seconds=int(float(sys.argv[1])))).strftime("%Y-%m-%d %H:%M:%S"))
PY
}

# hs_arm_kill <unit-suffix> <hours> <kill-script-args...>
hs_arm_kill() {
    local suffix="$1" hours="$2"; shift 2
    local unit="cerberus-kill-$suffix"
    local when; when="$(hs_now_plus "$(python3 -c "print(int(float('$hours') * 3600))")")"
    [[ -n "$when" ]] || { hs_say "could not compute the kill deadline; refusing" >&2; return 7; }
    systemd-run --user --unit="$unit" --on-calendar="$when" --timer-property=Persistent=yes \
        "$HERE/hyperstack_kill.sh" "$@" >/dev/null 2>&1 || {
        hs_say "could not arm the kill timer; refusing to create without one" >&2
        return 7
    }
    if ! systemctl --user list-timers "$unit.timer" --no-pager 2>/dev/null | grep -q "$unit"; then
        hs_say "kill timer did not appear in list-timers; cancelling and refusing" >&2
        hs_cancel_kill "$suffix"
        return 7
    fi
    hs_say "kill timer armed: $unit at $when"
}

hs_cancel_kill() {
    local suffix="$1" unit="cerberus-kill-$1"
    systemctl --user stop "$unit.timer" 2>/dev/null
    systemctl --user stop "$unit.service" 2>/dev/null
    systemctl --user reset-failed "$unit.service" 2>/dev/null
    return 0
}

hs_cancel_all_kill() {  # cancel every timer this run armed for one VM (id and name units)
    local id="$1" name="$2"
    hs_cancel_kill "$id"
    hs_cancel_kill "name-$name"
}

hs_resolve_vm() {  # hs_resolve_vm <id|name> -> JSON on stdout, empty when absent
    local out
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        out="$(hs_api find-vm --id "$1")" || return 1
    else
        out="$(hs_api find-vm --name "$1")" || return 1
    fi
    [[ "$out" == "null" || -z "$out" ]] && return 0
    printf '%s' "$out"
}

hs_current_ip() {
    if [[ -n "${HYPERSTACK_PUBLIC_IP:-}" ]]; then echo "$HYPERSTACK_PUBLIC_IP"; return; fi
    curl -s --max-time 15 https://ifconfig.me
}

hs_ssh_cmd() {  # hs_ssh_cmd <ip> [command...]
    local ip="$1"; shift
    ssh -i "$HS_SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
        -o ConnectTimeout=20 -o ServerAliveInterval=30 "ubuntu@$ip" "$@"
}
