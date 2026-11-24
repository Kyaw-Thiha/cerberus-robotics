#!/usr/bin/env bash
# Install the VM-side dead-man (run ON the VM, as root): secrets file, vm.json, log-only test, then the timer.
#
# The dead-man copies results to R2 and deletes THIS VM when the laptop heartbeat is stale or the VM is too old.
# Secrets come from the repo's .env on this VM; nothing is printed. Run after bootstrap so a dead-man copy has
# something to copy.
#
# Usage: sudo scripts/host/deadman_setup.sh --id VM_ID [--created-epoch E] [--hours 11.5] [--prefix PREFIX]
#        sudo scripts/host/deadman_setup.sh --log-only-test
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${CERBERUS_ENV_FILE:-$REPO_ROOT/.env}"
HOURS="11.5"
VM_ID=""
CREATED="$(date +%s)"
PREFIX="migration/session4/vm-backup"
TEST_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --id) VM_ID="$2"; shift 2 ;;
        --created-epoch) CREATED="$2"; shift 2 ;;
        --hours) HOURS="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --log-only-test) TEST_ONLY=1; shift ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "deadman_setup: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ "$(id -u)" == "0" ]] || { echo "deadman_setup: run as root (sudo)" >&2; exit 2; }

if [[ "$TEST_ONLY" == "1" ]]; then
    CERBERUS_DEADMAN_LOG_ONLY=1 "$HERE/deadman_vm.sh"
    exit $?
fi
[[ "$VM_ID" =~ ^[0-9]+$ ]] || { echo "deadman_setup: --id is required" >&2; exit 2; }
[[ -r "$ENV_FILE" ]] || { echo "deadman_setup: $ENV_FILE is missing" >&2; exit 2; }
install -d -m 755 /usr/local/lib/cerberus
install -m 755 "$HERE/deadman_vm.sh" /usr/local/lib/cerberus/deadman_vm.sh

install -d -m 700 /etc/cerberus
install -d -o ubuntu -g ubuntu -m 755 /var/lib/cerberus
touch /var/lib/cerberus/heartbeat
chown ubuntu:ubuntu /var/lib/cerberus/heartbeat

umask 077
{
    grep -E '^R2_(ACCOUNT_ID|ACCESS_KEY_ID|SECRET_ACCESS_KEY|BUCKET)=' "$ENV_FILE"
    grep -E '^HYPERSTACK_API_KEY=' "$ENV_FILE" | cut -d= -f2- > /etc/cerberus/hyperstack.key
} > /etc/cerberus/r2.env
chmod 600 /etc/cerberus/r2.env /etc/cerberus/hyperstack.key

python3 - "$VM_ID" "$CREATED" <<'PY'
import json, sys
json.dump({"id": int(sys.argv[1]), "created_epoch": int(sys.argv[2])},
          open("/etc/cerberus/vm.json", "w"))
PY

cat > /etc/default/cerberus-deadman <<EOF
CERBERUS_DEADMAN_HARD_HOURS=$HOURS
CERBERUS_DEADMAN_MAX_AGE_MIN=45
CERBERUS_DEADMAN_PREFIX=$PREFIX
CERBERUS_DEADMAN_LOG_ONLY=0
EOF

cat > /etc/systemd/system/cerberus-deadman.service <<EOF
[Unit]
Description=Cerberus VM dead-man (sync results, then delete this VM)
After=network-online.target docker.service

[Service]
Type=oneshot
EnvironmentFile=/etc/default/cerberus-deadman
ExecStart=/usr/local/lib/cerberus/deadman_vm.sh
EOF

cat > /etc/systemd/system/cerberus-deadman.timer <<'EOF'
[Unit]
Description=Cerberus dead-man check every 5 minutes

[Timer]
OnBootSec=3min
OnUnitActiveSec=5min
AccuracySec=30s

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now cerberus-deadman.timer
systemctl list-timers cerberus-deadman.timer --no-pager
echo "dead-man installed: heartbeat older than 45 min or VM older than $HOURS h -> sync then delete VM $VM_ID"
