#!/usr/bin/env bash
# Install the laptop guard as a systemd user service (not a child of any agent shell), held awake with
# systemd-inhibit while a pod is live, writing a 30 s tick that guard_health.py checks each turn.
# usage: scripts/install_guard_service.sh --targets <id:rate[,id:rate]> [--usd 3] [--wall 5400]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS=""; USD=3; WALL=5400; HARD=10800; STARTED=$(date +%s)
while [ $# -gt 0 ]; do
  case "$1" in
    --targets) TARGETS="$2"; shift 2 ;;
    --usd) USD="$2"; shift 2 ;;
    --wall) WALL="$2"; shift 2 ;;
    --hard-wall) HARD="$2"; shift 2 ;;
    --started) STARTED="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$TARGETS" ] || { echo "need --targets id:rate" >&2; exit 2; }
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/cerberus-guard.service" <<UNIT
[Unit]
Description=Cerberus pod guard (USD cap, wall clock, 30 s tick)
After=default.target

[Service]
Type=simple
WorkingDirectory=$REPO
EnvironmentFile=$REPO/.env
ExecStart=/usr/bin/systemd-inhibit --what=sleep:idle --why="Cerberus pod live" \\
  /usr/bin/python3 $REPO/scripts/session_guard.py --provider runpod \\
  --targets $TARGETS --started $STARTED --wall $WALL --hard-wall $HARD --usd $USD
Restart=always
RestartSec=5
KillMode=process

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now cerberus-guard.service
systemctl --user status cerberus-guard.service --no-pager | head -8
echo "guard service installed; tick file $REPO/outputs/guard_tick.json"
