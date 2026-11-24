#!/usr/bin/env bash
# Install the pinned NVIDIA driver on a fresh Ubuntu GPU VM and reboot into it.
#
# The provider image ships a driver that Isaac Sim 6.x crashes on (the allowlist in
# scripts/host/driver_allowlist.json is authoritative); this installs 595.58.03 from NVIDIA's runfile with DKMS,
# blacklists nouveau first (it held the GPU in session 3 and the installer refused), and reboots.
#
# Usage: sudo scripts/host/install_driver.sh [--version 595.58.03] [--no-reboot]
# Judge success by `nvidia-smi --query-gpu=driver_version` after the reboot, not by this script's exit code.
set -uo pipefail
VERSION="595.58.03"
REBOOT=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --no-reboot) REBOOT=0; shift ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "install_driver: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ "$(id -u)" == "0" ]] || { echo "install_driver: run as root (sudo)" >&2; exit 2; }

RUNNING=""
[[ -r /proc/driver/nvidia/version ]] && RUNNING="$(awk '{print $8}' /proc/driver/nvidia/version | head -1)"
if [[ "$RUNNING" == "$VERSION" ]]; then
    echo "driver $VERSION already installed"
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential dkms "linux-headers-$(uname -r)" curl ca-certificates

# The provider image ships the driver as Ubuntu packages; the runfile refuses to touch such an installation.
# Purge the packaged driver (not libnvidia-container / the container toolkit, which Docker needs).
if dpkg -l 2>/dev/null | grep -qE '^ii +(nvidia-(driver|dkms|kernel|fabricmanager|firmware|xserver)|libnvidia-(cfg|common|compute|decode|encode|extra|fbc|gl))'; then
    echo "purging the packaged NVIDIA driver"
    apt-get purge -y -qq '^nvidia-(driver|dkms|kernel|fabricmanager|firmware|xserver).*' \
        '^libnvidia-(cfg|common|compute|decode|encode|extra|fbc|gl).*' >/dev/null 2>&1 || true
    apt-get autoremove -y -qq >/dev/null 2>&1 || true
fi
systemctl stop nvidia-persistenced 2>/dev/null || true
for module in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do rmmod "$module" 2>/dev/null || true; done
# Ubuntu's package leaves a marker the runfile reads as "alternate driver installed"; remove it and the 570 leftovers.
rm -f /usr/lib/nvidia/alternate-install-present
dkms status 2>/dev/null | awk -F'[/,]' '/nvidia/ {print $2, $3}' | while read -r ver kernel; do
    dkms remove -m nvidia -v "$ver" -k "$kernel" >/dev/null 2>&1 || true
done
rm -rf /usr/lib/x86_64-linux-gnu/nvidia
# the container runtime's own libnvidia-container.so is NOT part of the driver; keep it
find /usr/lib/x86_64-linux-gnu -maxdepth 1 \( -name 'libnvidia-*.so*' -o -name 'libEGL_nvidia*' \
      -o -name 'libGLESv[12]_nvidia*' -o -name 'libGLX_nvidia*' \) ! -name 'libnvidia-container*' -delete
rm -f "/usr/lib/modules/$(uname -r)/updates/dkms/"nvidia*.ko.zst
depmod -a >/dev/null 2>&1 || true

printf 'blacklist nouveau\noptions nouveau modeset=0\n' > /etc/modprobe.d/blacklist-nouveau.conf
update-initramfs -u >/dev/null 2>&1 || true

RUNFILE="/tmp/NVIDIA-Linux-x86_64-${VERSION}.run"
if [[ ! -s "$RUNFILE" ]]; then
    curl -fL --retry 3 -o "$RUNFILE" \
        "https://us.download.nvidia.com/XFree86/Linux-x86_64/${VERSION}/NVIDIA-Linux-x86_64-${VERSION}.run" \
        || { echo "driver download failed" >&2; exit 3; }
fi
chmod +x "$RUNFILE"
"$RUNFILE" --silent --dkms --no-questions --ui=none || { echo "driver install failed (see /var/log/nvidia-installer.log)" >&2; exit 4; }
touch /tmp/setup_driver.done
echo "driver $VERSION installed; reboot required"
if [[ "$REBOOT" == "1" ]]; then
    (sleep 2; reboot) >/dev/null 2>&1 &
fi
