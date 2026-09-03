#!/usr/bin/env bash
# Cerberus remote image entrypoint.
#
# Built FROM nvcr.io/nvidia/isaac-sim, not a runpod/* base, so unlike an
# official Runpod image (which ships its own /start.sh) this container has to
# set up SSH itself or a container-native provider locks you out entirely —
# see docker/Dockerfile.remote's header comment for the per-vendor rundown.
set -eu

mkdir -p ~/.ssh
chmod 700 ~/.ssh

# RunPod's convention: injects the account's registered public key(s) into
# $PUBLIC_KEY at container start. vast.ai's SSH launch mode instead overrides
# this entrypoint entirely and runs its own onstart script — it only needs
# sshd installed and startable, which apt-get below already provides, so this
# file simply never executes there. On a bare-VM provider (Lambda, Brev) SSH
# happens at the host level before docker is even invoked, $PUBLIC_KEY is
# unset, and this block is a no-op — the container just needs to stay up.
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi

ssh-keygen -A

# RunPod injects RUNPOD_POD_ID/RUNPOD_API_KEY/etc into this container's PID 1
# environment (this script), but that does NOT reach a later `ssh host
# "command"` session on its own -- an official runpod/* base image apparently
# handles that propagation itself; this image (built FROM the Isaac Sim base,
# not a runpod/* one) doesn't get it for free. Rather than depend on PAM/
# /etc/environment semantics that vary by distro and by login-vs-non-login
# shell (an `ssh host "command"` session, which is how scripts/run_remote_job.sh
# is actually invoked, does NOT run as a login shell and so does NOT source
# /etc/profile.d/* either), write these explicitly to a file our own scripts
# source themselves -- see scripts/run_remote_job.sh.
env | grep '^RUNPOD_' | sed 's/^/export /' > /etc/cerberus_env.sh

exec /usr/sbin/sshd -D
