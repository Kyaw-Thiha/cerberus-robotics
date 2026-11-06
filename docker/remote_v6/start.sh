#!/usr/bin/env bash
# Cerberus remote image entrypoint (NEW stack: Isaac Sim 6.2.0 / Lab 3.0.0 GA).
#
# Same behaviour as docker/remote/start.sh: configure sshd from $PUBLIC_KEY
# (RunPod convention), export RUNPOD_*/ISAACLAB_PATH for later non-login
# `ssh host "command"` sessions via /etc/cerberus_env.sh, keep the container
# alive by running sshd in the foreground. The Isaac Sim base image's own
# ENTRYPOINT is cleared by the Dockerfile, so SSH must be set up here.
set -eu

mkdir -p ~/.ssh
chmod 700 ~/.ssh

if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi

ssh-keygen -A
mkdir -p /var/run/sshd

# ssh command sessions do not inherit PID 1's env nor Dockerfile ENV, and are
# not login shells; scripts/run_remote_job.sh sources this file explicitly.
# ISAACLAB_PATH is in the Dockerfile ENV, so it is present here.
# RUNPOD_API_KEY is deliberately excluded: this file gets cat'd and sourced by agents
# and the key leaked into a transcript once (2026-10-05). Pod self-termination from a
# pod shell must use the key from the container env, never from this file.
env | grep -E '^(RUNPOD_|ISAACLAB_PATH)' | grep -v '^RUNPOD_API_KEY=' | sed 's/^/export /' > /etc/cerberus_env.sh

exec /usr/sbin/sshd -D
