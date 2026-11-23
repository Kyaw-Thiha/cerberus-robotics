# Session 3 incident log: the VM died to its own dead-man

## What happened

At 20:07Z the L40 `cerberus-scene3` (id 1178616) was created with the full safety net: laptop kill timer at 7 h, VM
dead-man (deletes the VM when `/var/lib/cerberus/heartbeat` is older than 45 min), and a laptop heartbeat service
that SSHes in every 60 s to touch that file.

The laptop's public address is carrier-grade NAT. Between 20:40Z and 21:20Z it moved from `138.51.64.37` to
`138.51.73.111`. The Hyperstack firewall only admits TCP 22 from that `/32`, so SSH stopped answering; the heartbeat
stopped; 45 minutes later the dead-man did exactly what it is built to do: it copied `/home/ubuntu/cerberus-results`
to `R2:migration/session3/vm-backup/` and then deleted VM 1178616 through the API.

That is the safety net working, not a bug in it. The design flaw is that the heartbeat's liveness depends on the
same `/32` the heartbeat is supposed to protect against being unreachable: a NAT change looks exactly like a dead
laptop.

## What was lost

About 1.2 h of VM time (≈1.4 USD) and the bootstrap work (52.8 GB image pull, 25 GB scene fetch). No results were
lost: the only outputs so far were the bootstrap logs, `doctor.json` and `preview.json`, all of which are in the R2
backup above. The repository state is on GitHub.

## What changes for the next VM

1. The laptop-side heartbeat now refreshes the Hyperstack ingress rule for the *current* address before every touch,
   so a NAT change no longer stops it (it also retries and re-checks).
2. The dead-man's stale threshold is raised to 90 minutes, so a few minutes of network trouble cannot kill a
   productive VM.
3. The laptop kill timer stays as the hard stop, as before.
4. `~/cerberus_ssh.sh` does the same refresh for interactive work.

## Spend accounting

VM 1178616: created 20:07:23Z, deleted ~21:20Z by the dead-man. 1.2 h × 1.0866 ≈ **1.35 USD**. Creations used: 1 of 3.
