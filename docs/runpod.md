# Cerberus — RunPod Remote Execution Playbook

Practical, tactical reference for actually operating a RunPod pod session for
this project — accumulated from real sessions, not theoretical. See
`docs/isaac_lab_workflow.md` for the broader "why remote at all" / provider
strategy; this doc is the "how, concretely, day to day" companion to it.

## Pod lifecycle

Use the RunPod MCP tools (`create-pod`, `get-pod`, `list-pods`, `delete-pod`,
`stream-pod-logs`) rather than shelling out to `runpodctl` when they're
available — structured, no auth-string parsing.

- **Image**: `kevinkyawthiha/cerberus-remote:v1` (built from
  `docker/Dockerfile.remote`), GPU `NVIDIA GeForce RTX 4090`, `SECURE` cloud
  tier, 40GB container disk, port `22/tcp` exposed, SSH public key from the
  dedicated automation key (see "SSH access" below).
- **Cold start is unpredictable, not stuck**: pulling the ~7.5GB image takes
  anywhere from ~1 to ~5+ minutes depending on host/network. `get-pod`
  returning `runtime: null` for a while is normal. Check
  `stream-pod-logs(source="system", tail=5, maxWaitMs=3000)` for actual
  download progress before concluding anything is wrong — don't re-roll on a
  hunch. See `runpod_stuck_pod_diagnosis` precedent: this exact pattern was
  once mistaken for a stuck host and it wasn't.
- **`get-pod` gives the direct SSH mapping** once runtime is up
  (`runtime.ssh.direct` / the top-level `ssh.direct` field: host + port).
  Prefer this over the `ssh.io` proxy once available — fewer moving parts.
- **Availability varies** — `create-pod` can 400 with "no instances
  available" for a GPU type/region combo; retry without a restrictive
  `dataCenterIds` filter, or with a different explicit list, rather than
  giving up.
- **Terminating and recreating can land you back on the same physical host**
  if you don't constrain the datacenter differently — relevant if you're
  re-rolling specifically to escape a bad host (see driver section below).

## SSH access

Use the dedicated, passphrase-free automation key
(`~/.ssh/cerberus_runpod_ed25519`), never the personal key — the personal key
is passphrase-protected and silently fails for non-interactive SSH. Pass its
public key as `sshPublicKey` at pod-create time.

## GPU driver compatibility — check before any rendering work

Isaac Sim 5.1.0's RTX renderer is driver-version-sensitive, and this is
**host-level infrastructure outside the container's control** — whichever
physical machine RunPod allocates brings its own driver.

- **595.x (R590 branch) is confirmed-bad** — reproducibly crashes the RTX
  renderer (segfault at Hydra engine creation). `policy/cli_common.py`'s
  `check_gpu_driver_for_rendering()` blocks this branch specifically before
  paying Isaac Sim's multi-minute boot.
- **580.65.06 is the officially validated version.**
- **Everything else (570.x seen twice in one session, other untested
  branches) is a genuine unknown** — not pre-emptively blocked by the driver
  check, and re-rolling isn't guaranteed to escape it. Don't guess from the
  version number either way. Instead, for any pod session where rendering
  happens at some point (video capture, `--video`), run a **cheap render
  smoke test first**: a minimal standalone script — `AppLauncher` with
  `enable_cameras=True`, a tiny 1-2 env task, no trained checkpoint needed,
  a handful of steps checking `env.unwrapped.render()` returns non-None. Takes
  under 3 minutes (mostly Isaac Sim's own boot) and gives a definitive answer
  before committing pod time to an expensive run that only renders at the
  very end — much cheaper than discovering a crash after paying for the
  training part first.

## Syncing code to a pod

The image does **not** have `rsync` installed — use `tar` over SSH instead,
which needs zero remote setup:

```bash
tar czf - --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='policy/locomotion/checkpoints' --exclude='policy/locomotion/runs' \
    policy docs scripts .env | \
  ssh -i ~/.ssh/cerberus_runpod_ed25519 -p <port> root@<host> \
    "mkdir -p /workspace/cerberus && tar xzf - -C /workspace/cerberus"
```

`.env` is a dotfile — verify it actually synced with `ls -la`, not plain
`ls`, which silently hides it and can look like a sync failure when it
worked fine.

## Launching long-running work: always detached, always killable

Every remote invocation — training, eval, video capture, a whole multi-step
pipeline script — should be:

1. **Launched via `nohup ... & disown`** over a single SSH command. This is
   the entire mechanism that lets work survive the local terminal closing,
   the SSH connection dropping, or the calling Claude Code session ending
   entirely — the work has to live purely in the pod's own process tree, not
   depend on anything local staying connected.
2. **Wrapped in `timeout --signal=KILL --kill-after=<grace> <duration>`,
   never bare `timeout`.** A genuinely stuck Isaac Sim process (degenerate
   physics state, or a Python-level crash that doesn't cleanly tear down the
   Kit app) can ignore plain SIGTERM outright and spin indefinitely — this is
   confirmed, not theoretical, and has happened more than once. The SIGKILL
   wrapper is what self-recovers a hung step automatically; don't rely on
   active monitoring to catch it.
3. Note GNU `timeout`'s duration syntax wants a single unit suffix
   (`150m`), not combined units (`2h30m` is rejected as "invalid time
   interval").

For a multi-step sequence (train → video → eval → video → sync, say), write
**one shell script** that chains every step with its own R2 sync after each,
upload it once, launch it detached, then verify it's alive — don't try to
babysit each step with live tool calls if the human is about to close their
laptop or step away.

## Verifying before you say "safe to walk away"

Never tell a user it's safe to close their terminal or sleep their laptop
just because a launch command returned successfully — a launch that crashed
10 seconds in looks identical to "started fine" from the outside. Actually
check, right after launching:

- `ps aux | grep ...` — confirm the process tree exists **and** shows no
  controlling TTY (`?` in the TTY column) — that's what confirms it's
  genuinely detached, not just backgrounded in a session that's about to die.
- `nvidia-smi` — confirm real GPU utilization, not 0%.
- The actual log content — confirm real progress (an iteration count ticking
  up, a checkpoint being written), not just a process existing with no output.

## Monitoring a long remote job

A bare `ssh ... "tail -F remote.log" | grep ...` Monitor **can silently stop
delivering live updates for hours while the underlying job keeps working
correctly underneath** — this caused a real ~10-hour idle-pod cost incident
in this project's history (the training had actually finished on time; the
notification stream just went quiet). Always wrap remote log-tailing in a
reconnect loop:

```bash
while true; do
  ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -p <port> root@<host> "tail -n 0 -F remote.log" 2>&1 | your-filter
  echo "[monitor] disconnected, reconnecting in 5s..."
  sleep 5
done
```

Treat "disconnected, reconnecting" messages as normal background noise, not
alarming — as long as you independently re-verify the underlying job is
still progressing (direct log read, process check) rather than trusting the
notification stream alone as the sole source of truth.

**Default policy: pods terminate after their work + final sync succeeds**,
unless the very next step is queued immediately in the same session. If a
pod needs to stay up between steps (e.g. review results before deciding
whether to run eval), that's a deliberate `--no-terminate`-style choice, not
the default — and it should still get terminated explicitly once the human
confirms they're done reviewing, not left running "just in case."

## A `torch.inference_mode()` gotcha specific to multi-checkpoint inference scripts

If you write a script that loops over multiple checkpoints/conditions against
one persistent env, calling `env.reset()` between each: wrap the **entire**
loop (every reset, every step, across every checkpoint) in ONE persistent
`torch.inference_mode():` block from the start, not re-entered per
clip/iteration. A tensor touched inside `inference_mode()` stays permanently
marked as an "inference tensor" even after that block exits, and a later
in-place write to it from *outside* inference_mode (the next iteration's
`env.reset()`, if it's outside the block) throws `RuntimeError: Inplace
update to inference tensor outside InferenceMode is not allowed`. This only
bites scripts with multiple manual resets across separate inference_mode
scopes in one process — existing single-continuous-rollout scripts in this
repo never hit it because they never manually reset mid-script.

## R2 artifact sync

See `docs/artifact_storage.md` for the full bucket layout and rationale.
Practical note relevant to remote sessions specifically: `scripts/
run_remote_job.sh` handles periodic + final sync automatically for a
training invocation (survives a mid-run force-kill from RunPod's own balance
enforcement); anything outside that wrapper (eval sweeps, video capture)
should call `scripts/sync_to_r2.py sync-run` explicitly after each step in
your pipeline script, not just once at the very end.
