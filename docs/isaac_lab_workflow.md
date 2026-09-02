# Cerberus — Isaac Lab & Remote Compute Workflow

Reference doc for how local (Arch Linux) and remote (rented GPU) work is split,
why, and how the coding-agent dev loop fits in.

## Why nothing Isaac Sim/Isaac Lab related runs locally

Checked directly: current published minimum spec is a 16GB-VRAM, RTX
4080-class GPU; GPUs without RT cores aren't supported at all regardless of
VRAM. A 4GB RTX 3050 is roughly a quarter of even the older, more lenient
8GB minimum from earlier Isaac Sim versions — this isn't a driver/Arch
compatibility issue, it's a hardware floor issue. Local install would mean
fighting launch failures and OOMs, not a slower-but-usable experience.

**Rule: if it needs `omni.*` APIs or a physics step, it's remote-only.**
Everything else (research code, ROS2 node logic, configs, eval plotting from
already-pulled-down logs) is local.

## Local vs remote split

| | Local (Arch) | Remote (rented GPU) |
|---|---|---|
| Editing code (Neovim, agents) | ✅ primary | via SSH, same experience |
| Isaac Sim / Isaac Lab execution | ❌ never | ✅ only place it runs |
| Training, rollouts, rendering | ❌ | ✅ |
| pixi-managed dev tooling (ruff, pytest, plotting) | ✅ | — |
| Pulled-down artifacts (`.mp4`, logs, checkpoints) for eval/analysis | ✅ | source |

## Compute provider strategy

- **NVIDIA Brev** is the fast on-ramp: a pre-configured Isaac Sim + Isaac Lab
  Launchable (VSCode-style instance, Kit App Streaming, WebRTC ports already
  exposed, Docker Compose handling volumes/networking). Use it for early
  bring-up and any live/interactive debugging session — it removes setup
  friction, not compute cost (it brokers the same underlying GPU market as
  direct providers, so it isn't inherently cheaper).
- **Direct provider (Lambda / RunPod / vast.ai) + own `Dockerfile.remote`**
  for unattended, long, parallel-env training runs (the expensive Phase 0/1
  PPO jobs). No streaming needed for headless training, so shop whichever
  provider is cheapest that week — the Dockerfile is reusable across all
  three, so switching providers has near-zero setup tax once it's written.
- **Treat every instance as disposable.** Sync checkpoints/logs/videos to
  cheap object storage (S3-compatible bucket, Backblaze B2, etc.) at the end
  of each session, then kill the instance. Don't pay for a persistent
  GPU-attached volume "for continuity" — that costs more than a killed
  instance + a few GB/month of object storage.
- **Keep one small, cheap instance semi-persistent** (single L4 / RTX 4000
  Ada class) specifically for the coding-agent-facing dev loop — editing,
  smoke tests, quick verification. Reserve the big multi-GPU training
  instance for actual full runs only, spun up on demand. This keeps
  iterative dev time from burning training-grade GPU budget.

## Visualization: two distinct patterns, not one

1. **Video capture during training (default, ~90% of the time).** Isaac
   Lab's gymnasium wrappers record rollout videos at intervals, headless, no
   live connection needed. Pull the `.mp4`s down and review locally — this
   is the main "is the policy doing something sane" check.
2. **WebRTC streaming client (only when interaction is actually needed).**
   Keep the sim running remote with rendering on, stream the viewport to a
   lightweight local client. Use only for live/interactive debugging (nudge
   the robot, replay a specific failure, poke at physics) — more expensive
   than async video since the GPU renders continuously while you watch.

Avoid X11 forwarding — laggy and fiddly compared to either option above.

## Coding-agent dev loop (Neovim + Claude Code / OpenCode)

Terminal-first setup means no GUI-remoting problem to solve — just run
everything co-located on the remote box over SSH:

1. SSH into the box (use **mosh** instead of plain `ssh` if on flaky/roaming
   wifi — survives IP changes and drops far better).
2. Start or attach a **tmux** session. This is the actual fix for
   session-loss-on-disconnect: a plain SSH-wrapped agent process dies with
   the connection; a tmux-wrapped one keeps running on the box regardless of
   whether your laptop is connected.
3. One pane: Neovim. Another: Claude Code or OpenCode. Optional third:
   scratch/smoke-test commands.
4. Disconnect any time — laptop sleeps, wifi drops, terminal closes. `tmux
   attach` on reconnect restores everything: editor buffers, agent
   conversation, any running background job.
5. Keep **separate tmux sessions per context** — e.g. `dev` (Neovim + agent
   iteration) vs `train` (long unattended runs) — so an active typing session
   never risks interrupting a training job sharing the same window.

## What the coding agent actually needs (and doesn't)

Agents verify through **execution access**, not a GUI:

- **Headless smoke tests** — short runs (few hundred steps, small env count)
  that finish in seconds and give a clear pass/fail: no NaN loss, reward
  curve exists, no shape-mismatch errors. This covers most "does this code
  work" verification.
- **Logs/metrics** — tensorboard scalars, success-rate numbers, ablation
  tables — plain files/stdout, no visual component needed.
- **Video frames for qualitative sanity checks** — since periodic `.mp4`
  capture is already running, extracted frames can be handed to the agent to
  catch obviously broken behavior (immediate face-plant, limbs clipping
  through terrain). This works because the agent can view images directly;
  it's just not live/interactive.
- **The live WebRTC stream stays human-only** — real-time judgment calls on
  "does this gait look right" aren't something to route through the agent.

## Known rough edges to plan around

- Coding-agent sessions run over a plain SSH connection do **not** survive a
  dropped connection by default — the remote process dies with it. tmux is
  the workaround, not a nice-to-have.
- Local and remote coding-agent sessions are **separate, not continuous** —
  session history is scoped to the machine it ran on. Treat local (editing)
  and remote (execution/verification) as two loosely-coupled sessions synced
  via git, not one conversation that follows you across machines.
