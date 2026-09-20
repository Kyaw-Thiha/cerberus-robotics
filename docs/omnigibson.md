# Cerberus — OmniGibson Reference

Short reference for what OmniGibson is used for, which scenes, how they're split, and how
they fit the Docker/remote workflow. Complements `cerberus_project_context.md` (Mission
domain decision) and `isaac_lab_workflow.md` (remote build workflow). Full technical detail,
gotchas, and open verification items are in `docs/omnigibson_handoff.md` — this doc is the
short version.

## What OmniGibson is for this project

OmniGibson is used purely as a **scene-loading and decryption layer**, not as the simulator.
BEHAVIOR-1K's scene and object assets are encrypted and only resolve through OmniGibson's own
loading path — there's no way to import them as plain USD files. Once a scene is loaded onto
the underlying Isaac Sim stage, OmniGibson's own `Environment`, robot classes, and step loop
are skipped entirely — Isaac Lab drives the robot, physics stepping, and RL/IL loop the same
way it does for every other task in this repo.

## Scenes and missions

| Mission | Status | Scenes drawn from |
|---|---|---|
| Household | Primary | `*_int` / `*_garden` house scenes (Beechwood, Benevolence, Ihlen, Merom, Pomaria, Rs, Wainscott, `house_*`) |
| Grocery | Secondary | `grocery_store_cafe` |
| Restaurant | Secondary | `restaurant_brunch` |
| Office | Secondary | `office_large` |

Household has far more scene instances (~20) than the secondary categories (4–6 each) — this
imbalance matters for training (see below), not just scene selection.

## Dataset split: stratify by category, hold out per category, one policy

For Phase 1 nav training:

- Stratify scenes by mission category (household / grocery / restaurant / office).
- Hold out one scene per category for eval only, train on the rest — standard practice in
  embodied-nav benchmarks (Habitat/Gibson challenges; iGibson's own train/test split on this
  same scene family holds out `Ihlen_0/1_int`, `Rs_int`).
- **One policy trained across all categories**, not one policy per category.
- **Sample category uniformly, then scene uniformly within category**, rather than pooling all
  scenes and sampling by raw instance count. Naive pooling lets household (the largest category)
  dominate gradient updates purely on scene count — a well-documented failure mode in
  multi-domain training. Category-balanced sampling fixes this without discarding household's
  extra diversity.
- Exact held-out scene per category is **not locked yet** — decide when the Phase 1 scene set
  is finalized, per this project's "decide just before the phase starts" convention.

## Docker / remote build

- No local download step — OmniGibson only ever runs remote (same "if it needs `omni.*` APIs,
  it's remote-only" rule as everything else in `isaac_lab_workflow.md`).
- Dataset download/extraction (`download_assets()`, `download_behavior_1k_assets(accept_license=True)`)
  runs as a `RUN` step inside `Dockerfile.remote`, build once, push to a registry, `docker pull`
  on every new pod — no repeat download per pod.
- **Environment and code are separate images/layers.** `Dockerfile.remote` bakes in Isaac Lab +
  OmniGibson + the dataset only; your actual Cerberus code lives outside the image and is
  `git`-synced on each pod per the existing dev-loop in `isaac_lab_workflow.md`. Code changes
  never trigger a dataset re-download.

See `docs/omnigibson_handoff.md` for the exact API calls, partial-scene-loading flags, version
pinning caveats, and known integration gotchas.
