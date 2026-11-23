# Session 2 state — FINAL (2026-11-23)

## Host
- Hyperstack CANADA-1 L40 48 GB, 825 GB root, **id 1177066**, `cerberus-scene2`, **185.216.22.143**
- created 07:28:48Z; price 1.0866 USD/h; VM-age cap 9 h (kill timer `cerberus-kill-1177066` armed at +9 h 15 m)
- image `Ubuntu Server 24.04 LTS R570 CUDA 12.8 with Docker` (R595 is hard-restricted to B300 now)
- driver **595.58.03** runfile + DKMS, nouveau blacklisted; `host_doctor` PASS after reinstalling
  nvidia-container-toolkit; one Xid 43 cost a reboot

## Artifacts (on R2 `migration/session2/`, laptop `~/cerberus-results/2026-11-23/session2/`)
- **retrained checkpoint** `go2_arx5_locomotion_ga_v1_candidate.pt` (model_1499.pt, 6.9 MB, 1500 iters, 4096 envs,
  seed 42, 3109 s) + the run's params/ and full checkpoint set
- kinematic mission N=10 + `mission_kinematic.mp4` (1080p30) — draft release `showcase-2026-11-23`
- run logs, unit logs, session log, ledger

## Items
- DONE: preflight, VM, driver, bootstrap, safety nets, K3 harness, clip QA gate, camera rig (code), N1 mission code,
  K1 evidence, K4 report, N4 showcase doc, P2a arm embodiment, **P2c retrain (1500 iters)**
- BLOCKED: P0 look/stills and every RTX clip: MDL modules fail to compile (`MdlModule is NOT in the DB`,
  `USD_MDL ... is Invalid`), camera launches die at 14-20 s or exit 0 at the scene build; K2 matrix (probe bug, then
  cancelled for the retrain; V3 needs NGC_API_KEY); N3 JEPLO (their sim needs a MuJoCo commit that has both
  `user_images_new_` and `Simulate::render_callback`; 3.1.6/3.2.7/3.3.6 each lack one); N2 SLAM (no time box left);
  P1/P3/P2b/P2d/P2e/P5 (depend on the RTX path)
- Arm randomisation events (joint offsets, payload mass) blocked: PhysX device-side assert

## Spend
- ~2.9 USD at 10:10Z (2.7 h x 1.0866) of the 14 USD cap

## Wrap-up checklist
- secrets removed from the VM, dead-man stopped, laptop timer cancelled, heartbeat + inhibitor stopped, VM deleted
  and verified gone, docs pushed, draft release created (not published)
