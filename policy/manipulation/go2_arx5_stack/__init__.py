"""W2a: fixed-base Go2 + ARX5 (X5A) pick-and-place-on-a-table task for Isaac Lab Mimic.

Deliberately lightweight (no isaaclab import here). Module map:

- `layout.py`      pure python: scene geometry, reach envelope, shared constants
- `waypoints.py`   pure python: scripted pick-and-place waypoint state machine
- `register.py`    gymnasium registration (`register_all`), gymnasium imported lazily
- `env_cfg.py`     Isaac Lab env cfg (imports isaaclab; only import after the app launched)
- `mimic_env.py`, `mimic_env_cfg.py`   Isaac Lab Mimic env + cfg (same restriction)

Use `--external_callback policy.manipulation.go2_arx5_stack.register.register_all` with Isaac Lab's
`annotate_demos.py` / `generate_dataset.py` (repo root on PYTHONPATH) to make the task ids known to them.
"""
