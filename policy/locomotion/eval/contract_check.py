"""Contract check: does the checkpoint being evaluated see the same problem it was trained on?

A checkpoint is weights plus an implicit interface: observation layout and order, action scale and
offset, control frequency, command ranges, and (for rsl-rl) whether the policy input was normalised.
Move the stack under a fixed checkpoint (Lab 2.3.2 -> Lab 3.0 in this phase) and any of those can
change without an error: the policy simply behaves differently. That is exactly the failure mode
that makes "September vs GA" numbers meaningless if they are read without this check.

How it works
------------
``load_reference(<run dir>)`` reads a past run's ``params/env.yaml`` + ``params/agent.yaml`` -- the
dumped config of the run that produced the checkpoint, which is the only authoritative record of
what the checkpoint was trained against. ``live_contract(env_cfg, agent_cfg, env)`` extracts the same
fields from the freshly built configs (and, if ``env`` is given, the *actual* observation term dims
from the observation manager). ``compare()`` then diffs them field by field.

Derived quantities (and their assumptions, stated once here rather than in the numbers):
  * height-scan dimension = (size_x / resolution + 1) * (size_y / resolution + 1) for the grid
    pattern, i.e. 1.6 m x 1.0 m at 0.1 m = 17 x 11 = 187 cells.
  * joint width = ``GO2_JOINT_COUNT`` (12 actuated leg joints), because the dumped yaml stores joint
    *patterns* (``.*_hip_joint``), not the expanded list; 12 is the Go2's own actuated joint count.
  * observation dimension = sum of the term widths above; it is recomputed from the reference's own
    term list, so a changed term *order* or a changed height-scan grid shows up as a mismatch.

Pure python: no Isaac, no torch. The only non-stdlib import is PyYAML. Usable in local tests against
a run directory on disk (skip those tests when the run tree is absent -- it is gitignored).
"""

from __future__ import annotations

import math
from pathlib import Path

# Keys of an observation-group config that are not observation terms.
OBS_META_KEYS = frozenset(
    {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}
)

# Go2 actuated leg joints (hip/thigh/calf x 4 legs). Needed because the reference yaml stores joint
# patterns, not expanded names; see module docstring.
GO2_JOINT_COUNT = 12

# Fields whose reference/live values must match for a checkpoint to be evaluated under the same contract.
NUMERIC_TOLERANCE = 1e-6

# rsl-rl renamed its actor-critic network between the stacks: 3.x calls it ``ActorCritic``, 5.x
# ``MLPModel``. The name is library vocabulary, not architecture -- the architecture is pinned by
# ``actor_hidden_dims`` (and would be pinned by an explicit class name for anything non-MLP). Mapping
# the known pair keeps the comparison meaningful instead of either failing on a rename or dropping the
# field.
EQUIVALENT_MODEL_CLASSES = {"ActorCritic": "mlp-actor-critic", "MLPModel": "mlp-actor-critic"}


def canonical_model_class(name):
    """Normalise a known-equivalent model class name; anything unknown passes through unchanged."""
    if name is None:
        return None
    return EQUIVALENT_MODEL_CLASSES.get(name, name)


def _tolerant_loader():
    """A SafeLoader that accepts the ``!!python/...`` tags Isaac Lab's dumper emits.

    Unknown python tags are loaded as ``None``: none of the fields compared here need them (they are
    slices and tuples of values we read from a different key), and refusing to load would make the
    whole reference unreadable.
    """
    import yaml

    class TolerantLoader(yaml.SafeLoader):
        pass

    def _python_any(loader, suffix, node):
        # ``!!python/tuple`` is loadable as a plain sequence and several compared fields use it
        # (command ranges, resampling interval). Everything else (object/apply slices, objects) is
        # only used by fields this module never reads, so None is enough to keep the file loadable.
        if suffix == "tuple":
            return tuple(loader.construct_sequence(node, deep=True))
        return None

    TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_any)
    return TolerantLoader


def load_yaml_tolerant(path: str | Path):
    import yaml

    with open(path) as f:
        return yaml.load(f, Loader=_tolerant_loader())


def _term_entry(value) -> dict:
    """One observation/action term, normalised to a mapping.

    Isaac Lab's config dumper writes a term either as its full cfg (mapping) or, when nothing was
    customised, as the bare ``"module:func"`` string. Both forms appear in dumps from different
    versions, so the reader accepts either.
    """
    if isinstance(value, str):
        return {"func": value}
    if isinstance(value, dict):
        return value
    return {}


def _func_leaf(func) -> str | None:
    """``'isaaclab.envs.mdp.observations:joint_pos_rel'`` -> ``'joint_pos_rel'``."""
    if not isinstance(func, str):
        return None
    return func.split(":")[-1].split(".")[-1]


def _term_width(name: str, term_cfg: dict, env_yaml: dict) -> int | None:
    """Width of one observation term, or None when it depends on state we cannot read."""
    if name in ("base_lin_vel", "base_ang_vel", "projected_gravity"):
        return 3
    if name in ("velocity_commands", "generated_commands"):
        ranges = (env_yaml.get("commands") or {}).get("base_velocity", {}).get("ranges") or {}
        # "heading" is the target-yaw range used to derive ang_vel_z when heading_command is on; it is
        # not an extra command dimension (the command stays [lin_vel_x, lin_vel_y, ang_vel_z]).
        return len([k for k in ranges if k != "heading"]) or None
    if name in ("joint_pos", "joint_vel", "actions"):
        return GO2_JOINT_COUNT
    if name == "height_scan":
        # size/resolution live on the sensor the term references, not on the term itself.
        sensor_cfg = ((term_cfg or {}).get("params") or {}).get("sensor_cfg") or {}
        sensor_name = sensor_cfg.get("name")
        sensor = (env_yaml.get("scene") or {}).get(sensor_name) or {}
        pattern = sensor.get("pattern_cfg") or {}
        resolution = pattern.get("resolution")
        size = pattern.get("size")
        if isinstance(resolution, (int, float)) and isinstance(size, (list, tuple)) and len(size) == 2 and resolution:
            cells_x = round(size[0] / resolution) + 1
            cells_y = round(size[1] / resolution) + 1
            return int(cells_x * cells_y)
    return None


def load_reference(run_dir: str | Path) -> dict:
    """Extract the interface contract from a run directory's dumped ``params/`` yaml files."""
    run_dir = Path(run_dir)
    env_yaml = load_yaml_tolerant(run_dir / "params" / "env.yaml")
    agent_path = run_dir / "params" / "agent.yaml"
    agent_yaml = load_yaml_tolerant(agent_path) if agent_path.is_file() else {}

    obs_cfg = (env_yaml.get("observations") or {}).get("policy") or {}
    term_names = [k for k in obs_cfg if k not in OBS_META_KEYS]
    term_entries = {name: _term_entry(obs_cfg[name]) for name in term_names}
    term_widths = {name: _term_width(name, term_entries[name], env_yaml) for name in term_names}
    known_widths = [w for w in term_widths.values() if w is not None]

    action_cfg = _term_entry((env_yaml.get("actions") or {}).get("joint_pos"))
    command_cfg = (env_yaml.get("commands") or {}).get("base_velocity") or {}
    ranges = command_cfg.get("ranges") or {}
    policy_cfg = agent_yaml.get("policy") or {}
    algorithm = agent_yaml.get("algorithm") or {}

    return {
        "run_dir": str(run_dir),
        "source": "params/env.yaml+agent.yaml",
        "obs_group": "policy",
        "obs_term_names": term_names,
        "obs_term_funcs": [_func_leaf(term_entries[n].get("func")) for n in term_names],
        "obs_term_widths": term_widths,
        "obs_dim": sum(known_widths) if len(known_widths) == len(term_names) else None,
        "enable_corruption": obs_cfg.get("enable_corruption"),
        "action_dim": GO2_JOINT_COUNT,
        "action_scale": action_cfg.get("scale"),
        "action_offset": action_cfg.get("offset"),
        "action_use_default_offset": action_cfg.get("use_default_offset"),
        "decimation": env_yaml.get("decimation"),
        "sim_dt": (env_yaml.get("sim") or {}).get("dt"),
        "step_dt": _step_dt(env_yaml.get("sim", {}).get("dt"), env_yaml.get("decimation")),
        "command_ranges": {k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in ranges.items()},
        "standing_env_fraction": command_cfg.get("rel_standing_envs"),
        "command_resampling_s": tuple(command_cfg.get("resampling_time_range") or ()) or None,
        "seed": agent_yaml.get("seed"),
        "num_envs": (env_yaml.get("scene") or {}).get("num_envs"),
        "max_iterations": agent_yaml.get("max_iterations"),
        "num_steps_per_env": agent_yaml.get("num_steps_per_env"),
        "empirical_normalization": agent_yaml.get("empirical_normalization"),
        "actor_obs_normalization": policy_cfg.get("actor_obs_normalization"),
        "critic_obs_normalization": policy_cfg.get("critic_obs_normalization"),
        "actor_hidden_dims": list(policy_cfg.get("actor_hidden_dims") or []),
        "actor_class": canonical_model_class(policy_cfg.get("class_name")),
        "algorithm_class": algorithm.get("class_name"),
        "learning_rate": algorithm.get("learning_rate"),
        "gamma": algorithm.get("gamma"),
        "lam": algorithm.get("lam"),
        "num_learning_epochs": algorithm.get("num_learning_epochs"),
        "num_mini_batches": algorithm.get("num_mini_batches"),
        "entropy_coef": algorithm.get("entropy_coef"),
        "desired_kl": algorithm.get("desired_kl"),
    }


def _step_dt(dt, decimation):
    try:
        return float(dt) * int(decimation)
    except (TypeError, ValueError):
        return None


def _ranges_of(commands_cfg, name: str = "base_velocity") -> dict:
    """Command ranges from a live config object (``env_cfg.commands.base_velocity.ranges``)."""
    commands = getattr(commands_cfg, name, None)
    ranges = getattr(commands, "ranges", None)
    if ranges is None:
        return {}
    out = {}
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
        value = getattr(ranges, key, None)
        if value is not None:
            out[key] = tuple(value)
    return out


def _unset(value):
    """``dataclasses.MISSING`` means "not set", which the reference yaml spells as ``null``."""
    try:
        from dataclasses import MISSING
    except ImportError:  # pragma: no cover
        return value
    return None if isinstance(value, type(MISSING)) or value is MISSING else value


def live_contract(env_cfg, agent_cfg, env=None) -> dict:
    """Extract the same contract fields from freshly built configs.

    ``env`` is optional: when given (a gym env or its unwrapped form), the observation term order and
    dims are read from the observation manager, i.e. what the running policy will actually see.
    """
    obs_cfg = getattr(env_cfg.observations, "policy", None)
    term_names = [k for k in vars(obs_cfg) if k not in OBS_META_KEYS] if obs_cfg is not None else []
    term_widths: dict[str, int | None] = {}
    if env is not None:
        manager = getattr(env, "observation_manager", None) or getattr(getattr(env, "unwrapped", None), "observation_manager", None)
        if manager is not None:
            names = list(manager.active_terms["policy"])
            dims = [int(d[0]) for d in manager.group_obs_term_dim["policy"]]
            term_names = names
            term_widths = dict(zip(names, dims, strict=True))

    actions_cfg = getattr(env_cfg, "actions", None)
    action_cfg = getattr(actions_cfg, "joint_pos", None) if actions_cfg is not None else None
    asset = getattr(getattr(env_cfg, "scene", None), "robot", None)
    joint_count = len(getattr(asset, "joint_names", None) or []) or GO2_JOINT_COUNT

    commands_cfg = getattr(env_cfg, "commands", None)
    command_cfg = getattr(commands_cfg, "base_velocity", None) if commands_cfg is not None else None

    # rsl-rl >= 4 (the GA stack) splits the old `policy` block into `actor` / `critic`; the
    # September reference files still have the `policy` shape. Read whichever exists.
    policy_cfg = _unset(getattr(agent_cfg, "policy", None))
    actor_cfg = _unset(getattr(agent_cfg, "actor", None))
    critic_cfg = _unset(getattr(agent_cfg, "critic", None))
    hidden = None
    if actor_cfg is not None:
        hidden = _unset(getattr(actor_cfg, "hidden_dims", None))
    elif policy_cfg is not None:
        hidden = _unset(getattr(policy_cfg, "actor_hidden_dims", None))
    algorithm = getattr(agent_cfg, "algorithm", None)

    return {
        "obs_group": "policy",
        "obs_term_names": term_names,
        "obs_term_funcs": None,
        "obs_term_widths": term_widths or None,
        "obs_dim": sum(w for w in term_widths.values() if w is not None) if term_widths else None,
        "enable_corruption": getattr(obs_cfg, "enable_corruption", None),
        "action_dim": joint_count,
        "action_scale": getattr(action_cfg, "scale", None),
        "action_offset": getattr(action_cfg, "offset", None),
        "action_use_default_offset": getattr(action_cfg, "use_default_offset", None),
        "decimation": getattr(env_cfg, "decimation", None),
        "sim_dt": getattr(env_cfg.sim, "dt", None),
        "step_dt": _step_dt(getattr(env_cfg.sim, "dt", None), getattr(env_cfg, "decimation", None)),
        "command_ranges": _ranges_of(commands_cfg),
        "standing_env_fraction": getattr(command_cfg, "rel_standing_envs", None),
        "command_resampling_s": tuple(getattr(command_cfg, "resampling_time_range", ()) or ()) or None,
        "seed": getattr(agent_cfg, "seed", None),
        "num_envs": getattr(getattr(env_cfg, "scene", None), "num_envs", None),
        "max_iterations": getattr(agent_cfg, "max_iterations", None),
        "num_steps_per_env": getattr(agent_cfg, "num_steps_per_env", None),
        "empirical_normalization": _unset(getattr(agent_cfg, "empirical_normalization", None)),
        "actor_obs_normalization": _normalization_flag(actor_cfg, policy_cfg, "actor"),
        "critic_obs_normalization": _normalization_flag(critic_cfg, policy_cfg, "critic"),
        "actor_hidden_dims": list(hidden) if hidden else [],
        "actor_class": canonical_model_class(
            _unset(getattr(actor_cfg, "class_name", None))
            if actor_cfg is not None
            else _unset(getattr(policy_cfg, "class_name", None))
        ),
        "algorithm_class": getattr(algorithm, "class_name", None),
        "learning_rate": getattr(algorithm, "learning_rate", None),
        "gamma": getattr(algorithm, "gamma", None),
        "lam": getattr(algorithm, "lam", None),
        "num_learning_epochs": getattr(algorithm, "num_learning_epochs", None),
        "num_mini_batches": getattr(algorithm, "num_mini_batches", None),
        "entropy_coef": getattr(algorithm, "entropy_coef", None),
        "desired_kl": getattr(algorithm, "desired_kl", None),
    }


def _normalization_flag(model_cfg, policy_cfg, which: str):
    """Observation-normalisation flag, in either config layout (``obs_normalization`` / ``*_obs_normalization``)."""
    if model_cfg is not None:
        value = _unset(getattr(model_cfg, "obs_normalization", None))
        if value is not None:
            return value
    if policy_cfg is not None:
        return _unset(getattr(policy_cfg, f"{which}_obs_normalization", None))
    return None


def _is_missing(value) -> bool:
    from dataclasses import MISSING

    return isinstance(value, type(MISSING))


def _values_match(ref, live) -> bool:
    if ref is None or live is None:
        return ref is None and live is None
    if isinstance(ref, (int, float)) and isinstance(live, (int, float)):
        if isinstance(ref, bool) or isinstance(live, bool):
            return bool(ref) == bool(live)
        return math.isclose(float(ref), float(live), rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE)
    if isinstance(ref, (list, tuple)) and isinstance(live, (list, tuple)):
        return len(ref) == len(live) and all(_values_match(a, b) for a, b in zip(ref, live, strict=True))
    if isinstance(ref, dict) and isinstance(live, dict):
        return ref.keys() == live.keys() and all(_values_match(v, live[k]) for k, v in ref.items())
    return ref == live


# Fields that describe the run (not the interface the checkpoint sees): reported, never gating.
INFORMATIONAL_FIELDS = frozenset({"num_envs", "seed", "max_iterations", "enable_corruption", "run_dir", "source"})


def compare(reference: dict, live: dict) -> list[dict]:
    """Field-by-field diff -> rows of {field, reference, live, ok, observed}.

    ``ok`` is None for informational fields; ``observed`` is False when the live side could not
    report the field at all (which is INCOMPLETE, not a mismatch -- a missing measurement is not
    evidence of sameness, so it must never read as "ok").
    """
    rows = []
    for field, ref_value in reference.items():
        if field in ("run_dir", "source"):
            continue
        live_value = live.get(field)
        # A field that is legitimately None on both sides (e.g. `empirical_normalization: null` means
        # "disabled") is an observation, not a gap: both sides agree it is absent.
        observed = live_value is not None or ref_value is None
        if field in INFORMATIONAL_FIELDS or not observed:
            ok = None
        else:
            ok = _values_match(ref_value, live_value)
        rows.append({"field": field, "reference": ref_value, "live": live_value, "ok": ok, "observed": observed})
    return rows


# Fields that are derived from others: unobserved is fine (obs_dim already covers the totals), but
# an observed value still gates.
DERIVED_FIELDS = frozenset({"obs_term_widths", "obs_term_funcs"})


def verdict(rows: list[dict]) -> str:
    """CONTRACT OK / CONTRACT MISMATCH / INCOMPLETE (a gating field could not be observed)."""
    gating = [r for r in rows if r["field"] not in INFORMATIONAL_FIELDS]
    if any(r["ok"] is False for r in gating):
        return "CONTRACT MISMATCH"
    if any(not r["observed"] and r["field"] not in DERIVED_FIELDS for r in gating):
        return "INCOMPLETE"
    return "CONTRACT OK"


def format_table(rows: list[dict]) -> str:
    """Human-readable table; mismatches are marked so they cannot be skimmed past."""
    lines = [f"{'field':32} {'reference':38} {'live':38} status"]
    for r in rows:
        if r["ok"] is None:
            status = "info" if r["observed"] or r["field"] in INFORMATIONAL_FIELDS else "not observed"
        else:
            status = "ok" if r["ok"] else "MISMATCH"
        lines.append(f"{r['field']:32} {str(r['reference'])[:38]:38} {str(r['live'])[:38]:38} {status}")
    return "\n".join(lines)


def mismatch_summary(rows: list[dict]) -> list[str]:
    """One line per mismatch, for the report and the eval log."""
    out = []
    for r in rows:
        if r["ok"] is False:
            out.append(f"{r['field']}: reference={r['reference']!r} live={r['live']!r}")
    return out
