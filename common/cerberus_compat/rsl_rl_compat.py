"""rsl-rl adapters spanning the old and the new Isaac Lab line.

* OLD stack: Isaac Lab v2.3.2, rsl-rl-lib 3.1.2 (``runner.alg.policy`` is one ActorCritic, the runner owns
  ``log``/``save``, checkpoints hold ``model_state_dict``).
* NEW stack: Isaac Lab 3.0 line, rsl-rl-lib 5.x (``runner.alg.get_policy()`` is the actor model, logging moved
  to ``runner.logger``, checkpoints hold ``actor_state_dict`` + ``critic_state_dict`` or
  ``student_state_dict`` + ``teacher_state_dict``). rsl-rl 4.x already has the new API, so every
  branch below is ``major >= 4`` vs ``major < 4``.

Everything that touches ``rsl_rl``, ``isaaclab`` or ``torch`` is imported lazily inside functions, so this
module imports on a machine without any of them. Branches for ``major >= 4`` are not exercised on this
repo's old-stack CI; they are marked ``# UNVERIFIED:`` and must be checked on a Lab 3.0 pod.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
import sys
from collections.abc import Mapping

RSL_RL_DIST_NAME = "rsl-rl-lib"
NEW_API_MIN_MAJOR = 4
"""First rsl-rl major with the split logger / actor+critic models / split checkpoint keys."""

LEGACY_CHECKPOINT_KEY = "model_state_dict"
SPLIT_CHECKPOINT_KEYS = (
    "actor_state_dict",
    "critic_state_dict",
    "student_state_dict",
    "teacher_state_dict",
)

CHECKPOINT_FORMAT_LEGACY = "legacy"
CHECKPOINT_FORMAT_SPLIT = "split"
CHECKPOINT_FORMAT_UNKNOWN = "unknown"


class CheckpointFormatError(ValueError):
    """A checkpoint's key layout does not match the installed rsl-rl-lib."""


# --------------------------------------------------------------------------------------
# Version
# --------------------------------------------------------------------------------------


def rsl_rl_version() -> str:
    """Installed ``rsl-rl-lib`` version string, e.g. ``'3.1.2'`` or ``'5.5.1'``."""
    try:
        return importlib.metadata.version(RSL_RL_DIST_NAME)
    except importlib.metadata.PackageNotFoundError as e:
        raise RuntimeError(
            f"The '{RSL_RL_DIST_NAME}' distribution is not installed; the locomotion train/play scripts need it "
            "(old stack: rsl-rl-lib 3.1.2, new stack: rsl-rl-lib 5.x)."
        ) from e


def rsl_rl_major() -> int:
    """Major version of the installed ``rsl-rl-lib`` (3 on the old stack, 5 on the new one)."""
    version = rsl_rl_version()
    head = version.strip().split(".")[0]
    digits = "".join(ch for ch in head if ch.isdigit())
    if not digits:
        raise RuntimeError(
            f"Cannot parse the major version from {RSL_RL_DIST_NAME} version {version!r}."
        )
    return int(digits)


def uses_new_rsl_rl_api(major: int | None = None) -> bool:
    """True for rsl-rl >= 4 (``runner.logger``, split checkpoints, model-based policy)."""
    return (rsl_rl_major() if major is None else major) >= NEW_API_MIN_MAJOR


# --------------------------------------------------------------------------------------
# Script plumbing that moved between Lab 2.3.2 and Lab 3.0
# --------------------------------------------------------------------------------------


def load_cli_args():
    """Return Isaac Lab's RSL-RL ``cli_args`` module (``add_rsl_rl_args`` / ``update_rsl_rl_cfg``).

    Lab 2.3.2 ships it as ``scripts/reinforcement_learning/rsl_rl/cli_args.py`` next to its train script
    (found through ``$ISAACLAB_PATH``, put on ``sys.path`` exactly as the scripts used to do inline).
    Lab 3.0 moved it into the package as ``isaaclab_rl.entrypoints.backends.cli_args_rsl_rl``.
    """
    scripts_dir = os.path.join(
        os.environ.get("ISAACLAB_PATH", ""),
        "scripts",
        "reinforcement_learning",
        "rsl_rl",
    )
    if os.path.isfile(os.path.join(scripts_dir, "cli_args.py")):
        sys.path.insert(0, scripts_dir)
        import cli_args  # isort: skip

        return cli_args
    # UNVERIFIED: import-time side effects of isaaclab_rl.entrypoints.common before the app is launched.
    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl

    return cli_args_rsl_rl


def hydra_task_config(task_name: str, agent_cfg_entry_point: str):
    """``isaaclab_tasks.utils.hydra.hydra_task_config`` on Lab 2.3.2; a ``resolve_task_config`` shim on Lab 3.0.

    Lab 3.0 removed the decorator. The shim keeps the scripts' ``@hydra_task_config(task, agent)`` shape:
    it resolves ``(env_cfg, agent_cfg)`` from the registry plus the Hydra overrides in ``sys.argv`` and calls
    the wrapped function with them.
    """
    try:
        from isaaclab_tasks.utils.hydra import (
            hydra_task_config as lab_hydra_task_config,
        )
    except ImportError:
        lab_hydra_task_config = None
    if lab_hydra_task_config is not None:
        return lab_hydra_task_config(task_name, agent_cfg_entry_point)

    # UNVERIFIED: Lab 3.0 branch (resolve_task_config signature taken from release/3.0.0 source).
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from isaaclab_tasks.utils import resolve_task_config

            env_cfg, agent_cfg = resolve_task_config(task_name, agent_cfg_entry_point)
            return func(env_cfg, agent_cfg, *args, **kwargs)

        return wrapper

    return decorator


# --------------------------------------------------------------------------------------
# Config migration + runner construction
# --------------------------------------------------------------------------------------


def _is_missing(value) -> bool:
    from dataclasses import MISSING

    return isinstance(value, type(MISSING))


def _looks_migrated(agent_cfg) -> bool:
    """Lab's ``handle_deprecated_rsl_rl_cfg`` is not idempotent for rsl-rl >= 5: it deletes the legacy
    ``stochastic`` fields on the model cfgs and then reads them again on a second pass."""
    for name in ("actor", "critic", "student", "teacher"):
        model = getattr(agent_cfg, name, None)
        if model is None or _is_missing(model):
            continue
        return not hasattr(model, "stochastic")
    return False


def migrate_agent_cfg(agent_cfg):
    """Apply Lab's ``handle_deprecated_rsl_rl_cfg`` to a runner cfg object for the installed rsl-rl.

    No-op on rsl-rl 3.x (the old stack passes the cfg through untouched, as before). On >= 4 it delegates
    to ``isaaclab_rl.rsl_rl.handle_deprecated_rsl_rl_cfg`` (mutates and returns ``agent_cfg``) rather than
    re-implementing the ``policy`` -> ``actor``/``critic`` + ``distribution_cfg`` conversion; skipped if that
    helper does not exist or the cfg was already migrated.
    """
    major = rsl_rl_major()
    if major < NEW_API_MIN_MAJOR:
        return agent_cfg
    # UNVERIFIED: new-stack branch, needs Isaac Lab 3.0 + rsl-rl-lib 5.x.
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    except ImportError:
        return agent_cfg
    if _looks_migrated(agent_cfg):
        return agent_cfg
    return handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version())


def _cfg_to_dict(agent_cfg) -> dict:
    if hasattr(agent_cfg, "to_dict"):
        return agent_cfg.to_dict()
    from isaaclab.utils import (
        to_dict,  # Lab 3.0 helper name; UNVERIFIED for cfgs without .to_dict()
    )

    return to_dict(agent_cfg)


def make_runner(env, agent_cfg_dict, log_dir, device, runner_cls=None):
    """Build an ``OnPolicyRunner`` (or ``DistillationRunner``, or ``runner_cls``) for either rsl-rl major.

    Args:
        env: ``RslRlVecEnvWrapper``-wrapped environment.
        agent_cfg_dict: Either the runner cfg *object* (``RslRlBaseRunnerCfg``; preferred -- on rsl-rl >= 4
            it is first migrated in place with Lab's ``handle_deprecated_rsl_rl_cfg`` and then converted with
            ``.to_dict()``) or an already-converted/migrated ``dict`` (used as is: no migration is possible
            on a dict).
        log_dir: Run directory, or None for inference.
        device: Torch device string for the runner.
        runner_cls: Runner class to instantiate, e.g. ``BestCheckpointOnPolicyRunner``. If None, chosen
            from the cfg's ``class_name`` (``OnPolicyRunner`` / ``DistillationRunner``).

    On rsl-rl 3.x this is exactly ``runner_cls(env, cfg.to_dict(), log_dir=log_dir, device=device)``.
    """
    if isinstance(agent_cfg_dict, Mapping):
        train_cfg = agent_cfg_dict
    else:
        # UNVERIFIED: the migration is a no-op on 3.x and only does work on the new stack.
        agent_cfg_dict = migrate_agent_cfg(agent_cfg_dict)
        train_cfg = _cfg_to_dict(agent_cfg_dict)

    if runner_cls is None:
        runner_cls = _default_runner_cls(train_cfg)
    return runner_cls(env, train_cfg, log_dir=log_dir, device=device)


def _default_runner_cls(train_cfg: Mapping):
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    class_name = train_cfg.get("class_name")
    if class_name == "OnPolicyRunner":
        return OnPolicyRunner
    if class_name == "DistillationRunner":
        return DistillationRunner
    raise ValueError(f"Unsupported runner class: {class_name}")


# --------------------------------------------------------------------------------------
# Policy access + export
# --------------------------------------------------------------------------------------


def get_policy_for_export(runner):
    """The torch module that carries the deployed policy and has ``reset(dones)``.

    rsl-rl 3.x: ``runner.alg.policy`` (older: ``runner.alg.actor_critic``), an ActorCritic / StudentTeacher.
    rsl-rl >= 4: ``runner.alg.get_policy()``, the actor (or student) model.
    """
    if rsl_rl_major() < NEW_API_MIN_MAJOR:
        try:
            return runner.alg.policy
        except AttributeError:
            return runner.alg.actor_critic
    # UNVERIFIED: new stack.
    return runner.alg.get_policy()


def _legacy_obs_normalizer(policy_nn):
    if hasattr(policy_nn, "actor_obs_normalizer"):
        return policy_nn.actor_obs_normalizer
    if hasattr(policy_nn, "student_obs_normalizer"):
        return policy_nn.student_obs_normalizer
    return None


def export_policy(
    runner,
    out_dir: str,
    jit_filename: str = "policy.pt",
    onnx_filename: str = "policy.onnx",
) -> None:
    """Export the runner's policy as TorchScript and ONNX into ``out_dir``.

    rsl-rl 3.x: Lab's ``export_policy_as_jit`` / ``export_policy_as_onnx`` with the actor/student obs normalizer,
    exactly as the old play script did. rsl-rl >= 4: the runner's own ``export_policy_to_jit`` /
    ``export_policy_to_onnx`` (what Isaac Lab 3.0's play entrypoint calls); Lab 3.0's ``export_policy_as_*``
    expect a module with ``.actor`` and are not used because the 5.x models export themselves.
    """
    if rsl_rl_major() < NEW_API_MIN_MAJOR:
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

        policy_nn = get_policy_for_export(runner)
        normalizer = _legacy_obs_normalizer(policy_nn)
        export_policy_as_jit(
            policy_nn, normalizer=normalizer, path=out_dir, filename=jit_filename
        )
        export_policy_as_onnx(
            policy_nn, normalizer=normalizer, path=out_dir, filename=onnx_filename
        )
        return
    # UNVERIFIED: new stack.
    runner.export_policy_to_jit(path=out_dir, filename=jit_filename)
    runner.export_policy_to_onnx(path=out_dir, filename=onnx_filename)


# --------------------------------------------------------------------------------------
# Checkpoint format guard
# --------------------------------------------------------------------------------------


def detect_checkpoint_format(loaded) -> str:
    """Classify an already-loaded checkpoint dict as ``'legacy'``, ``'split'`` or ``'unknown'``."""
    if not isinstance(loaded, Mapping):
        return CHECKPOINT_FORMAT_UNKNOWN
    if any(key in loaded for key in SPLIT_CHECKPOINT_KEYS):
        return CHECKPOINT_FORMAT_SPLIT
    if LEGACY_CHECKPOINT_KEY in loaded:
        return CHECKPOINT_FORMAT_LEGACY
    return CHECKPOINT_FORMAT_UNKNOWN


def check_checkpoint_format(path: str, major: int | None = None) -> str:
    """Fail fast, with a message naming both formats, if ``path`` cannot load on the installed rsl-rl.

    Loads the file on CPU (``torch.load(..., map_location='cpu', weights_only=False)``, same trust level as
    the runner's own load) and compares its keys with the installed major:

    * legacy (rsl-rl 3.x: ``model_state_dict``) opened on rsl-rl >= 4 -> :class:`CheckpointFormatError`
    * split (rsl-rl >= 4: ``actor_state_dict``/``critic_state_dict`` or ``student_state_dict``) opened on
      rsl-rl 3.x -> :class:`CheckpointFormatError`

    Returns the detected format (``'legacy'``/``'split'``/``'unknown'``). Unreadable or unrecognised files
    return ``'unknown'`` and are left for ``runner.load`` to report.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    import torch

    major = rsl_rl_major() if major is None else major
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 -- corrupt file etc: let runner.load raise its own, more specific error
        print(
            f"[WARNING] Could not pre-read checkpoint {path} to check its format ({e})."
        )
        return CHECKPOINT_FORMAT_UNKNOWN

    fmt = detect_checkpoint_format(loaded)
    new_api = major >= NEW_API_MIN_MAJOR
    if fmt == CHECKPOINT_FORMAT_LEGACY and new_api:
        raise CheckpointFormatError(
            f"Checkpoint '{path}' is in the LEGACY rsl-rl 3.x format (single '{LEGACY_CHECKPOINT_KEY}' key), but "
            f"rsl-rl-lib {rsl_rl_version()} (new stack, major {major}) expects the SPLIT format "
            "('actor_state_dict' + 'critic_state_dict', or 'student_state_dict' for distillation). Legacy "
            "checkpoints do not load on the new stack: run/export it on the OLD stack (Isaac Sim 5.1.0, "
            "Isaac Lab v2.3.2, rsl-rl-lib 3.1.2) or retrain."
        )
    if fmt == CHECKPOINT_FORMAT_SPLIT and not new_api:
        raise CheckpointFormatError(
            f"Checkpoint '{path}' is in the SPLIT rsl-rl >= 4 format ('actor_state_dict' / 'critic_state_dict' / "
            f"'student_state_dict'), but rsl-rl-lib {rsl_rl_version()} (old stack, major {major}) expects the "
            f"LEGACY format (single '{LEGACY_CHECKPOINT_KEY}' key). Load it on the NEW stack (Isaac Sim 6.1.0, "
            "Isaac Lab 3.0, rsl-rl-lib 5.x) instead."
        )
    return fmt
