"""Registration import for the fused Go2 + ARX5 platform.

``isaaclab_tasks.utils.import_packages`` only imports each package's ``__init__.py`` -- never leaf modules -- so this
line is what runs ``platform.py``'s ``register_platform(GO2_ARX5)`` (same contract as ``core/platforms/go2``).
"""

from policy.locomotion.core.platforms.go2_arx5 import platform  # noqa: F401
