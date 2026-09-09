"""isaaclab_tasks.utils.import_packages only auto-imports the __init__.py of
each package it walks into -- it never imports leaf modules like platform.py
on its own (confirmed by reading its source: it calls __import__ on a
sub-package's name, not on every module pkgutil.iter_modules yields). This
import is what actually triggers platform.py's register_platform(GO2) call
when the parent core/platforms/__init__.py's import_packages() walk reaches
this package -- matches LeggedManip_Lab's own convention of each combo's
__init__.py doing its own registration import, not relying on the walker to
find leaf modules.
"""

from policy.locomotion.core.platforms.go2 import platform  # noqa: F401
