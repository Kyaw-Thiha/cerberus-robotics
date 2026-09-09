"""Cerberus locomotion task registration. Importing this module auto-discovers
and registers every platform package under core/platforms/ -- add a platform
by adding a new core/platforms/<name>/ directory (see
core/platforms/registry.py for what it must define), never by editing this
file.
"""

from isaaclab_tasks.utils import import_packages

# "registry" holds the shared PlatformCfg contract itself, not a platform --
# skip it so import_packages doesn't try to treat it as one.
_BLACKLIST_PKGS = ["registry"]

import_packages(__name__ + ".core.platforms", _BLACKLIST_PKGS)
