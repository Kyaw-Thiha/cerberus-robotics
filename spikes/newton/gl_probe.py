"""Report whether this container can give the Newton GL visualizer a graphics context.

Kit-less Newton does not need a GPU renderer for physics, but ``--video`` records frames through a
visualizer (``newton_gl``), which uses pyglet's EGL backend when ``$DISPLAY`` is unset. This probe
records what the container actually has, so the video result can be read as "EGL worked" or "EGL was
unavailable and the recorder fell back", not guessed.

    python -m spikes.newton.gl_probe
"""

from __future__ import annotations

import ctypes.util
import json
import os
import sys


def probe() -> dict:
    report: dict = {
        "display": os.environ.get("DISPLAY"),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        "egl_platform": os.environ.get("EGL_PLATFORM"),
        "dev_dri": os.path.isdir("/dev/dri"),
        "libs": {},
    }
    for name in ("EGL", "GL", "OpenGL", "GLX", "Xrender", "vulkan"):
        report["libs"][name] = ctypes.util.find_library(name)
    try:
        import pyglet

        report["pyglet_version"] = pyglet.version
        os.environ.pop("DISPLAY", None)  # force the headless EGL path
        display = pyglet.canvas.get_display()
        report["pyglet_display_class"] = type(display).__name__
        try:
            window = pyglet.window.Window(width=64, height=64, visible=False)
            report["offscreen_window"] = True
            window.close()
        except Exception as exc:  # noqa: BLE001 - the whole point is to record the failure
            report["offscreen_window"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        report["pyglet_version"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        import OpenGL.GL as gl

        report["opengl_version_string"] = gl.glGetString(gl.GL_VERSION)
    except Exception as exc:  # noqa: BLE001
        report["opengl_version_string"] = f"unavailable: {type(exc).__name__}: {exc}"
    return report


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    data = probe()
    text = json.dumps(data, indent=2)
    print(text)
    if out:
        from pathlib import Path

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
