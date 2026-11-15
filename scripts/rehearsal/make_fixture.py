#!/usr/bin/env python3
"""Create the tiny rehearsal scene (a few KB) under worlds/rehearsal/ with the real layout and scene-id forms."""
from __future__ import annotations

import sys
from pathlib import Path

USD = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "scene"
    {
        def Mesh "floor"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]
            uniform token subdivisionScheme = "none"
        }
        def Mesh "wall"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-2, 1, 0), (2, 1, 0), (2, 1, 1.5), (-2, 1, 1.5)]
            uniform token subdivisionScheme = "none"
        }
    }
}
"""


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rehearsal_fixture")
    scene = root / "scenes" / "tiny_usd"
    (root / "Materials").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    scene.mkdir(parents=True, exist_ok=True)
    (scene / "start_result_tiny.usd").write_text(USD)
    (scene / "metadata.json").write_text('{"scene_id": "tiny", "rehearsal": true}\n')
    (root / "Materials" / "README.txt").write_text("rehearsal stub, no materials\n")
    (root / "models" / "README.txt").write_text("rehearsal stub, no models\n")
    (root / "manifest.json").write_text('{"rehearsal": true, "scene_ids": ["tiny"]}\n')
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
