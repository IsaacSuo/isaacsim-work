"""Measure the connected horizontal support around Blender world origin."""

import json
import sys
from collections import deque
from pathlib import Path

import bpy
from mathutils import Vector


OUTPUT = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
TARGET_Z = -0.5130774229783968
STEP = 0.02
RADIUS = 1.5
COUNT = int(round(2.0 * RADIUS / STEP)) + 1
depsgraph = bpy.context.evaluated_depsgraph_get()
scene = bpy.context.scene


def coordinate(index):
    return -RADIUS + index * STEP


hits = {}
for ix in range(COUNT):
    x = coordinate(ix)
    for iy in range(COUNT):
        y = coordinate(iy)
        hit, location, _normal, _face, obj, _matrix = scene.ray_cast(
            depsgraph, Vector((x, y, 2.0)), Vector((0.0, 0.0, -1.0)), distance=5.0
        )
        if hit and abs(float(location.z) - TARGET_Z) <= 0.055:
            hits[(ix, iy)] = {"z": float(location.z), "object": obj.name}

origin_index = int(round(RADIUS / STEP))
seeds = [
    key for key in hits
    if abs(key[0] - origin_index) <= 3 and abs(key[1] - origin_index) <= 3
]
if not seeds:
    raise RuntimeError("No tabletop ray hits were found around Blender origin")

component = set()
queue = deque(seeds)
while queue:
    cell = queue.popleft()
    if cell in component or cell not in hits:
        continue
    component.add(cell)
    ix, iy = cell
    queue.extend(((ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)))

xs = [coordinate(ix) for ix, _iy in component]
ys = [coordinate(iy) for _ix, iy in component]
objects = sorted({hits[cell]["object"] for cell in component})
report = {
    "blend": bpy.data.filepath,
    "target_z": TARGET_Z,
    "grid_step": STEP,
    "connected_samples": len(component),
    "blender_bounds": {
        "min_x": min(xs) - STEP * 0.5,
        "max_x": max(xs) + STEP * 0.5,
        "min_y": min(ys) - STEP * 0.5,
        "max_y": max(ys) + STEP * 0.5,
    },
    "isaac_bounds": {
        "min_x": min(xs) - STEP * 0.5,
        "max_x": max(xs) + STEP * 0.5,
        "min_z": -(max(ys) + STEP * 0.5),
        "max_z": -(min(ys) - STEP * 0.5),
    },
    "hit_objects": objects,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
