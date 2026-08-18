"""Audit hospital pedestal objects and horizontal cross-sections near origin."""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


OUTPUT = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()


def world_bounds(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return {
        "min": [min(point[axis] for point in corners) for axis in range(3)],
        "max": [max(point[axis] for point in corners) for axis in range(3)],
    }


nearby = []
for obj in bpy.data.objects:
    if obj.type != "MESH" or obj.hide_render:
        continue
    bounds = world_bounds(obj)
    lo, hi = bounds["min"], bounds["max"]
    if hi[0] < -1.5 or lo[0] > 1.5 or hi[1] < -1.5 or lo[1] > 1.5:
        continue
    if hi[2] < -4.0 or lo[2] > 0.5:
        continue
    nearby.append(
        {
            "name": obj.name,
            "bounds": bounds,
            "vertices": len(obj.data.vertices),
            "polygons": len(obj.data.polygons),
        }
    )


def nearest_axis_hit(origin, direction, axis, limit=1.5):
    hit, location, normal, face, obj, _matrix = scene.ray_cast(
        depsgraph, Vector(origin), Vector(direction), distance=4.0
    )
    if not hit or abs(float(location[axis])) > limit:
        return None
    return {
        "location": list(location),
        "normal": list(normal),
        "object": obj.name,
        "face": int(face),
    }


sections = []
for index in range(51):
    z = -0.55 - index * 0.05
    sections.append(
        {
            "z": z,
            "from_neg_x": nearest_axis_hit((-2.0, 0.0, z), (1.0, 0.0, 0.0), 0),
            "from_pos_x": nearest_axis_hit((2.0, 0.0, z), (-1.0, 0.0, 0.0), 0),
            "from_neg_y": nearest_axis_hit((0.0, -2.0, z), (0.0, 1.0, 0.0), 1),
            "from_pos_y": nearest_axis_hit((0.0, 2.0, z), (0.0, -1.0, 0.0), 1),
        }
    )

report = {"blend": bpy.data.filepath, "nearby_objects": nearby, "sections": sections}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
