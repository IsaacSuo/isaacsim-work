"""Print per-mesh world bounds and topology counts for the current blend."""

import json

import bpy
from mathutils import Vector


rows = []
for obj in bpy.context.scene.objects:
    if obj.type != "MESH":
        continue
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    rows.append(
        {
            "name": obj.name,
            "vertices": len(obj.data.vertices),
            "faces": len(obj.data.polygons),
            "bounds": [
                [min(float(point[axis]) for point in corners) for axis in range(3)],
                [max(float(point[axis]) for point in corners) for axis in range(3)],
            ],
        }
    )

print("BLENDER_MESH_AUDIT=" + json.dumps(rows, ensure_ascii=False))
