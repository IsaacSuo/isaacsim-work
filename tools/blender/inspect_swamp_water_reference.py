"""Print water/terrain reference transforms from a Swamp Blender file."""

import json
import sys

import bpy
from mathutils import Vector


def bounds_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return {
        "minimum": [min(point[axis] for point in corners) for axis in range(3)],
        "maximum": [max(point[axis] for point in corners) for axis in range(3)],
    }


rows = []
for obj in bpy.data.objects:
    materials = [slot.material.name for slot in obj.material_slots if slot.material]
    searchable = " ".join((obj.name, *materials)).lower()
    if obj.type == "MESH":
        rows.append(
            {
                "name": obj.name,
                "type": obj.type,
                "location": list(obj.matrix_world.translation),
                "dimensions": list(obj.dimensions),
                "bounds_world": bounds_world(obj),
                "materials": materials,
            }
        )
rows.sort(key=lambda row: row["dimensions"][0] * row["dimensions"][1], reverse=True)
print(json.dumps(rows[:40], indent=2))
