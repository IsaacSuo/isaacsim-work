"""Scan upper-hemisphere visibility from the apartment soft-body landing point."""

import json
import math

import bpy
from mathutils import Vector


scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
origin = Vector((-4.1, -6.4, -0.35))
rows = []
for elevation_degrees in range(40, 71):
    elevation = math.radians(elevation_degrees)
    for azimuth_degrees in range(90, 131):
        azimuth = math.radians(azimuth_degrees)
        direction = Vector(
            (
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            )
        )
        hit, location, _normal, _index, obj, _matrix = scene.ray_cast(
            depsgraph, origin, direction, distance=100.0
        )
        rows.append(
            {
                "azimuth": azimuth_degrees,
                "elevation": elevation_degrees,
                "clear": not hit,
                "distance": None if not hit else (location - origin).length,
                "object": None if not hit else obj.name,
            }
        )

clear_rows = [row for row in rows if row["clear"]]
print("WINDOW_CLEAR_RAYS=" + json.dumps(clear_rows))
