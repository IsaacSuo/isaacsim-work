"""Sample the authored mountain camera frustum against evaluated scene geometry."""

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--samples", type=int, default=9)
args = parser.parse_args(argv)

scene = bpy.context.scene
camera = scene.camera
if camera is None:
    raise RuntimeError("Scene has no active camera")
depsgraph = bpy.context.evaluated_depsgraph_get()
corners = camera.data.view_frame(scene=scene)
# Blender returns bottom-left, bottom-right, top-right, top-left in camera space.
bottom_left, bottom_right, top_right, top_left = corners
rows = []
for row in range(args.samples):
    v = row / (args.samples - 1)
    left = bottom_left.lerp(top_left, v)
    right = bottom_right.lerp(top_right, v)
    for column in range(args.samples):
        u = column / (args.samples - 1)
        local = left.lerp(right, u)
        direction = (camera.matrix_world.to_quaternion() @ local).normalized()
        hit, location, normal, face, obj, _matrix = scene.ray_cast(
            depsgraph, camera.location, direction, distance=1000.0
        )
        rows.append(
            {
                "row_from_bottom": row,
                "column_from_left": column,
                "u": u,
                "v": v,
                "hit": bool(hit),
                "location": list(location) if hit else None,
                "normal": list(normal) if hit else None,
                "face": int(face) if hit else None,
                "object": obj.name if hit else None,
            }
        )

payload = {
    "schema": 1,
    "product": "mountain_waterfall_camera_site_inspection",
    "coordinate_system": "Blender world XYZ, Z up, metres",
    "camera": {
        "name": camera.name,
        "location": list(camera.location),
        "rotation_euler": list(camera.rotation_euler),
        "lens_mm": camera.data.lens,
        "sensor_width_mm": camera.data.sensor_width,
    },
    "samples_per_axis": args.samples,
    "rays": rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"MOUNTAIN_SITE_INSPECTION={args.output.resolve()} rays={len(rows)}")
