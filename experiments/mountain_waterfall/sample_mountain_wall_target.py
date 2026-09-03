"""Measure a real front-facing mountain wall patch for a poured-water gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--source-x", type=float, default=3.0)
parser.add_argument("--maximum-x", type=float, default=4.35)
parser.add_argument("--y-range", nargs=2, type=float, default=(-1.98, -1.42))
parser.add_argument("--z-range", nargs=2, type=float, default=(-0.25, 0.88))
parser.add_argument("--y-samples", type=int, default=15)
parser.add_argument("--z-samples", type=int, default=24)
parser.add_argument("--minimum-facing", type=float, default=0.35)
args = parser.parse_args(argv)

if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
if args.maximum_x <= args.source_x:
    raise ValueError("maximum-x must be greater than source-x")
if args.y_samples < 2 or args.z_samples < 2:
    raise ValueError("Both sample counts must be at least two")

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
direction = Vector((1.0, 0.0, 0.0))
distance = args.maximum_x - args.source_x
y_values = np.linspace(*args.y_range, args.y_samples)
z_values = np.linspace(*args.z_range, args.z_samples)
rows = []
accepted = []


def blender_to_isaac(vector, normal=False):
    # Authored mountain adapter is RotateX(-90): (x, y, z) -> (x, z, -y).
    result = [float(vector[0]), float(vector[2]), float(-vector[1])]
    if normal:
        length = float(np.linalg.norm(result))
        if length > 0.0:
            result = [component / length for component in result]
    return result


for z in z_values:
    for y in y_values:
        origin = Vector((args.source_x, float(y), float(z)))
        hit, location, normal, face, obj, _matrix = scene.ray_cast(
            depsgraph, origin, direction, distance=distance
        )
        facing = float(-normal.x) if hit else None
        accept = bool(hit and facing >= args.minimum_facing)
        row = {
            "sample_y": float(y),
            "sample_z": float(z),
            "hit": bool(hit),
            "accepted": accept,
            "location_blender": list(map(float, location)) if hit else None,
            "normal_blender": list(map(float, normal)) if hit else None,
            "location_isaac": blender_to_isaac(location) if hit else None,
            "normal_isaac": blender_to_isaac(normal, normal=True) if hit else None,
            "facing_minus_x": facing,
            "face": int(face) if hit else None,
            "object": obj.name if hit else None,
        }
        rows.append(row)
        if accept:
            accepted.append(row)

if not accepted:
    raise RuntimeError("No front-facing mountain wall samples were accepted")

locations_blender = np.asarray(
    [row["location_blender"] for row in accepted], dtype=np.float64
)
locations_isaac = np.asarray(
    [row["location_isaac"] for row in accepted], dtype=np.float64
)
normals_isaac = np.asarray(
    [row["normal_isaac"] for row in accepted], dtype=np.float64
)
mean_normal = np.mean(normals_isaac, axis=0)
mean_normal /= np.linalg.norm(mean_normal)
object_counts = Counter(row["object"] for row in accepted)
faces_by_object = {
    name: sorted({row["face"] for row in accepted if row["object"] == name})
    for name in sorted(object_counts)
}

# The contact gate uses this measured AABB only to identify particles entering
# the real-wall neighbourhood.  A velocity deflection is also required before
# it may be counted as a collision; merely crossing the box is not sufficient.
contact_margin = np.asarray((0.06, 0.06, 0.06), dtype=np.float64)
target_minimum = locations_isaac.min(axis=0) - contact_margin
target_maximum = locations_isaac.max(axis=0) + contact_margin

payload = {
    "schema": 1,
    "product": "mountain_pour_wall_target",
    "coordinate_system": {
        "blender": "right-handed XYZ, Z-up, metres",
        "isaac": "right-handed XYZ, Y-up, metres",
        "mapping": "(x, y, z) -> (x, z, -y)",
    },
    "sampling": {
        "ray_origin_x": args.source_x,
        "ray_direction_blender": [1.0, 0.0, 0.0],
        "maximum_x": args.maximum_x,
        "y_range": list(map(float, args.y_range)),
        "z_range": list(map(float, args.z_range)),
        "shape": [args.z_samples, args.y_samples],
        "minimum_facing_minus_x": args.minimum_facing,
        "total_rays": len(rows),
        "hit_rays": sum(row["hit"] for row in rows),
        "accepted_front_facing_rays": len(accepted),
    },
    "measured_wall": {
        "location_minimum_blender": locations_blender.min(axis=0).tolist(),
        "location_maximum_blender": locations_blender.max(axis=0).tolist(),
        "location_median_blender": np.median(locations_blender, axis=0).tolist(),
        "location_minimum_isaac": locations_isaac.min(axis=0).tolist(),
        "location_maximum_isaac": locations_isaac.max(axis=0).tolist(),
        "location_median_isaac": np.median(locations_isaac, axis=0).tolist(),
        "mean_normal_isaac": mean_normal.tolist(),
        "contact_gate_aabb_minimum_isaac": target_minimum.tolist(),
        "contact_gate_aabb_maximum_isaac": target_maximum.tolist(),
        "objects": dict(sorted(object_counts.items())),
        "faces_by_object": faces_by_object,
    },
    "rays": rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(
    f"MOUNTAIN_WALL_TARGET={args.output.resolve()} "
    f"accepted={len(accepted)}/{len(rows)} "
    f"objects={dict(object_counts)}"
)
