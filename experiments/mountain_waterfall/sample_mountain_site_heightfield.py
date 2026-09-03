"""Ray-sample a local Blender XY heightfield for waterfall site selection."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--bounds", nargs=4, type=float, required=True, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
parser.add_argument("--spacing", type=float, default=0.1)
parser.add_argument("--ray-top", type=float, default=5.0)
parser.add_argument("--ray-bottom", type=float, default=-5.0)
args = parser.parse_args(argv)

xmin, xmax, ymin, ymax = args.bounds
if not xmin < xmax or not ymin < ymax or args.spacing <= 0.0:
    raise ValueError("Bounds and spacing are invalid")
x_values = np.arange(xmin, xmax + 0.5 * args.spacing, args.spacing, dtype=np.float64)
y_values = np.arange(ymin, ymax + 0.5 * args.spacing, args.spacing, dtype=np.float64)
heights = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
objects = np.full(heights.shape, "", dtype=object)
scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
distance = args.ray_top - args.ray_bottom
for row, y in enumerate(y_values):
    for column, x in enumerate(x_values):
        hit, location, _normal, _face, obj, _matrix = scene.ray_cast(
            depsgraph,
            Vector((float(x), float(y), args.ray_top)),
            Vector((0.0, 0.0, -1.0)),
            distance=distance,
        )
        if hit:
            heights[row, column] = location.z
            objects[row, column] = obj.name

dx = np.abs(np.diff(heights, axis=1))
dy = np.abs(np.diff(heights, axis=0))
candidates = []
for flat_index in np.argsort(dx.ravel())[::-1][:20]:
    row, column = np.unravel_index(flat_index, dx.shape)
    if np.isfinite(dx[row, column]):
        candidates.append(
            {
                "axis": "x",
                "drop_m": float(dx[row, column]),
                "first": [float(x_values[column]), float(y_values[row]), float(heights[row, column])],
                "second": [float(x_values[column + 1]), float(y_values[row]), float(heights[row, column + 1])],
            }
        )
for flat_index in np.argsort(dy.ravel())[::-1][:20]:
    row, column = np.unravel_index(flat_index, dy.shape)
    if np.isfinite(dy[row, column]):
        candidates.append(
            {
                "axis": "y",
                "drop_m": float(dy[row, column]),
                "first": [float(x_values[column]), float(y_values[row]), float(heights[row, column])],
                "second": [float(x_values[column]), float(y_values[row + 1]), float(heights[row + 1, column])],
            }
        )
candidates.sort(key=lambda item: item["drop_m"], reverse=True)
payload = {
    "schema": 1,
    "product": "mountain_waterfall_site_heightfield",
    "coordinate_system": "Blender world XYZ, Z up, metres",
    "bounds": [xmin, xmax, ymin, ymax],
    "spacing": args.spacing,
    "shape": list(heights.shape),
    "valid_samples": int(np.count_nonzero(np.isfinite(heights))),
    "height_minimum": float(np.nanmin(heights)),
    "height_maximum": float(np.nanmax(heights)),
    "largest_adjacent_drops": candidates[:30],
    "x_values": x_values.tolist(),
    "y_values": y_values.tolist(),
    "heights": heights.tolist(),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(
    f"MOUNTAIN_HEIGHTFIELD={args.output.resolve()} shape={heights.shape} "
    f"height={np.nanmin(heights):.3f}..{np.nanmax(heights):.3f}"
)
