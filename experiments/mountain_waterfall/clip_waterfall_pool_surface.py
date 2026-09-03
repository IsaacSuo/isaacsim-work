"""Clip only the submerged pool shell while preserving the vertical waterfall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input_obj", type=Path)
parser.add_argument("output_obj", type=Path)
parser.add_argument("--minimum-y", type=float, default=-0.37)
parser.add_argument("--report", type=Path)
args = parser.parse_args()

if not args.input_obj.is_file():
    raise FileNotFoundError(args.input_obj)
for target in (args.output_obj, args.report):
    if target is not None and target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")

mesh = meshio.read(args.input_obj)
points = np.asarray(mesh.points, dtype=np.float64)
triangles = np.concatenate(
    [np.asarray(block.data, dtype=np.int64) for block in mesh.cells if block.type == "triangle"]
)
if not len(points) or not len(triangles):
    raise RuntimeError("Input OBJ contains no triangle surface")


def intersect(first, second):
    denominator = second[1] - first[1]
    if abs(denominator) < 1.0e-12:
        return first.copy()
    amount = (args.minimum_y - first[1]) / denominator
    result = first + amount * (second - first)
    result[1] = args.minimum_y
    return result


def clip_polygon(triangle_indices):
    output = []
    previous_index = int(triangle_indices[-1])
    previous = points[previous_index]
    previous_inside = previous[1] >= args.minimum_y
    for raw_current_index in triangle_indices:
        current_index = int(raw_current_index)
        current = points[current_index]
        current_inside = current[1] >= args.minimum_y
        if current_inside != previous_inside:
            edge_key = ("edge", *sorted((previous_index, current_index)))
            output.append((edge_key, intersect(previous, current)))
        if current_inside:
            output.append((("vertex", current_index), current.copy()))
        previous_index = current_index
        previous = current
        previous_inside = current_inside
    return output


output_points = []
output_triangles = []
vertex_lookup = {}


def vertex_index(key, value):
    index = vertex_lookup.get(key)
    if index is None:
        index = len(output_points)
        vertex_lookup[key] = index
        output_points.append(np.asarray(value, dtype=np.float64))
    return index


discarded = 0
split = 0
for triangle in triangles:
    polygon = clip_polygon(triangle)
    if len(polygon) < 3:
        discarded += 1
        continue
    if len(polygon) != 3:
        split += 1
    root = vertex_index(*polygon[0])
    for corner in range(1, len(polygon) - 1):
        face = (
            root,
            vertex_index(*polygon[corner]),
            vertex_index(*polygon[corner + 1]),
        )
        if len(set(face)) == 3:
            output_triangles.append(face)

output_points = np.asarray(output_points, dtype=np.float32)
output_triangles = np.asarray(output_triangles, dtype=np.int64)
if not len(output_points) or not len(output_triangles):
    raise RuntimeError("Clipping removed the entire surface")

args.output_obj.parent.mkdir(parents=True, exist_ok=True)
meshio.write(
    args.output_obj,
    meshio.Mesh(points=output_points, cells=[("triangle", output_triangles)]),
    file_format="obj",
)

edge_counts = {}
for triangle in output_triangles:
    for first, second in (
        (triangle[0], triangle[1]),
        (triangle[1], triangle[2]),
        (triangle[2], triangle[0]),
    ):
        key = tuple(sorted((int(first), int(second))))
        edge_counts[key] = edge_counts.get(key, 0) + 1

report = {
    "schema": 1,
    "product": "mountain_waterfall_pool_shell_clip",
    "mode": "global_minimum_y_preserves_vertical_waterfall_above_pool",
    "input": str(args.input_obj.resolve()),
    "output": str(args.output_obj.resolve()),
    "minimum_y": args.minimum_y,
    "input_vertices": int(len(points)),
    "input_triangles": int(len(triangles)),
    "output_vertices": int(len(output_points)),
    "output_triangles": int(len(output_triangles)),
    "discarded_triangles": discarded,
    "split_triangles": split,
    "boundary_edges": int(sum(count == 1 for count in edge_counts.values())),
    "nonmanifold_edges": int(sum(count > 2 for count in edge_counts.values())),
    "bounds_minimum": output_points.min(axis=0).astype(float).tolist(),
    "bounds_maximum": output_points.max(axis=0).astype(float).tolist(),
}
if args.report is not None:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
