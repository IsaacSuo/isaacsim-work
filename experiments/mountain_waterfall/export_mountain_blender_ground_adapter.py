"""Export a scene-derived local ground ring around the Mountain catch basin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_directory", type=Path)
parser.add_argument(
    "--bounds",
    nargs=4,
    type=float,
    default=(2.25, 5.10, -2.75, -0.65),
    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
)
parser.add_argument("--spacing", type=float, default=0.04)
parser.add_argument("--ray-top", type=float, default=5.0)
parser.add_argument("--ray-bottom", type=float, default=-5.0)
parser.add_argument("--basin-centre", nargs=2, type=float, default=(3.49, -1.69))
parser.add_argument("--basin-exclusion-radii", nargs=2, type=float, default=(0.90, 0.68))
parser.add_argument(
    "--maximum-edge-length",
    type=float,
    default=0.18,
    help="Do not bridge separate scan shells or steep first-hit discontinuities.",
)
args = parser.parse_args(argv)

if args.output_directory.exists() and any(args.output_directory.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_directory}")
xmin, xmax, ymin, ymax = args.bounds
if not xmin < xmax or not ymin < ymax or args.spacing <= 0.0:
    raise ValueError("Invalid sampling bounds or spacing")
if not args.ray_bottom < args.ray_top:
    raise ValueError("ray-bottom must be below ray-top")
basin_centre = np.asarray(args.basin_centre, dtype=np.float64)
basin_radii = np.asarray(args.basin_exclusion_radii, dtype=np.float64)
if np.any(basin_radii <= 0.0):
    raise ValueError("Basin exclusion radii must be positive")
if args.maximum_edge_length <= args.spacing:
    raise ValueError("maximum-edge-length must exceed the planar sample spacing")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def outside_basin(x, y):
    offset = (np.asarray((x, y), dtype=np.float64) - basin_centre) / basin_radii
    return float(np.dot(offset, offset)) >= 1.0


x_values = np.arange(xmin, xmax + 0.5 * args.spacing, args.spacing, dtype=np.float64)
y_values = np.arange(ymin, ymax + 0.5 * args.spacing, args.spacing, dtype=np.float64)
locations = np.full((len(y_values), len(x_values), 3), np.nan, dtype=np.float64)
objects = np.full((len(y_values), len(x_values)), "", dtype=object)
faces = np.full((len(y_values), len(x_values)), -1, dtype=np.int64)
normals = np.full_like(locations, np.nan)

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
distance = args.ray_top - args.ray_bottom
for row, y in enumerate(y_values):
    for column, x in enumerate(x_values):
        if not outside_basin(x, y):
            continue
        hit, location, normal, face, obj, _matrix = scene.ray_cast(
            depsgraph,
            Vector((float(x), float(y), args.ray_top)),
            Vector((0.0, 0.0, -1.0)),
            distance=distance,
        )
        if hit:
            locations[row, column] = location[:]
            normals[row, column] = normal[:]
            objects[row, column] = obj.name
            faces[row, column] = int(face)

vertex_lookup = {}
output_vertices_blender = []
output_faces = []
used_hits = set()


def output_vertex(row, column):
    key = (int(row), int(column))
    index = vertex_lookup.get(key)
    if index is None:
        index = len(output_vertices_blender)
        vertex_lookup[key] = index
        output_vertices_blender.append(locations[row, column].copy())
        used_hits.add(key)
    return index


for row in range(len(y_values) - 1):
    for column in range(len(x_values) - 1):
        corners = (
            (row, column),
            (row, column + 1),
            (row + 1, column + 1),
            (row + 1, column),
        )
        for first, second, third in ((0, 1, 2), (0, 2, 3)):
            triangle = (corners[first], corners[second], corners[third])
            if not all(np.isfinite(locations[item]).all() for item in triangle):
                continue
            centroid_xy = np.mean(
                [[x_values[item[1]], y_values[item[0]]] for item in triangle], axis=0
            )
            if not outside_basin(*centroid_xy):
                continue
            triangle_points = np.asarray([locations[item] for item in triangle])
            edge_lengths = (
                np.linalg.norm(triangle_points[1] - triangle_points[0]),
                np.linalg.norm(triangle_points[2] - triangle_points[1]),
                np.linalg.norm(triangle_points[0] - triangle_points[2]),
            )
            if max(edge_lengths) > args.maximum_edge_length:
                continue
            output_faces.append([output_vertex(*item) for item in triangle])

if not output_faces:
    raise RuntimeError("Ground adapter selection is empty")
output_vertices_blender = np.asarray(output_vertices_blender, dtype=np.float64)
output_faces = np.asarray(output_faces, dtype=np.int64)
output_vertices_isaac = output_vertices_blender[:, (0, 2, 1)] * np.asarray((1.0, 1.0, -1.0))
# This mapping is a +90 degree X rotation with determinant +1, so it preserves
# triangle orientation. The upward Blender +Z normal becomes Isaac +Y.

edge_counts = defaultdict(int)
for triangle in output_faces:
    for first, second in (
        (triangle[0], triangle[1]),
        (triangle[1], triangle[2]),
        (triangle[2], triangle[0]),
    ):
        edge_counts[tuple(sorted((int(first), int(second))))] += 1
nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
if nonmanifold_edges:
    raise RuntimeError(f"Ground adapter has {nonmanifold_edges} non-manifold edges")

args.output_directory.mkdir(parents=True, exist_ok=True)
mesh_path = args.output_directory / "mountain_blender_ground_ring.obj"
selection_path = args.output_directory / "selection_record.json"
lines = ["# Scene-derived Mountain ground ring; Isaac Y-up metres."]
for vertex in output_vertices_isaac:
    lines.append("v " + " ".join(format(float(value), ".17g") for value in vertex))
for triangle in output_faces:
    lines.append("f " + " ".join(str(int(value) + 1) for value in triangle))
atomic_text(mesh_path, "\n".join(lines) + "\n")

hit_objects = Counter(objects[row, column] for row, column in used_hits)
hit_faces = defaultdict(set)
for row, column in used_hits:
    hit_faces[str(objects[row, column])].add(int(faces[row, column]))
blend_path = Path(bpy.data.filepath).resolve()
record = {
    "schema": 1,
    "product": "mountain_blender_ground_collision_selection",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source": {
        "blend_path": str(blend_path),
        "blend_sha256": sha256_file(blend_path),
        "coordinate_conversion": "blender_xyz_to_isaac_x_z_minus_y",
    },
    "selector": {
        "method": "regular_downward_scene_ray_samples_outside_analytic_basin_exclusion",
        "bounds_blender_xy": list(map(float, args.bounds)),
        "spacing_m": float(args.spacing),
        "ray_top_m": float(args.ray_top),
        "ray_bottom_m": float(args.ray_bottom),
        "basin_centre_blender_xy": basin_centre.tolist(),
        "basin_exclusion_radii_m": basin_radii.tolist(),
        "maximum_triangle_edge_length_m": float(args.maximum_edge_length),
        "full_scene_first_hit_export": False,
        "purpose": "Catch real splash landing outside the analytic storage basin without replacing its interior.",
    },
    "selection": {
        "sample_grid_shape": [int(len(y_values)), int(len(x_values))],
        "used_hit_samples": int(len(used_hits)),
        "objects": {
            name: {
                "used_hit_samples": int(hit_objects[name]),
                "hit_face_indices": sorted(hit_faces[name]),
            }
            for name in sorted(hit_objects)
        },
    },
    "selected_mesh": {
        "path": str(mesh_path.resolve()),
        "sha256": sha256_file(mesh_path),
        "vertex_count": int(len(output_vertices_isaac)),
        "triangle_count": int(len(output_faces)),
        "boundary_edge_count": int(sum(count == 1 for count in edge_counts.values())),
        "nonmanifold_edge_count": int(nonmanifold_edges),
        "bounds_minimum_isaac": output_vertices_isaac.min(axis=0).tolist(),
        "bounds_maximum_isaac": output_vertices_isaac.max(axis=0).tolist(),
        "watertight_expected": False,
    },
}
atomic_text(selection_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
