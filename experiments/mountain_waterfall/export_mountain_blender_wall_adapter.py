"""Export an audited local collision wall from the rendered Mountain scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np


argv = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("wall_target", type=Path)
parser.add_argument("output_directory", type=Path)
parser.add_argument("--roi-margin", nargs=3, type=float, default=(0.25, 0.25, 0.25))
args = parser.parse_args(argv)

if not args.wall_target.is_file():
    raise FileNotFoundError(args.wall_target)
if args.output_directory.exists() and any(args.output_directory.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_directory}")
margin = np.asarray(args.roi_margin, dtype=np.float64)
if margin.shape != (3,) or np.any(margin <= 0.0):
    raise ValueError("roi-margin must contain three positive values")


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


def blender_to_isaac(points):
    points = np.asarray(points, dtype=np.float64)
    return points[:, (0, 2, 1)] * np.asarray((1.0, 1.0, -1.0))


target = json.loads(args.wall_target.read_text(encoding="utf-8"))
if target.get("product") != "mountain_pour_wall_target":
    raise ValueError("Unexpected wall target product")
measured = target["measured_wall"]
roi_minimum = np.asarray(measured["location_minimum_blender"], dtype=np.float64) - margin
roi_maximum = np.asarray(measured["location_maximum_blender"], dtype=np.float64) + margin
seed_faces_by_object = {
    name: set(map(int, faces))
    for name, faces in measured["faces_by_object"].items()
}

depsgraph = bpy.context.evaluated_depsgraph_get()
selected_records = []
world_vertices_by_object = {}
for object_name, seed_polygons in seed_faces_by_object.items():
    source_object = bpy.data.objects.get(object_name)
    if source_object is None or source_object.type != "MESH":
        raise RuntimeError(f"Wall target object is unavailable: {object_name}")
    evaluated = source_object.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        local = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
        homogeneous = np.column_stack((local, np.ones(len(local))))
        world = (homogeneous @ matrix.T)[:, :3]
        world_vertices_by_object[object_name] = world
        candidate_indices = []
        triangle_vertices = []
        polygon_indices = []
        for triangle_index, triangle in enumerate(mesh.loop_triangles):
            vertices = np.asarray(triangle.vertices, dtype=np.int64)
            points = world[vertices]
            overlaps = bool(
                np.all(points.max(axis=0) >= roi_minimum)
                and np.all(points.min(axis=0) <= roi_maximum)
            )
            if overlaps:
                candidate_indices.append(triangle_index)
                triangle_vertices.append(vertices)
                polygon_indices.append(int(triangle.polygon_index))
        triangle_vertices = np.asarray(triangle_vertices, dtype=np.int64)
        polygon_indices = np.asarray(polygon_indices, dtype=np.int64)
        parent = np.arange(len(triangle_vertices), dtype=np.int64)

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        def union(first, second):
            first = find(first)
            second = find(second)
            if first != second:
                parent[second] = first

        owners = {}
        for local_index, face in enumerate(triangle_vertices):
            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = tuple(sorted(map(int, edge)))
                previous = owners.get(key)
                if previous is None:
                    owners[key] = local_index
                else:
                    union(previous, local_index)
        labels = np.asarray([find(index) for index in range(len(parent))], dtype=np.int64)
        seeded_labels = {
            int(label)
            for label, polygon in zip(labels, polygon_indices)
            if int(polygon) in seed_polygons
        }
        missing_seeds = sorted(seed_polygons - set(map(int, polygon_indices)))
        if missing_seeds:
            raise RuntimeError(
                f"Target faces fell outside the expanded ROI for {object_name}: {missing_seeds[:8]}"
            )
        if not seeded_labels:
            raise RuntimeError(f"No seeded wall component resolved for {object_name}")
        keep = np.asarray([int(label) in seeded_labels for label in labels], dtype=bool)
        for face, polygon in zip(triangle_vertices[keep], polygon_indices[keep]):
            selected_records.append((object_name, face.copy(), int(polygon)))
    finally:
        evaluated.to_mesh_clear()

if not selected_records:
    raise RuntimeError("Wall adapter selection is empty")

vertex_lookup = {}
output_vertices_blender = []
output_faces = []
selected_polygons = defaultdict(set)
for object_name, face, polygon in selected_records:
    output_face = []
    for source_vertex in face:
        key = (object_name, int(source_vertex))
        output_index = vertex_lookup.get(key)
        if output_index is None:
            output_index = len(output_vertices_blender)
            vertex_lookup[key] = output_index
            output_vertices_blender.append(
                world_vertices_by_object[object_name][int(source_vertex)]
            )
        output_face.append(output_index)
    output_faces.append(output_face)
    selected_polygons[object_name].add(polygon)

output_vertices_isaac = blender_to_isaac(output_vertices_blender)
output_faces = np.asarray(output_faces, dtype=np.int64)
triangles = output_vertices_isaac[output_faces]
area_vector = np.sum(
    np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
    axis=0,
)
if float(np.dot(area_vector, (-1.0, 0.0, 0.0))) < 0.0:
    output_faces[:, 1:3] = output_faces[:, 2:0:-1]

edge_counts = defaultdict(int)
for face in output_faces:
    for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
        edge_counts[tuple(sorted(map(int, edge)))] += 1
nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
if nonmanifold_edges:
    raise RuntimeError(f"Selected wall has {nonmanifold_edges} non-manifold edges")

args.output_directory.mkdir(parents=True, exist_ok=True)
mesh_path = args.output_directory / "mountain_blender_wall.obj"
selection_path = args.output_directory / "selection_record.json"
lines = ["# Audited Mountain wall collision adapter; Isaac Y-up metres."]
for vertex in output_vertices_isaac:
    lines.append("v " + " ".join(format(float(value), ".17g") for value in vertex))
for face in output_faces:
    lines.append("f " + " ".join(str(int(value) + 1) for value in face))
atomic_text(mesh_path, "\n".join(lines) + "\n")

blend_path = Path(bpy.data.filepath).resolve()
record = {
    "schema": 1,
    "product": "mountain_blender_wall_collision_selection",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source": {
        "blend_path": str(blend_path),
        "blend_sha256": sha256_file(blend_path),
        "wall_target_path": str(args.wall_target.resolve()),
        "wall_target_sha256": sha256_file(args.wall_target),
        "coordinate_conversion": "blender_xyz_to_isaac_x_z_minus_y",
    },
    "selector": {
        "method": "ray_hit_polygon_seeded_connected_components_inside_expanded_roi",
        "roi_minimum_blender": roi_minimum.tolist(),
        "roi_maximum_blender": roi_maximum.tolist(),
        "roi_margin": margin.tolist(),
        "global_first_hit_full_scene_export": False,
    },
    "selection": {
        "objects": {
            name: {
                "seed_polygon_indices": sorted(seed_faces_by_object[name]),
                "selected_polygon_indices": sorted(selected_polygons[name]),
            }
            for name in sorted(selected_polygons)
        }
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
        "normal_convention": "toward_source_minus_x",
        "watertight_expected": False,
    },
}
atomic_text(selection_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
