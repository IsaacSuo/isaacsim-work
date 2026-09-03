"""Verify that the exported Blender wall exactly covers the measured target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import trimesh


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("adapter_directory", type=Path)
parser.add_argument("wall_target", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


mesh_path = args.adapter_directory / "mountain_blender_wall.obj"
selection_path = args.adapter_directory / "selection_record.json"
for path in (mesh_path, selection_path, args.wall_target):
    if not path.is_file():
        raise FileNotFoundError(path)
selection = json.loads(selection_path.read_text(encoding="utf-8"))
target = json.loads(args.wall_target.read_text(encoding="utf-8"))
mesh = trimesh.load(mesh_path, force="mesh", process=False, validate=False)
rows = [row for row in target["rays"] if row.get("accepted") is True]
points = np.asarray([row["location_isaac"] for row in rows], dtype=np.float64)
normals = np.asarray([row["normal_isaac"] for row in rows], dtype=np.float64)
closest, distances, triangle_ids = trimesh.proximity.closest_point(mesh, points)
face_normals = np.asarray(mesh.face_normals[triangle_ids], dtype=np.float64)
normal_alignment = np.sum(face_normals * normals, axis=1)
edge_counts = {}
for face in np.asarray(mesh.faces, dtype=np.int64):
    for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
        key = tuple(sorted(map(int, edge)))
        edge_counts[key] = edge_counts.get(key, 0) + 1
criteria = {
    "selection_mesh_hash_matches": (
        sha256_file(mesh_path) == selection["selected_mesh"]["sha256"]
    ),
    "wall_target_hash_matches_selection": (
        sha256_file(args.wall_target) == selection["source"]["wall_target_sha256"]
    ),
    "mesh_is_triangulated_and_winding_consistent": (
        isinstance(mesh, trimesh.Trimesh)
        and mesh.faces.shape[1] == 3
        and mesh.is_winding_consistent
    ),
    "mesh_has_no_nonmanifold_edges": not any(count > 2 for count in edge_counts.values()),
    "all_measured_wall_samples_are_covered_within_1mm": float(np.max(distances)) <= 0.001,
    "at_least_95_percent_of_sample_normals_align": (
        float(np.mean(normal_alignment >= 0.5)) >= 0.95
    ),
}
criteria = {name: bool(value) for name, value in criteria.items()}
report = {
    "schema": 1,
    "product": "mountain_blender_wall_collision_adapter_audit",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "adapter": str(mesh_path.resolve()),
    "adapter_sha256": sha256_file(mesh_path),
    "selection": str(selection_path.resolve()),
    "selection_sha256": sha256_file(selection_path),
    "measured_samples": len(points),
    "maximum_sample_distance_m": float(np.max(distances)),
    "mean_sample_distance_m": float(np.mean(distances)),
    "minimum_normal_alignment": float(np.min(normal_alignment)),
    "mean_normal_alignment": float(np.mean(normal_alignment)),
    "normal_alignment_fraction_at_least_0_5": float(np.mean(normal_alignment >= 0.5)),
    "vertices": int(len(mesh.vertices)),
    "triangles": int(len(mesh.faces)),
    "boundary_edges": int(sum(count == 1 for count in edge_counts.values())),
    "nonmanifold_edges": int(sum(count > 2 for count in edge_counts.values())),
    "bounds_minimum": np.asarray(mesh.bounds[0], dtype=float).tolist(),
    "bounds_maximum": np.asarray(mesh.bounds[1], dtype=float).tolist(),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
temporary = args.output.with_name(args.output.name + ".tmp")
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, args.output)
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
