"""Independently audit a sampled Mountain ground-ring collision adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import meshio
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mesh", type=Path)
parser.add_argument("selection", type=Path)
parser.add_argument("report", type=Path)
args = parser.parse_args()
if args.report.exists():
    raise FileExistsError(f"Refusing to overwrite {args.report}")
selection = json.loads(args.selection.read_text(encoding="utf-8"))
if selection.get("product") != "mountain_blender_ground_collision_selection":
    raise ValueError("Unexpected ground selection product")

source = selection["source"]
source_blend = Path(source["blend_path"])
selected = selection["selected_mesh"]
selector = selection["selector"]
mesh = meshio.read(args.mesh)
points = np.asarray(mesh.points, dtype=np.float64)
triangles = np.concatenate(
    [np.asarray(cell.data, dtype=np.int64) for cell in mesh.cells if cell.type == "triangle"]
)
triangle_points = points[triangles]
edge_lengths = np.stack(
    (
        np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 0], axis=1),
        np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 1], axis=1),
        np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 2], axis=1),
    ),
    axis=1,
)
area_vectors = np.cross(
    triangle_points[:, 1] - triangle_points[:, 0],
    triangle_points[:, 2] - triangle_points[:, 0],
)
normal_lengths = np.linalg.norm(area_vectors, axis=1)
upward_fraction = float(np.mean(area_vectors[:, 1] > 0.0))

edge_counts = {}
for triangle in triangles:
    for first, second in (
        (triangle[0], triangle[1]),
        (triangle[1], triangle[2]),
        (triangle[2], triangle[0]),
    ):
        edge = tuple(sorted((int(first), int(second))))
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

centre_blender = np.asarray(selector["basin_centre_blender_xy"], dtype=np.float64)
centre_isaac_xz = np.asarray((centre_blender[0], -centre_blender[1]))
radii = np.asarray(selector["basin_exclusion_radii_m"], dtype=np.float64)
centroids = triangle_points.mean(axis=1)
normalized = (centroids[:, (0, 2)] - centre_isaac_xz) / radii
minimum_exclusion_norm = float(np.sqrt(np.sum(normalized * normalized, axis=1)).min())
maximum_allowed_edge = float(selector["maximum_triangle_edge_length_m"])

criteria = {
    "source_blend_exists_and_hash_matches": source_blend.is_file()
    and sha256(source_blend) == source["blend_sha256"],
    "mesh_hash_matches_selection": sha256(args.mesh) == selected["sha256"],
    "mesh_counts_match_selection": len(points) == selected["vertex_count"]
    and len(triangles) == selected["triangle_count"],
    "all_triangles_are_non_degenerate": bool(np.all(normal_lengths > 1.0e-10)),
    "all_triangle_normals_point_upward": upward_fraction == 1.0,
    "maximum_edge_length_gate_is_respected": float(edge_lengths.max())
    <= maximum_allowed_edge + 1.0e-6,
    "analytic_basin_interior_is_excluded": minimum_exclusion_norm >= 1.0,
    "mesh_has_no_nonmanifold_edges": all(count <= 2 for count in edge_counts.values()),
}
report = {
    "schema": 1,
    "product": "mountain_blender_ground_collision_audit",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": {
        "vertices": int(len(points)),
        "triangles": int(len(triangles)),
        "boundary_edges": int(sum(count == 1 for count in edge_counts.values())),
        "nonmanifold_edges": int(sum(count > 2 for count in edge_counts.values())),
        "maximum_triangle_edge_length_m": float(edge_lengths.max()),
        "upward_normal_fraction": upward_fraction,
        "minimum_basin_exclusion_normalized_radius": minimum_exclusion_norm,
        "bounds_minimum_isaac": points.min(axis=0).tolist(),
        "bounds_maximum_isaac": points.max(axis=0).tolist(),
    },
    "mesh_sha256": sha256(args.mesh),
    "source_blend_sha256": sha256(source_blend) if source_blend.is_file() else None,
}
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
