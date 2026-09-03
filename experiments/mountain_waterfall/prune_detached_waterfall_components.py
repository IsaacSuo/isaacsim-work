"""Remove only tiny detached surface components outside an authored trust domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input_obj", type=Path)
parser.add_argument("output_obj", type=Path)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--minimum-component-triangles", type=int, default=100)
parser.add_argument("--domain-min", nargs=3, type=float, default=(2.4, -0.5, 0.9))
parser.add_argument("--domain-max", nargs=3, type=float, default=(4.5, 1.2, 2.5))
parser.add_argument(
    "--remove-small-components-everywhere",
    action="store_true",
    help="Route every sub-threshold detached component to the secondary-droplet layer.",
)
args = parser.parse_args()

if not args.input_obj.is_file():
    raise FileNotFoundError(args.input_obj)
for target in (args.output_obj, args.report):
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
if args.minimum_component_triangles < 1:
    raise ValueError("Minimum component triangle count must be positive")

mesh = meshio.read(args.input_obj)
points = np.asarray(mesh.points, dtype=np.float32)
triangles = np.concatenate(
    [np.asarray(block.data, dtype=np.int64) for block in mesh.cells if block.type == "triangle"]
)
vertex_faces = [[] for _ in range(len(points))]
for face_index, face in enumerate(triangles):
    for vertex in face:
        vertex_faces[int(vertex)].append(face_index)

visited = np.zeros(len(triangles), dtype=bool)
components = []
for seed in range(len(triangles)):
    if visited[seed]:
        continue
    stack = [seed]
    visited[seed] = True
    faces = []
    vertices = set()
    while stack:
        face_index = stack.pop()
        faces.append(face_index)
        for vertex in triangles[face_index]:
            vertex = int(vertex)
            vertices.add(vertex)
            for neighbour in vertex_faces[vertex]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
    vertex_array = np.fromiter(vertices, dtype=np.int64)
    bounds_minimum = points[vertex_array].min(axis=0)
    bounds_maximum = points[vertex_array].max(axis=0)
    components.append((np.asarray(faces, dtype=np.int64), vertex_array, bounds_minimum, bounds_maximum))

domain_minimum = np.asarray(args.domain_min, dtype=np.float32)
domain_maximum = np.asarray(args.domain_max, dtype=np.float32)
kept_faces = []
removed = []
for faces, vertices, bounds_minimum, bounds_maximum in components:
    outside_domain = bool(
        np.any(bounds_minimum < domain_minimum) or np.any(bounds_maximum > domain_maximum)
    )
    is_small = len(faces) < args.minimum_component_triangles
    should_remove = is_small and (
        outside_domain or args.remove_small_components_everywhere
    )
    if should_remove:
        removed.append(
            {
                "triangles": int(len(faces)),
                "vertices": int(len(vertices)),
                "outside_trust_domain": outside_domain,
                "reason": (
                    "small_component_outside_trust_domain"
                    if outside_domain
                    else "small_component_routed_to_secondary_droplet_layer"
                ),
                "bounds_minimum": bounds_minimum.astype(float).tolist(),
                "bounds_maximum": bounds_maximum.astype(float).tolist(),
            }
        )
    else:
        kept_faces.append(faces)

kept_face_indices = np.concatenate(kept_faces)
kept_triangles = triangles[kept_face_indices]
used_vertices = np.unique(kept_triangles)
remap = np.full(len(points), -1, dtype=np.int64)
remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
output_points = points[used_vertices]
output_triangles = remap[kept_triangles]

args.output_obj.parent.mkdir(parents=True, exist_ok=True)
meshio.write(
    args.output_obj,
    meshio.Mesh(points=output_points, cells=[("triangle", output_triangles)]),
    file_format="obj",
)
report = {
    "schema": 1,
    "product": "mountain_waterfall_detached_component_prune",
    "input": str(args.input_obj.resolve()),
    "output": str(args.output_obj.resolve()),
    "domain_minimum": domain_minimum.astype(float).tolist(),
    "domain_maximum": domain_maximum.astype(float).tolist(),
    "minimum_component_triangles": args.minimum_component_triangles,
    "remove_small_components_everywhere": args.remove_small_components_everywhere,
    "component_policy": (
        "remove_all_sub_threshold_components_for_secondary_droplet_rendering"
        if args.remove_small_components_everywhere
        else "remove_only_sub_threshold_components_outside_trust_domain"
    ),
    "input_components": len(components),
    "removed_components": removed,
    "removed_component_count": len(removed),
    "removed_component_policy_valid": all(
        item["triangles"] < args.minimum_component_triangles
        and (item["outside_trust_domain"] or args.remove_small_components_everywhere)
        for item in removed
    ),
    "input_triangles": int(len(triangles)),
    "output_triangles": int(len(output_triangles)),
}
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
