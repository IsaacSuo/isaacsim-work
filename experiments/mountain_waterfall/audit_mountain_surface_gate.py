"""Audit representative Mountain Splashsurf meshes against their PhysX parent gates.

This gate deliberately separates two claims:

* the PhysX source remains connected from the outlet to the registered rock wall;
* the reconstructed meshes are structurally usable after clipping and pruning.

It does not mistake detached post-impact droplets for holes in the primary stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import meshio
import numpy as np


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mesh_metrics(path: Path) -> dict:
    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=np.float64)
    blocks = [np.asarray(cell.data, dtype=np.int64) for cell in mesh.cells if cell.type == "triangle"]
    if not len(points) or not blocks:
        raise RuntimeError(f"No triangle surface in {path}")
    triangles = np.concatenate(blocks)

    edge_counts: dict[tuple[int, int], int] = {}
    vertex_faces: list[list[int]] = [[] for _ in range(len(points))]
    for face_index, triangle in enumerate(triangles):
        for vertex in triangle:
            vertex_faces[int(vertex)].append(face_index)
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    visited = np.zeros(len(triangles), dtype=bool)
    component_triangles = []
    for seed in range(len(triangles)):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        count = 0
        while stack:
            face_index = stack.pop()
            count += 1
            for vertex in triangles[face_index]:
                for neighbour in vertex_faces[int(vertex)]:
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        stack.append(neighbour)
        component_triangles.append(count)

    component_triangles.sort(reverse=True)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "vertices": int(len(points)),
        "triangles": int(len(triangles)),
        "boundary_edges": int(sum(count == 1 for count in edge_counts.values())),
        "nonmanifold_edges": int(sum(count > 2 for count in edge_counts.values())),
        "connected_components": int(len(component_triangles)),
        "largest_component_triangles": int(component_triangles[0]),
        "small_detached_components_below_100_triangles": int(
            sum(count < 100 for count in component_triangles[1:])
        ),
        "bounds_minimum": points.min(axis=0).astype(float).tolist(),
        "bounds_maximum": points.max(axis=0).astype(float).tolist(),
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("surface_directory", type=Path)
parser.add_argument("physx_report", type=Path)
parser.add_argument("wall_contact_audit", type=Path)
parser.add_argument("stream_connectivity_audit", type=Path)
parser.add_argument("clip_report_directory", type=Path)
parser.add_argument("prune_report_directory", type=Path)
parser.add_argument("output_report", type=Path)
parser.add_argument("--frames", nargs="+", type=int, default=(0, 6, 12))
args = parser.parse_args()

if args.output_report.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output_report}")

physx = load_json(args.physx_report)
wall = load_json(args.wall_contact_audit)
stream = load_json(args.stream_connectivity_audit)

frame_reports = []
for frame in args.frames:
    stem = f"surface_{frame:04d}"
    metrics = mesh_metrics(args.surface_directory / f"{stem}.obj")
    clip = load_json(args.clip_report_directory / f"{stem}.json")
    prune = load_json(args.prune_report_directory / f"{stem}.json")
    metrics.update(
        {
            "frame": frame,
            "clip_nonmanifold_edges": int(clip["nonmanifold_edges"]),
            "removed_component_count": int(prune["removed_component_count"]),
            "removed_component_policy_valid": bool(
                prune.get("removed_component_policy_valid", prune["removed_component_count"] == 0)
            ),
            "component_policy": prune.get(
                "component_policy", "legacy_remove_only_outside_trust_domain"
            ),
        }
    )
    frame_reports.append(metrics)

registration = physx.get("collision_registration", {})
criteria = {
    "physx_parent_is_valid": physx.get("valid") is True,
    "registered_blender_wall_adapter_is_used": registration.get("method")
    in {
        "audited_local_blender_wall_adapter",
        "audited_local_blender_wall_and_ground_adapters",
    },
    "unregistered_usd_mesh_colliders_are_disabled": (
        registration.get("disabled_unregistered_referenced_meshes") == 13
    ),
    "independent_wall_contact_audit_is_valid": wall.get("valid") is True,
    "independent_stream_connectivity_audit_is_valid": stream.get("valid") is True,
    "outlet_to_registered_wall_particle_path_exists": stream.get("criteria", {}).get(
        "source_to_wall_particle_path_exists_at_1_5_spacing"
    )
    is True,
    "seven_neighbour_dense_core_reaches_wall": stream.get("criteria", {}).get(
        "source_to_wall_seven_neighbour_density_core_exists"
    )
    is True,
    "all_surfaces_have_zero_nonmanifold_edges": all(
        frame["nonmanifold_edges"] == 0 and frame["clip_nonmanifold_edges"] == 0
        for frame in frame_reports
    ),
    "all_removed_components_follow_the_declared_policy": all(
        frame["removed_component_policy_valid"] for frame in frame_reports
    ),
}

report = {
    "schema": 1,
    "product": "mountain_wall_pour_representative_surface_gate",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": all(criteria.values()),
    "scope": {
        "proves": [
            "registered rock-wall collision",
            "dense outlet-to-wall primary particle stream",
            "structurally usable representative Splashsurf meshes",
        ],
        "does_not_prove": [
            "continuous wall film from impact to pool",
            "temporal stability of a full-duration surface sequence",
            "final foam, spray, or bubble quality",
        ],
    },
    "parents": {
        "physx_report": str(args.physx_report.resolve()),
        "wall_contact_audit": str(args.wall_contact_audit.resolve()),
        "stream_connectivity_audit": str(args.stream_connectivity_audit.resolve()),
        "wall_adapter_sha256": registration.get("adapter_sha256"),
    },
    "representative_frames": list(args.frames),
    "criteria": criteria,
    "frames": frame_reports,
}

args.output_report.parent.mkdir(parents=True, exist_ok=True)
args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
