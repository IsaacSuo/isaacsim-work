"""Independently audit a Mountain whitewater v6 scene adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "swamp_fluid"))
from whitewater.scene_contract import SceneContract  # noqa: E402
from whitewater.terrain_fields import load_open_terrain_mesh, query_open_terrain_points  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def connected_face_components(faces):
    faces = np.asarray(faces, dtype=np.int64)
    parent = np.arange(len(faces), dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left = find(left)
        right = find(right)
        if left != right:
            parent[right] = left

    owners = {}
    for face_index, face in enumerate(faces):
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted(map(int, edge)))
            if key in owners:
                union(face_index, owners[key])
            else:
                owners[key] = face_index
    return len({find(index) for index in range(len(faces))})


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("adapter_directory", type=Path)
args = parser.parse_args()
directory = args.adapter_directory.resolve()
manifest_path = directory / "build_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
scene_path = Path(manifest["scene_contract"]).resolve()
terrain_path = Path(manifest["terrain"]).resolve()
selection_path = Path(manifest["terrain_selection"]).resolve()
wall_path = Path(manifest["wall_solid"]).resolve()
source_wall_path = Path(manifest["source_wall"]).resolve()
selection = json.loads(selection_path.read_text(encoding="utf-8"))

contract = SceneContract.load(scene_path)
loaded_terrain = load_open_terrain_mesh(contract.terrain)
terrain_mesh = loaded_terrain.mesh
wall_mesh = trimesh.load(wall_path, force="mesh", process=False, validate=False)
source_wall = trimesh.load(source_wall_path, force="mesh", process=False, validate=False)
terrain_normals = np.asarray(terrain_mesh.face_normals)
terrain_component_count = connected_face_components(terrain_mesh.faces)

# Probe the declared primary-liquid carrier domain. The wider collision adapter
# is a physical landing buffer, not permission to absorb late escaped spray
# into the primary body. Internal invalid edges are forbidden in this domain;
# the sole allowed terrain boundary remains outside it.
trusted_domain_xz = (2.00, 5.00, 0.50, 3.50)
x_values = np.linspace(trusted_domain_xz[0], trusted_domain_xz[1], 42)
z_values = np.linspace(trusted_domain_xz[2], trusted_domain_xz[3], 37)
probe_x, probe_z = np.meshgrid(x_values, z_values, indexing="ij")
probe_points = np.column_stack(
    (probe_x.reshape(-1), np.full(probe_x.size, 2.25), probe_z.reshape(-1))
)
terrain_query = query_open_terrain_points(loaded_terrain, probe_points)

source_vertex_count = len(source_wall.vertices)
extrusion = float(manifest["wall_extrusion_m"])
front_matches = np.allclose(
    np.asarray(wall_mesh.vertices[:source_vertex_count]),
    np.asarray(source_wall.vertices),
    rtol=0.0,
    atol=1.0e-12,
)
back_matches = np.allclose(
    np.asarray(wall_mesh.vertices[source_vertex_count:]),
    np.asarray(source_wall.vertices) + np.asarray((extrusion, 0.0, 0.0)),
    rtol=0.0,
    atol=1.0e-12,
)

selected = selection["selected_mesh"]
selector = selection["selector"]
criteria = {
    "build_manifest_complete": manifest.get("complete") is True,
    "build_hashes_match": (
        manifest.get("scene_contract_sha256") == sha256_file(scene_path)
        and manifest.get("terrain_sha256") == sha256_file(terrain_path)
        and manifest.get("terrain_selection_sha256") == sha256_file(selection_path)
        and manifest.get("wall_solid_sha256") == sha256_file(wall_path)
        and manifest.get("source_wall_sha256") == sha256_file(source_wall_path)
    ),
    "scene_contract_loads_as_schema_2": contract.schema == 2 and contract.terrain is not None,
    "terrain_selection_owns_mesh": (
        selection.get("product") == "whitewater_open_terrain_selection"
        and selected.get("sha256") == sha256_file(terrain_path)
        and int(selected.get("vertex_count", -1)) == len(terrain_mesh.vertices)
        and int(selected.get("triangle_count", -1)) == len(terrain_mesh.faces)
    ),
    "terrain_is_one_open_consistent_component": (
        not terrain_mesh.is_watertight
        and terrain_mesh.is_winding_consistent
        and terrain_component_count == 1
        and int(selected.get("nonmanifold_edge_count", -1)) == 0
    ),
    "terrain_normals_face_fluid": bool(np.all(terrain_normals[:, 1] > 0.0)),
    "terrain_has_no_internal_invalid_boundary_in_trusted_domain": bool(
        np.all(terrain_query["valid"])
    ),
    "ground_hole_fill_is_local_and_bounded": (
        int(selector["missing_outside_basin_samples_filled"])
        / int(selector["outside_basin_samples"])
        <= 0.02
        and float(selector["maximum_local_fill_distance_m"])
        <= float(selector["maximum_allowed_local_fill_distance_m"])
    ),
    "wall_front_and_extrusion_match_audited_source": front_matches and back_matches,
    "wall_solid_is_watertight_consistent_positive": (
        isinstance(wall_mesh, trimesh.Trimesh)
        and wall_mesh.is_watertight
        and wall_mesh.is_winding_consistent
        and float(wall_mesh.volume) > 0.0
    ),
    "scene_contains_exact_catch_wall_segments": (
        len([item for item in contract.colliders if item.identifier.startswith("catch_basin_wall_")])
        == 32
    ),
}
report = {
    "schema": 1,
    "product": "mountain_whitewater_scene_adapter_audit",
    "audited_utc": datetime.now(timezone.utc).isoformat(),
    "valid": all(criteria.values()),
    "criteria": criteria,
    "manifest": str(manifest_path),
    "manifest_sha256": sha256_file(manifest_path),
    "scene_contract": str(scene_path),
    "scene_contract_sha256": sha256_file(scene_path),
    "metrics": {
        "terrain_vertices": int(len(terrain_mesh.vertices)),
        "terrain_triangles": int(len(terrain_mesh.faces)),
        "terrain_boundary_edges": int(selected["boundary_edge_count"]),
        "terrain_connected_components": terrain_component_count,
        "terrain_minimum_normal_y": float(terrain_normals[:, 1].min()),
        "terrain_probe_points": int(len(probe_points)),
        "terrain_trusted_domain_xz": list(trusted_domain_xz),
        "terrain_valid_probe_fraction": float(np.mean(terrain_query["valid"])),
        "ground_holes_filled": int(selector["missing_outside_basin_samples_filled"]),
        "ground_hole_fill_fraction": float(
            int(selector["missing_outside_basin_samples_filled"])
            / int(selector["outside_basin_samples"])
        ),
        "maximum_ground_fill_distance_m": float(selector["maximum_local_fill_distance_m"]),
        "wall_vertices": int(len(wall_mesh.vertices)),
        "wall_triangles": int(len(wall_mesh.faces)),
        "wall_volume_m3": float(wall_mesh.volume),
        "wall_watertight": bool(wall_mesh.is_watertight),
        "wall_winding_consistent": bool(wall_mesh.is_winding_consistent),
        "catch_wall_segments": 32,
    },
}
atomic_json(directory / "audit_report.json", report)
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
