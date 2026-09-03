"""Synthetic gates for open terrain selection, provenance and one-sided SDFs."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from whitewater.liquid_fields import GridSpec
from whitewater.scene_contract import SceneContract
from whitewater.terrain_fields import (
    open_terrain_collision_sdf,
    rasterize_open_terrain_heightfield,
    select_open_terrain_faces,
)
from whitewater.render_surface_support import load_render_surface_support_recipe


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_identity():
    return np.eye(4, dtype=np.float64).tolist()


temporary_directory = tempfile.TemporaryDirectory(prefix="whitewater_open_terrain_")
temporary = Path(temporary_directory.name)

# Two disconnected sheets overlap in X/Z.  Only the lower sheet is below the
# liquid support level, so component expansion must reject the unrelated shell.
vertices = np.asarray(
    [
        (-0.5, 0.0, -0.5),
        (-0.5, 0.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.0, -0.5),
        (-0.5, 1.0, -0.5),
        (-0.5, 1.0, 0.5),
        (0.5, 1.0, 0.5),
        (0.5, 1.0, -0.5),
    ],
    dtype=np.float64,
)
faces = np.asarray(((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)), dtype=np.int64)
selected_vertices, selected_faces, selection = select_open_terrain_faces(
    vertices,
    faces,
    (-0.6, -0.1, -0.6),
    (0.6, 1.1, 0.6),
    maximum_surface_y=0.25,
    expand_supported_components=True,
)

mesh_path = temporary / "selected_plane.obj"
lines = []
for vertex in selected_vertices:
    lines.append("v " + " ".join(format(float(value), ".17g") for value in vertex))
for face in selected_faces:
    lines.append("f " + " ".join(str(int(value) + 1) for value in face))
mesh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
mesh_hash = sha256_file(mesh_path)
selection_path = temporary / "selection.json"
selection_record = {
    "schema": 1,
    "product": "whitewater_open_terrain_selection",
    "normal_convention": "toward_fluid",
    "selected_mesh": {
        "path": str(mesh_path),
        "sha256": mesh_hash,
        "vertex_count": int(len(selected_vertices)),
        "triangle_count": int(len(selected_faces)),
        "connected_component_count": selection["connected_component_count"],
    },
    "selector": selection["selector"],
    "selection": {
        "source_face_indices": selection["source_face_indices"],
        "support_seed_face_indices": selection["support_seed_face_indices"],
    },
}
selection_path.write_text(
    json.dumps(selection_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
selection_hash = sha256_file(selection_path)

payload = {
    "schema": 2,
    "product": "whitewater_scene_contract",
    "name": "synthetic_open_terrain",
    "coordinate_system": {
        "axes": "xyz",
        "handedness": "right",
        "metres_per_unit": 1.0,
    },
    "physics": {"gravity": [0.0, -9.81, 0.0]},
    "terrain": {
        "id": "ground",
        "representation": "open_triangle_mesh",
        "parameters": {
            "path": str(mesh_path),
            "sha256": mesh_hash,
            "selection_path": str(selection_path),
            "selection_sha256": selection_hash,
            "normal_convention": "toward_fluid",
            "require_open": True,
        },
        "motion": {
            "kind": "static",
            "matrix_layout": "row_translation",
            "transform": row_identity(),
        },
    },
    "colliders": [],
    "metadata": {"purpose": "unit_test"},
}
contract = SceneContract.from_mapping(payload)
spec = GridSpec((-0.25, -0.2, -0.25), 0.1, (6, 5, 6))
sdf, valid, metadata = open_terrain_collision_sdf(spec, contract.terrain)
raster_axis = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
raster_y, raster_metadata = rasterize_open_terrain_heightfield(
    contract.terrain, raster_axis, raster_axis
)

scene_path = temporary / "scene.json"
scene_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
raster_path = temporary / "terrain_raster.npz"
np.savez_compressed(
    raster_path,
    schema=np.int32(1),
    x_values=raster_axis,
    z_values=raster_axis,
    terrain_y=raster_y,
    spacing=np.float64(0.1),
    scene_contract_sha256=np.asarray(sha256_file(scene_path)),
    terrain_mesh_sha256=np.asarray(mesh_hash),
    terrain_selection_sha256=np.asarray(selection_hash),
)
recipe_path = temporary / "splashsurf_manifest.json"
recipe_payload = {
    "schema": 2,
    "configuration": {
        "shoreline_mode": "terrain",
        "shoreline_particles_ply": None,
        "terrain": {
            "representation": "open_triangle_mesh_raster_adapter",
            "scene_contract": str(scene_path),
            "scene_contract_sha256": sha256_file(scene_path),
            "terrain_mesh": str(mesh_path),
            "terrain_mesh_sha256": mesh_hash,
            "terrain_selection": str(selection_path),
            "terrain_selection_sha256": selection_hash,
            "normal_convention": "toward_fluid",
            "raster": {
                "path": str(raster_path),
                "sha256": sha256_file(raster_path),
                "derived": True,
                "metadata": raster_metadata,
            },
        },
        "impact": [0.0, 0.0, 0.25],
        "spacing": 0.1,
        "water_level": 0.3,
        "minimum_layers": 2,
        "shoreline_erosion_cells": 0,
    },
    "state": {"complete": True},
}
recipe_path.write_text(json.dumps(recipe_payload, indent=2) + "\n", encoding="utf-8")
recipe = load_render_surface_support_recipe(recipe_path)

outside_spec = GridSpec((-0.6, -0.1, -0.1), 0.1, (3, 3, 3))
outside_sdf, outside_valid, _ = open_terrain_collision_sdf(
    outside_spec, contract.terrain
)

invalid_cases = []
bad_schema_one = dict(payload)
bad_schema_one["schema"] = 1
try:
    SceneContract.from_mapping(bad_schema_one)
except ValueError:
    invalid_cases.append("schema1_embedded_terrain")

bad_selection_hash = json.loads(json.dumps(payload))
bad_selection_hash["terrain"]["parameters"]["selection_sha256"] = "0" * 64
try:
    SceneContract.from_mapping(bad_selection_hash)
except ValueError:
    invalid_cases.append("selection_hash_mismatch")

metrics = {
    "support_seed_face_count": len(selection["support_seed_face_indices"]),
    "selected_face_count": len(selection["source_face_indices"]),
    "selected_source_faces": selection["source_face_indices"],
    "valid_fraction": float(valid.mean()),
    "sdf_below": float(sdf[2, 0, 2]),
    "sdf_surface": float(sdf[2, 2, 2]),
    "sdf_above": float(sdf[2, 4, 2]),
    "outside_invalid_nodes": int(np.count_nonzero(~outside_valid.astype(bool))),
    "outside_infinite_nodes": int(np.count_nonzero(~np.isfinite(outside_sdf))),
    "boundary_edges": metadata["boundary_edge_count"],
    "raster_valid_cells": int(np.count_nonzero(np.isfinite(raster_y))),
    "raster_height_absolute_maximum": float(np.nanmax(np.abs(raster_y))),
    "raster_global_first_hit": raster_metadata["global_first_hit_ray_cast"],
    "recipe_terrain_representation": recipe.terrain_representation,
    "invalid_cases_rejected": sorted(invalid_cases),
}
criteria = {
    "stacked_unrelated_shell_is_rejected": metrics["selected_source_faces"] == [0, 1],
    "liquid_support_seed_is_explicit": metrics["support_seed_face_count"] == 2,
    "selected_sheet_covers_interior_grid": metrics["valid_fraction"] == 1.0,
    "one_sided_sign_points_toward_fluid": (
        metrics["sdf_below"] < 0.0
        and abs(metrics["sdf_surface"]) <= 1.0e-7
        and metrics["sdf_above"] > 0.0
    ),
    "open_boundary_is_invalid_not_extruded": (
        metrics["outside_invalid_nodes"] > 0
        and metrics["outside_infinite_nodes"] == metrics["outside_invalid_nodes"]
    ),
    "selected_surface_raster_is_exact": (
        metrics["raster_valid_cells"] == raster_y.size
        and metrics["raster_height_absolute_maximum"] <= 1.0e-12
        and metrics["raster_global_first_hit"] is False
    ),
    "render_support_recipe_owns_open_terrain": (
        metrics["recipe_terrain_representation"]
        == "open_triangle_mesh_raster_adapter"
    ),
    "strict_terrain_provenance_is_enforced": set(invalid_cases)
    == {"schema1_embedded_terrain", "selection_hash_mismatch"},
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_open_terrain",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": metrics,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
