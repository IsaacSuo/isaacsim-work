"""Build the audited Mountain scene adapter consumed by whitewater v6.

The adapter deliberately separates two geometric meanings:

* a single, continuous, upward-facing open support sheet for terrain distance;
* watertight solid colliders for the measured cliff and analytic basin wall.

No source scene is modified and no existing cache is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREVIEW = ROOT / "output" / "mountain_waterfall_preview"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--ground-directory",
        type=Path,
        default=DEFAULT_PREVIEW / "whitewater_mountain_ground_v4_extended",
    )
    parser.add_argument(
        "--wall-directory",
        type=Path,
        default=DEFAULT_PREVIEW / "whitewater_mountain_terrain_v2",
    )
    parser.add_argument("--pool-centre", nargs=2, type=float, default=(3.49, 1.69))
    parser.add_argument("--pool-floor-y", type=float, default=-0.525)
    parser.add_argument("--pool-transition-inner", type=float, default=0.93)
    parser.add_argument("--wall-extrusion", type=float, default=0.04)
    parser.add_argument("--maximum-ground-fill-distance", type=float, default=0.15)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_obj(path):
    vertices = []
    faces = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"Incomplete OBJ vertex at {path}:{line_number}")
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            indices = [int(token.split("/", 1)[0]) - 1 for token in fields[1:]]
            if len(indices) != 3:
                raise ValueError(f"Non-triangle OBJ face at {path}:{line_number}")
            faces.append(indices)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError(f"OBJ contains no vertices: {path}")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise ValueError(f"OBJ contains no triangle faces: {path}")
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError(f"OBJ contains invalid geometry: {path}")
    return vertices, faces


def write_obj(path, comment, vertices, faces):
    lines = [f"# {comment}"]
    lines.extend(
        "v " + " ".join(format(float(value), ".17g") for value in vertex)
        for vertex in vertices
    )
    lines.extend(
        "f " + " ".join(str(int(value) + 1) for value in face)
        for face in faces
    )
    atomic_text(path, "\n".join(lines) + "\n")


def edge_statistics(faces):
    counts = defaultdict(int)
    directed = {}
    for face in np.asarray(faces, dtype=np.int64):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(first), int(second))))
            counts[key] += 1
            directed.setdefault(key, (int(first), int(second)))
    return counts, directed


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


args = parse_args()
output = args.output_directory.resolve()
if output.exists() and any(output.iterdir()):
    raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
if not 0.0 < args.pool_transition_inner < 1.0:
    raise ValueError("pool-transition-inner must lie between zero and one")
if args.wall_extrusion <= 0.0 or args.maximum_ground_fill_distance <= 0.0:
    raise ValueError("Wall extrusion and fill distance must be positive")

ground_directory = args.ground_directory.resolve()
wall_directory = args.wall_directory.resolve()
ground_path = ground_directory / "mountain_blender_ground_ring.obj"
ground_selection_path = ground_directory / "selection_record.json"
ground_audit_path = ground_directory / "audit_report.json"
wall_path = wall_directory / "mountain_blender_wall.obj"
wall_selection_path = wall_directory / "selection_record.json"
wall_audit_path = wall_directory / "audit_report.json"
for path in (
    ground_path,
    ground_selection_path,
    ground_audit_path,
    wall_path,
    wall_selection_path,
    wall_audit_path,
):
    if not path.is_file():
        raise FileNotFoundError(path)

ground_selection = json.loads(ground_selection_path.read_text(encoding="utf-8"))
ground_audit = json.loads(ground_audit_path.read_text(encoding="utf-8"))
wall_selection = json.loads(wall_selection_path.read_text(encoding="utf-8"))
wall_audit = json.loads(wall_audit_path.read_text(encoding="utf-8"))
if (
    ground_selection.get("product") != "mountain_blender_ground_collision_selection"
    or ground_selection.get("selected_mesh", {}).get("sha256") != sha256_file(ground_path)
    or ground_audit.get("valid") is not True
    or ground_audit.get("mesh_sha256") != sha256_file(ground_path)
):
    raise ValueError("Ground adapter provenance is invalid")
if (
    wall_selection.get("product") != "mountain_blender_wall_collision_selection"
    or wall_selection.get("selected_mesh", {}).get("sha256") != sha256_file(wall_path)
    or wall_audit.get("valid") is not True
    or wall_audit.get("adapter_sha256") != sha256_file(wall_path)
):
    raise ValueError("Wall adapter provenance is invalid")

# Reconstruct the exact regular Blender-derived ground sampling lattice.
selector = ground_selection["selector"]
xmin, xmax, blender_ymin, blender_ymax = map(float, selector["bounds_blender_xy"])
zmin, zmax = -blender_ymax, -blender_ymin
spacing = float(selector["spacing_m"])
nx = int(round((xmax - xmin) / spacing)) + 1
nz = int(round((zmax - zmin) / spacing)) + 1
x_values = xmin + spacing * np.arange(nx, dtype=np.float64)
z_values = zmin + spacing * np.arange(nz, dtype=np.float64)
ground_vertices, _ground_faces = load_obj(ground_path)
ground_y = np.full((nx, nz), np.nan, dtype=np.float64)
for vertex in ground_vertices:
    ix = int(round((vertex[0] - xmin) / spacing))
    iz = int(round((vertex[2] - zmin) / spacing))
    if not 0 <= ix < nx or not 0 <= iz < nz:
        raise ValueError("Ground vertex lies outside its declared sample grid")
    expected_xz = np.asarray((x_values[ix], z_values[iz]))
    if np.max(np.abs(vertex[(0, 2),] - expected_xz)) > 2.0e-5:
        raise ValueError("Ground vertex does not align to the declared sample grid")
    if np.isfinite(ground_y[ix, iz]) and not np.isclose(
        ground_y[ix, iz], vertex[1], rtol=0.0, atol=1.0e-8
    ):
        raise ValueError("Ground grid contains conflicting heights")
    ground_y[ix, iz] = vertex[1]
if not np.isfinite(ground_y).any():
    raise ValueError("Ground adapter contains no regular samples")

missing = ~np.isfinite(ground_y)
fill_distance, nearest = distance_transform_edt(
    missing, sampling=spacing, return_indices=True
)
filled_ground_y = ground_y.copy()
filled_ground_y[missing] = ground_y[nearest[0][missing], nearest[1][missing]]

centre = np.asarray(args.pool_centre, dtype=np.float64)
exclusion_radii = np.asarray(selector["basin_exclusion_radii_m"], dtype=np.float64)
grid_x, grid_z = np.meshgrid(x_values, z_values, indexing="ij")
radius = np.sqrt(
    ((grid_x - centre[0]) / exclusion_radii[0]) ** 2
    + ((grid_z - centre[1]) / exclusion_radii[1]) ** 2
)
outside_basin = radius >= 1.0
maximum_outside_fill_distance = float(fill_distance[outside_basin].max(initial=0.0))
if maximum_outside_fill_distance > args.maximum_ground_fill_distance + 1.0e-12:
    raise ValueError(
        "Ground holes exceed the declared local fill distance: "
        f"{maximum_outside_fill_distance:.6g} m"
    )

# The blend occurs entirely through the analytically authored wall thickness.
# It closes the internal open edge without pretending the basin wall is terrain;
# the actual wall remains a set of solid box colliders below.
blend = smoothstep(
    (radius - args.pool_transition_inner) / (1.0 - args.pool_transition_inner)
)
terrain_y = args.pool_floor_y + blend * (filled_ground_y - args.pool_floor_y)
terrain_y[radius >= 1.0] = filled_ground_y[radius >= 1.0]
terrain_vertices = np.column_stack(
    (grid_x.reshape(-1), terrain_y.reshape(-1), grid_z.reshape(-1))
)
terrain_faces = []
for ix in range(nx - 1):
    for iz in range(nz - 1):
        a = ix * nz + iz
        b = (ix + 1) * nz + iz
        c = (ix + 1) * nz + iz + 1
        d = ix * nz + iz + 1
        # x cross z points toward -Y; reverse each triangle toward the fluid.
        terrain_faces.extend(((a, c, b), (a, d, c)))
terrain_faces = np.asarray(terrain_faces, dtype=np.int64)
terrain_edges, _ = edge_statistics(terrain_faces)

# Convert the measured, fluid-facing open wall into a thin solid by extruding
# in +X, the audited direction into the rock for this shot.
wall_vertices, wall_faces = load_obj(wall_path)
wall_count = len(wall_vertices)
solid_vertices = np.vstack(
    (wall_vertices, wall_vertices + np.asarray((args.wall_extrusion, 0.0, 0.0)))
)
solid_faces = [tuple(map(int, face)) for face in wall_faces]
solid_faces.extend(
    tuple(map(int, face))
    for face in (wall_faces[:, (0, 2, 1)] + wall_count)
)
wall_edges, wall_directed = edge_statistics(wall_faces)
for key, count in wall_edges.items():
    if count != 1:
        continue
    first, second = wall_directed[key]
    solid_faces.extend(
        (
            (second, first, first + wall_count),
            (second, first + wall_count, second + wall_count),
        )
    )
solid_faces = np.asarray(solid_faces, dtype=np.int64)

output.mkdir(parents=True, exist_ok=True)
terrain_path = output / "mountain_whitewater_support_terrain.obj"
terrain_selection_path = output / "terrain_selection.json"
solid_wall_path = output / "mountain_whitewater_wall_solid.obj"
scene_path = output / "mountain_whitewater.scene.json"
build_manifest_path = output / "build_manifest.json"
write_obj(
    terrain_path,
    "Continuous Mountain support terrain; upward/fluid-facing; Isaac Y-up metres.",
    terrain_vertices,
    terrain_faces,
)
write_obj(
    solid_wall_path,
    "Measured Mountain wall extruded +X into rock; watertight solid collider.",
    solid_vertices,
    solid_faces,
)

terrain_selection = {
    "schema": 1,
    "product": "whitewater_open_terrain_selection",
    "created_utc": utc_now(),
    "normal_convention": "toward_fluid",
    "source": {
        "ground_mesh": str(ground_path),
        "ground_mesh_sha256": sha256_file(ground_path),
        "ground_selection": str(ground_selection_path),
        "ground_selection_sha256": sha256_file(ground_selection_path),
        "ground_audit": str(ground_audit_path),
        "ground_audit_sha256": sha256_file(ground_audit_path),
        "analytic_catch_basin": {
            "centre_xz": centre.tolist(),
            "exclusion_radii_xz": exclusion_radii.tolist(),
            "floor_y": float(args.pool_floor_y),
            "transition_inner_normalized_radius": float(args.pool_transition_inner),
            "transition_outer_normalized_radius": 1.0,
            "transition_location": "inside the analytic segmented wall thickness",
        },
    },
    "selector": {
        "method": "regular_ground_samples_plus_analytic_basin_continuous_support",
        "grid_origin_xz": [float(x_values[0]), float(z_values[0])],
        "grid_spacing_m": spacing,
        "grid_shape_xz": [nx, nz],
        "missing_outside_basin_samples_filled": int(
            np.count_nonzero(missing & outside_basin)
        ),
        "outside_basin_samples": int(np.count_nonzero(outside_basin)),
        "maximum_local_fill_distance_m": maximum_outside_fill_distance,
        "maximum_allowed_local_fill_distance_m": float(
            args.maximum_ground_fill_distance
        ),
        "fill_method": "nearest_audited_regular_ground_sample",
        "global_scene_first_hit_repeated": False,
    },
    "selected_mesh": {
        "path": str(terrain_path),
        "sha256": sha256_file(terrain_path),
        "vertex_count": int(len(terrain_vertices)),
        "triangle_count": int(len(terrain_faces)),
        "connected_component_count": 1,
        "boundary_edge_count": int(sum(value == 1 for value in terrain_edges.values())),
        "nonmanifold_edge_count": int(sum(value > 2 for value in terrain_edges.values())),
        "bounds_minimum": terrain_vertices.min(axis=0).astype(float).tolist(),
        "bounds_maximum": terrain_vertices.max(axis=0).astype(float).tolist(),
        "watertight_expected": False,
    },
}
atomic_json(terrain_selection_path, terrain_selection)


def static_motion(translation=(0.0, 0.0, 0.0), row_rotation=None):
    rotation = np.eye(3) if row_rotation is None else np.asarray(row_rotation)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[3, :3] = np.asarray(translation, dtype=np.float64)
    return {
        "kind": "static",
        "matrix_layout": "row_translation",
        "transform": matrix.astype(float).tolist(),
    }


colliders = [
    {
        "id": "measured_cliff_wall_solid",
        "shape": "mesh",
        "roles": ["solid", "churn_source"],
        "parameters": {
            "path": str(solid_wall_path),
            "sha256": sha256_file(solid_wall_path),
            "require_watertight": True,
        },
        "motion": static_motion(),
    }
]
basin_centre = centre
basin_radii = np.asarray((0.87, 0.65), dtype=np.float64)
for index in range(32):
    first_angle = 2.0 * math.pi * index / 32
    second_angle = 2.0 * math.pi * (index + 1) / 32
    first = basin_centre + basin_radii * np.asarray(
        (math.cos(first_angle), math.sin(first_angle))
    )
    second = basin_centre + basin_radii * np.asarray(
        (math.cos(second_angle), math.sin(second_angle))
    )
    midpoint = 0.5 * (first + second)
    chord = second - first
    chord_length = float(np.linalg.norm(chord))
    local_x = np.asarray((chord[0], 0.0, chord[1])) / chord_length
    local_y = np.asarray((0.0, 1.0, 0.0))
    local_z = np.cross(local_x, local_y)
    colliders.append(
        {
            "id": f"catch_basin_wall_{index:02d}",
            "shape": "box",
            "roles": ["solid", "churn_source"],
            "parameters": {
                "half_extents": [0.5 * chord_length + 0.015, 0.28, 0.035]
            },
            "motion": static_motion(
                (float(midpoint[0]), -0.34, float(midpoint[1])),
                np.vstack((local_x, local_y, local_z)),
            ),
        }
    )

scene = {
    "schema": 2,
    "product": "whitewater_scene_contract",
    "name": "mountain_wall_pour_with_measured_cliff_and_catch_basin",
    "coordinate_system": {
        "axes": "xyz",
        "handedness": "right",
        "metres_per_unit": 1.0,
    },
    "physics": {"gravity": [0.0, -9.81, 0.0]},
    "terrain": {
        "id": "mountain_continuous_support",
        "representation": "open_triangle_mesh",
        "parameters": {
            "path": str(terrain_path),
            "sha256": sha256_file(terrain_path),
            "selection_path": str(terrain_selection_path),
            "selection_sha256": sha256_file(terrain_selection_path),
            "normal_convention": "toward_fluid",
            "require_open": True,
        },
        "motion": static_motion(),
    },
    "colliders": colliders,
    "metadata": {
        "scene_family": "continuous_pour_cliff_impact_and_plunge_pool",
        "adapter": "audited_blender_ground_plus_analytic_basin_and_measured_wall",
        "manual_scene_configuration": True,
        "per_frame_repairs": False,
        "source_scene_modified": False,
        "terrain_role": "gravity support and analytic pool floor",
        "solid_role": "measured cliff plus exact 32-segment PhysX catch wall",
    },
}
atomic_json(scene_path, scene)

manifest = {
    "schema": 1,
    "product": "mountain_whitewater_scene_adapter_build",
    "created_utc": utc_now(),
    "complete": True,
    "scene_contract": str(scene_path),
    "scene_contract_sha256": sha256_file(scene_path),
    "terrain": str(terrain_path),
    "terrain_sha256": sha256_file(terrain_path),
    "terrain_selection": str(terrain_selection_path),
    "terrain_selection_sha256": sha256_file(terrain_selection_path),
    "wall_solid": str(solid_wall_path),
    "wall_solid_sha256": sha256_file(solid_wall_path),
    "wall_extrusion_m": float(args.wall_extrusion),
    "source_wall": str(wall_path),
    "source_wall_sha256": sha256_file(wall_path),
    "source_wall_selection_sha256": sha256_file(wall_selection_path),
    "source_wall_audit_sha256": sha256_file(wall_audit_path),
}
atomic_json(build_manifest_path, manifest)
print(json.dumps(manifest, indent=2, sort_keys=True))
