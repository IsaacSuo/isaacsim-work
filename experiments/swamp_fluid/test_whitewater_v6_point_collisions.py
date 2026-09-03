"""Regression gates for sparse secondary-domain point collisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from whitewater.point_collisions import SparsePointCollisionScene
from whitewater.scene_contract import SceneContract
from whitewater.terrain_fields import query_open_terrain_points


def row_transform(translation):
    matrix = np.eye(4, dtype=np.float64)
    matrix[3, :3] = translation
    return matrix


contract = SceneContract.from_mapping(
    {
        "schema": 1,
        "product": "whitewater_scene_contract",
        "name": "sparse_point_collision_test",
        "coordinate_system": {
            "axes": "xyz",
            "handedness": "right",
            "metres_per_unit": 1.0,
        },
        "physics": {"gravity": [0.0, -9.81, 0.0]},
        "colliders": [
            {
                "id": "moving_sphere",
                "shape": "sphere",
                "roles": ["solid", "dynamic"],
                "parameters": {"radius": 0.5},
                "motion": {
                    "kind": "source_matrix",
                    "matrix_layout": "row_translation",
                    "source_field": "sphere_transform",
                },
            },
            {
                "id": "static_box",
                "shape": "box",
                "roles": ["solid"],
                "parameters": {"half_extents": [1.0, 0.5, 0.25]},
                "motion": {
                    "kind": "static",
                    "matrix_layout": "row_translation",
                    "transform": row_transform([5.0, 0.0, 0.0]).tolist(),
                },
            },
        ],
        "metadata": {},
    }
)
scene = SparsePointCollisionScene(contract)
snapshot0 = {"sphere_transform": row_transform([0.0, 0.0, 0.0])}
snapshot1 = {"sphere_transform": row_transform([2.0, 0.0, 0.0])}
points = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [6.2, 0.0, 0.0],
    ],
    dtype=np.float64,
)
query = scene.query(points, snapshot0=snapshot0, snapshot1=snapshot1, alpha=0.5)

real_contract_path = (
    Path(__file__).parent / "configs" / "swamp_sphere_impact_open_terrain.scene.json"
)
real_scene = SparsePointCollisionScene(SceneContract.load(real_contract_path))
terrain = real_scene.terrain
triangle = np.asarray(terrain.mesh.triangles[0], dtype=np.float64)
centroid = triangle.mean(axis=0)
normal = np.asarray(terrain.mesh.face_normals[0], dtype=np.float64)
boundary_face, opposite_vertex = np.argwhere(
    terrain.boundary_opposite_vertex
)[0]
boundary_triangle = np.asarray(
    terrain.mesh.triangles[boundary_face], dtype=np.float64
)
boundary_edge_vertices = [
    index for index in range(3) if index != int(opposite_vertex)
]
boundary_midpoint = boundary_triangle[boundary_edge_vertices].mean(axis=0)
terrain_query = query_open_terrain_points(
    terrain,
    np.stack(
        (centroid + 0.01 * normal, centroid - 0.01 * normal, boundary_midpoint)
    ),
)

criteria = {
    "moving_sphere_translation_is_interpolated": bool(
        np.isclose(query["distance"][0], -0.5, atol=1.0e-9)
        and np.isclose(query["distance"][1], 0.0, atol=1.0e-9)
    ),
    "static_box_signed_distance_is_correct": bool(
        np.isclose(query["distance"][2], -0.25, atol=1.0e-9)
        and np.isclose(query["distance"][3], 0.2, atol=1.0e-9)
    ),
    "query_normals_are_unit_length": bool(
        np.allclose(np.linalg.norm(query["normal"], axis=1), 1.0, atol=1.0e-9)
    ),
    "open_terrain_sign_follows_authored_normal": bool(
        terrain_query["valid"][0]
        and terrain_query["valid"][1]
        and terrain_query["distance"][0] > 0.0
        and terrain_query["distance"][1] < 0.0
    ),
    "open_terrain_boundary_is_invalid_not_extruded": bool(
        not terrain_query["valid"][2]
        and terrain_query["touches_boundary"][2]
        and np.isinf(terrain_query["distance"][2])
    ),
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_point_collisions",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": {
        "primitive_distances_m": query["distance"].tolist(),
        "terrain_distances_m": terrain_query["distance"].tolist(),
        "terrain_boundary_invalid": int(
            np.count_nonzero(terrain_query["touches_boundary"])
        ),
    },
}
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
