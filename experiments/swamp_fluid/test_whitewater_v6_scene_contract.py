"""Synthetic gates for the versioned scene contract and collider fields."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from whitewater.collision_fields import build_collider_fields, collider_sdf
from whitewater.liquid_fields import GridSpec, sphere_collision_sdf
from whitewater.scene_contract import SceneContract, legacy_sphere_impact_contract


def row_transform(translation=(0.0, 0.0, 0.0), rotation=None):
    matrix = np.eye(4, dtype=np.float64)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[3, :3] = translation
    return matrix


def contract_payload(colliders, metres_per_unit=1.0):
    return {
        "schema": 1,
        "product": "whitewater_scene_contract",
        "name": "synthetic_scene",
        "coordinate_system": {
            "axes": "xyz",
            "handedness": "right",
            "metres_per_unit": metres_per_unit,
        },
        "physics": {"gravity": [0.0, -9.81, 0.0]},
        "colliders": colliders,
        "metadata": {"purpose": "unit_test"},
    }


def static_collider(identifier, shape, parameters, transform=None, roles=None):
    return {
        "id": identifier,
        "shape": shape,
        "roles": roles or ["solid"],
        "parameters": parameters,
        "motion": {
            "kind": "static",
            "matrix_layout": "row_translation",
            "transform": row_transform() if transform is None else transform,
        },
    }


spec = GridSpec((-0.5, -0.5, -0.5), 0.1, (11, 11, 11))
center = tuple(size // 2 for size in spec.shape)

sphere_contract = SceneContract.from_mapping(
    contract_payload(
        [
            static_collider(
                "sphere", "sphere", {"radius": 0.2}, row_transform((0.1, 0.0, 0.0))
            )
        ]
    )
)
sphere = collider_sdf(spec, sphere_contract.colliders[0])
legacy_sphere = sphere_collision_sdf(spec, (0.1, 0.0, 0.0), 0.2)

angle = np.pi / 2.0
rotation_z = np.asarray(
    (
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )
)
box_contract = SceneContract.from_mapping(
    contract_payload(
        [
            static_collider(
                "box",
                "box",
                {"half_extents": [0.1, 0.2, 0.3]},
                row_transform(rotation=rotation_z),
            )
        ]
    )
)
box = collider_sdf(spec, box_contract.colliders[0])

capsule_contract = SceneContract.from_mapping(
    contract_payload(
        [
            static_collider(
                "capsule",
                "capsule",
                {"radius": 0.1, "half_length": 0.2, "axis": "y"},
            )
        ]
    )
)
capsule = collider_sdf(spec, capsule_contract.colliders[0])

plane_contract = SceneContract.from_mapping(
    contract_payload(
        [static_collider("ground", "plane", {"offset": -0.1})]
    )
)
plane = collider_sdf(spec, plane_contract.colliders[0])

dynamic_payload = contract_payload(
    [
        static_collider("wall", "plane", {"offset": -0.4}),
        {
            "id": "impactor",
            "shape": "sphere",
            "roles": ["solid", "dynamic", "churn_source"],
            "parameters": {"radius": 0.15},
            "motion": {
                "kind": "source_matrix",
                "source_field": "impactor_transform",
                "matrix_layout": "row_translation",
            },
        },
    ]
)
dynamic_contract = SceneContract.from_mapping(dynamic_payload)
dynamic_fields = build_collider_fields(
    spec,
    dynamic_contract,
    snapshot={"impactor_transform": row_transform((0.0, 0.1, 0.0))},
)
static_only_fields = build_collider_fields(
    spec, dynamic_contract, include_motion={"static"}
)
dynamic_only_fields = build_collider_fields(
    spec,
    dynamic_contract,
    snapshot={"impactor_transform": row_transform((0.0, 0.1, 0.0))},
    include_motion={"source_matrix"},
)

# A centimetre-authored scene must normalize both transform translation and
# primitive dimensions into the metre-based solver grid.
scaled_contract = SceneContract.from_mapping(
    contract_payload(
        [
            static_collider(
                "centimetre_sphere",
                "sphere",
                {"radius": 20.0},
                row_transform((10.0, 0.0, 0.0)),
            )
        ],
        metres_per_unit=0.01,
    )
)
scaled = build_collider_fields(spec, scaled_contract)["collider_collision_sdf"]

mesh_path = Path(__file__).with_name("test_assets") / "unit_cube_outward.obj"
mesh_sha256 = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
mesh_contract = SceneContract.from_mapping(
    contract_payload(
        [
            static_collider(
                "mesh_cube",
                "mesh",
                {
                    "path": str(mesh_path),
                    "sha256": mesh_sha256,
                    "require_watertight": True,
                },
                row_transform((0.25, 0.0, 0.0)),
            )
        ],
        metres_per_unit=0.4,
    )
)
mesh_sdf = build_collider_fields(spec, mesh_contract)["collider_collision_sdf"]
mesh_reference_contract = SceneContract.from_mapping(
    contract_payload(
        [
            static_collider(
                "box_reference",
                "box",
                {"half_extents": [0.5, 0.5, 0.5]},
                row_transform((0.25, 0.0, 0.0)),
            )
        ],
        metres_per_unit=0.4,
    )
)
mesh_reference = build_collider_fields(spec, mesh_reference_contract)[
    "collider_collision_sdf"
]

legacy_contract = legacy_sphere_impact_contract(
    {"impactor": {"radius": 0.2}, "gravity": [0.0, -9.81, 0.0]}
)
legacy_fields = build_collider_fields(
    spec,
    legacy_contract,
    snapshot={"sphere_transform": row_transform((0.1, 0.0, 0.0))},
)

invalid_cases = []
for label, mutate in (
    ("unknown_top_level_key", lambda value: value.update({"unexpected": True})),
    (
        "duplicate_collider_id",
        lambda value: value["colliders"].append(dict(value["colliders"][0])),
    ),
    (
        "dynamic_role_missing",
        lambda value: value["colliders"][0].update({"roles": ["solid"]}),
    ),
    (
        "non_rigid_transform",
        lambda value: value["colliders"][0]["motion"].update(
            {"kind": "static", "transform": np.diag((2.0, 1.0, 1.0, 1.0)).tolist()}
        ),
    ),
):
    value = contract_payload(
        [
            {
                "id": "moving",
                "shape": "sphere",
                "roles": ["solid", "dynamic"],
                "parameters": {"radius": 0.1},
                "motion": {
                    "kind": "source_matrix",
                    "source_field": "transform",
                    "matrix_layout": "row_translation",
                },
            }
        ]
    )
    mutate(value)
    try:
        SceneContract.from_mapping(value)
    except ValueError:
        invalid_cases.append(label)

wrong_hash_payload = contract_payload(
    [
        static_collider(
            "wrong_hash_mesh",
            "mesh",
            {
                "path": str(mesh_path),
                "sha256": "0" * 64,
                "require_watertight": True,
            },
        )
    ]
)
try:
    SceneContract.from_mapping(wrong_hash_payload)
except ValueError:
    invalid_cases.append("mesh_hash_mismatch")

non_required_payload = contract_payload(
    [
        static_collider(
            "non_required_mesh",
            "mesh",
            {
                "path": str(mesh_path),
                "sha256": mesh_sha256,
                "require_watertight": False,
            },
        )
    ]
)
try:
    SceneContract.from_mapping(non_required_payload)
except ValueError:
    invalid_cases.append("mesh_watertight_requirement_disabled")

inward_path = Path(__file__).with_name("test_assets") / "unit_cube_inward.obj"
inward_contract = SceneContract.from_mapping(
    contract_payload(
        [
            static_collider(
                "inward_mesh",
                "mesh",
                {
                    "path": str(inward_path),
                    "sha256": hashlib.sha256(inward_path.read_bytes()).hexdigest(),
                    "require_watertight": True,
                },
            )
        ]
    )
)
try:
    build_collider_fields(spec, inward_contract)
except ValueError:
    invalid_cases.append("mesh_inward_winding")

metrics = {
    "sphere_legacy_error_max": float(np.max(np.abs(sphere - legacy_sphere))),
    "sphere_translated_center_sdf": float(sphere[6, 5, 5]),
    "box_center_sdf": float(box[center]),
    "box_rotated_long_axis_inside_sdf": float(box[3, 5, 5]),
    "box_rotated_short_axis_outside_sdf": float(box[5, 7, 5]),
    "capsule_center_sdf": float(capsule[center]),
    "capsule_end_cap_surface_sdf": float(capsule[5, 8, 5]),
    "plane_below_sdf": float(plane[5, 3, 5]),
    "plane_above_sdf": float(plane[5, 5, 5]),
    "scaled_sphere_error_max": float(np.max(np.abs(scaled - legacy_sphere))),
    "legacy_adapter_error_max": float(
        np.max(np.abs(legacy_fields["dynamic_collision_sdf"] - legacy_sphere))
    ),
    "mesh_box_error_max": float(np.max(np.abs(mesh_sdf - mesh_reference))),
    "invalid_cases_rejected": sorted(invalid_cases),
}
criteria = {
    "sphere_matches_legacy_analytic_sdf": metrics["sphere_legacy_error_max"] <= 1.0e-7,
    "sphere_translation_is_respected": abs(metrics["sphere_translated_center_sdf"] + 0.2) <= 1.0e-7,
    "oriented_box_has_correct_signs": (
        metrics["box_center_sdf"] < 0.0
        and metrics["box_rotated_long_axis_inside_sdf"] < 0.0
        and metrics["box_rotated_short_axis_outside_sdf"] > 0.0
    ),
    "capsule_has_correct_distance": (
        metrics["capsule_center_sdf"] < 0.0
        and abs(metrics["capsule_end_cap_surface_sdf"]) <= 1.0e-7
    ),
    "plane_has_correct_half_space_sign": (
        metrics["plane_below_sdf"] < 0.0 and metrics["plane_above_sdf"] > 0.0
    ),
    "roles_are_not_conflated": (
        set(dynamic_fields) >= {
            "collider_collision_sdf",
            "dynamic_collision_sdf",
            "churn_source_sdf",
        }
        and dynamic_fields["metadata"]["field_roles"]["churn_source"] == ["impactor"]
        and dynamic_fields["metadata"]["field_roles"]["solid"] == ["wall", "impactor"]
    ),
    "static_and_dynamic_fields_partition_exactly": (
        static_only_fields["metadata"]["field_roles"]["solid"] == ["wall"]
        and "dynamic_collision_sdf" not in static_only_fields
        and "churn_source_sdf" not in static_only_fields
        and dynamic_only_fields["metadata"]["field_roles"]["solid"]
        == ["impactor"]
        and dynamic_only_fields["metadata"]["field_roles"]["churn_source"]
        == ["impactor"]
    ),
    "scene_units_normalize_to_metres": metrics["scaled_sphere_error_max"] <= 1.0e-7,
    "legacy_sphere_adapter_is_exact": metrics["legacy_adapter_error_max"] <= 1.0e-7,
    "watertight_mesh_matches_analytic_box": metrics["mesh_box_error_max"] <= 1.0e-6,
    "strict_schema_rejects_invalid_contracts": set(invalid_cases)
    == {
        "unknown_top_level_key",
        "duplicate_collider_id",
        "dynamic_role_missing",
        "non_rigid_transform",
        "mesh_hash_mismatch",
        "mesh_watertight_requirement_disabled",
        "mesh_inward_winding",
    },
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_scene_contract",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": metrics,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
