"""Cross-scene acceptance for impact, river, waterfall, wake and hero foam."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.collision_fields import build_collider_fields
from whitewater.emission_fields import EmissionModel, raw_emission_fields
from whitewater.flow_profiles import FlowProfile
from whitewater.liquid_fields import GridSpec
from whitewater.scene_contract import SceneContract


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_directory", type=Path)
    parser.add_argument("plateau_audit", type=Path)
    parser.add_argument("render_audit", type=Path)
    parser.add_argument("output_report", type=Path)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_transform(translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4, dtype=np.float64)
    matrix[3, :3] = translation
    return matrix


def collider(identifier, shape, parameters, roles, dynamic=False):
    return {
        "id": identifier,
        "shape": shape,
        "roles": roles,
        "parameters": parameters,
        "motion": (
            {
                "kind": "source_matrix",
                "matrix_layout": "row_translation",
                "source_field": identifier + "_transform",
            }
            if dynamic
            else {
                "kind": "static",
                "matrix_layout": "row_translation",
                "transform": row_transform().tolist(),
            }
        ),
    }


def contract(name, colliders):
    return SceneContract.from_mapping(
        {
            "schema": 1,
            "product": "whitewater_scene_contract",
            "name": name,
            "coordinate_system": {
                "axes": "xyz",
                "handedness": "right",
                "metres_per_unit": 1.0,
            },
            "physics": {"gravity": [0.0, -9.81, 0.0]},
            "colliders": colliders,
            "metadata": {"acceptance_fixture": True},
        }
    )


def scenario_fields(model, include_source):
    shape = (13, 13, 13)
    spacing = model.spacing
    axis = (np.arange(shape[0]) - shape[0] // 2) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    phi = y.astype(np.float32)
    normal = np.zeros(shape + (3,), dtype=np.float32)
    normal[..., 1] = 1.0
    surface = np.abs(phi) <= 2.5 * model.surface_half_width_cells * spacing
    outward = x < 0.0
    velocity = np.zeros(shape + (3,), dtype=np.float32)
    acceleration = np.zeros_like(velocity)
    velocity[..., 0] = 0.35 * model.characteristic_velocity
    velocity[..., 1] = np.where(outward, 1.1, -1.1) * model.characteristic_velocity
    acceleration[..., 1] = np.where(outward, 5.5, -9.0) * model.gravity
    curvature = np.zeros(shape, dtype=np.float32)
    curvature[surface] = 0.90 / model.characteristic_length
    divergence = np.full(
        shape, -0.42 / model.characteristic_time, dtype=np.float32
    )
    vorticity = np.zeros(shape + (3,), dtype=np.float32)
    vorticity[..., 2] = 0.48 / model.characteristic_time
    fields = {
        "phi": phi,
        "normal": normal,
        "velocity": velocity,
        "acceleration": acceleration,
        "curvature": curvature,
        "divergence": divergence,
        "vorticity": vorticity,
        "strain_rate": np.full(
            shape, 0.35 / model.characteristic_time, dtype=np.float32
        ),
        "collision_sdf": np.full(shape, 4.0 * spacing, dtype=np.float32),
        "surface_valid": np.ones(shape, dtype=np.uint8),
        "velocity_valid": np.ones(shape, dtype=np.uint8),
        "acceleration_valid": np.ones(shape, dtype=np.uint8),
    }
    if include_source:
        fields["churn_source_sdf"] = np.sqrt(x * x + y * y + z * z) - spacing
    return fields


args = parse_args()
profile_directory = args.profile_directory.resolve()
profile_paths = {
    path.stem: path for path in sorted(profile_directory.glob("*.json"))
}
profiles = {name: FlowProfile.load(path) for name, path in profile_paths.items()}
errors = []
if set(profiles) != {"impact", "river", "waterfall", "wake"}:
    errors.append("the four required flow profiles are not present")

spec = GridSpec((-0.12, -0.12, -0.12), 0.02, (13, 13, 13))
contracts = {
    "impact": contract(
        "impact_fixture",
        [collider("impact", "sphere", {"radius": 0.05}, ["solid", "dynamic", "churn_source"], True)],
    ),
    "river": contract(
        "river_fixture",
        [collider("bed", "plane", {"offset": -0.08}, ["solid"])],
    ),
    "waterfall": contract(
        "waterfall_fixture",
        [collider("plunge_bed", "box", {"half_extents": [0.12, 0.02, 0.12]}, ["solid"])],
    ),
    "wake": contract(
        "wake_fixture",
        [collider("hull", "capsule", {"radius": 0.025, "half_length": 0.08, "axis": "x"}, ["solid", "dynamic", "churn_source"], True)],
    ),
}
snapshots = {
    "impact": {"impact_transform": row_transform((0.0, 0.02, 0.0))},
    "river": {},
    "waterfall": {},
    "wake": {"hull_transform": row_transform((0.0, 0.0, 0.0))},
}

scene_rows = {}
for name in ("impact", "river", "waterfall", "wake"):
    profile = profiles[name]
    contract_fields = build_collider_fields(
        spec, contracts[name], snapshot=snapshots[name]
    )
    collision = contract_fields["collider_collision_sdf"]
    model = EmissionModel(
        spec.spacing,
        conditioning_sigma=profile.conditioning_sigma_cells,
        surface_half_width_cells=profile.surface_half_width_cells,
        churn_depth_cells=profile.churn_depth_cells,
        sphere_influence_cells=profile.churn_source_influence_cells,
        profile=profile,
    )
    include_source = name in {"impact", "wake"}
    fields = scenario_fields(model, include_source=include_source)
    # The flow fixture has independent free-liquid clearance; collider fields
    # are audited separately so the analytic solid cannot mask the channel test.
    output = raw_emission_fields(fields, model)
    maxima = {
        channel: float(output[channel + "_raw"].max(initial=0.0))
        for channel in ("spray", "entrained_air", "churn")
    }
    without_source = scenario_fields(model, include_source=False)
    no_source_churn = float(
        raw_emission_fields(without_source, model)["churn_raw"].max(initial=0.0)
    )
    expected = {
        "impact": maxima["spray"] > 0 and maxima["entrained_air"] > 0 and maxima["churn"] > 0,
        "river": maxima["spray"] > 0 and maxima["entrained_air"] > 0 and maxima["churn"] == 0,
        "waterfall": maxima["spray"] > 0 and maxima["entrained_air"] > 0 and maxima["churn"] > 0,
        "wake": maxima["spray"] > 0 and maxima["entrained_air"] > 0 and maxima["churn"] > 0 and no_source_churn == 0,
    }[name]
    if not expected:
        errors.append(f"{name}: expected channel policy failed")
    if not np.isfinite(collision).all() or float(collision.min()) >= 0.0:
        errors.append(f"{name}: analytic collider field has no finite solid interior")
    scene_rows[name] = {
        "flow_family": profile.flow_family,
        "contract_colliders": [item.metadata() for item in contracts[name].colliders],
        "collision_sdf_range_m": [float(collision.min()), float(collision.max())],
        "channel_maxima": maxima,
        "churn_without_tagged_source": no_source_churn,
        "policy_passed": bool(expected),
    }

# Resolution normalization below the capillary length is checked on all profiles.
scale_rows = {}
for name, profile in profiles.items():
    outputs = []
    for spacing in (0.001, 0.002):
        model = EmissionModel(
            spacing,
            conditioning_sigma=profile.conditioning_sigma_cells,
            surface_half_width_cells=profile.surface_half_width_cells,
            churn_depth_cells=profile.churn_depth_cells,
            sphere_influence_cells=profile.churn_source_influence_cells,
            profile=profile,
        )
        raw = raw_emission_fields(
            scenario_fields(model, include_source=name in {"impact", "wake"}), model
        )
        outputs.append(
            {
                channel: float(raw[channel + "_raw"][6, 6, 6])
                for channel in ("spray", "entrained_air", "churn")
            }
        )
    differences = {
        channel: abs(outputs[0][channel] - outputs[1][channel])
        for channel in outputs[0]
    }
    if max(differences.values()) > 1.0e-6:
        errors.append(f"{name}: sub-capillary scale normalization failed")
    scale_rows[name] = differences

plateau_audit_path = args.plateau_audit.resolve()
render_audit_path = args.render_audit.resolve()
plateau_audit = json.loads(plateau_audit_path.read_text(encoding="utf-8"))
render_audit = json.loads(render_audit_path.read_text(encoding="utf-8"))
hero_passed = plateau_audit.get("valid") is True and render_audit.get("valid") is True
if not hero_passed:
    errors.append("hero Plateau foam acceptance failed")

report = {
    "schema": 1,
    "product": "whitewater_v6_cross_scene_acceptance",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "valid": not errors,
    "scope": {
        "analytic_scene_contracts": ["impact", "river", "waterfall", "wake"],
        "flow_channel_fixtures": ["impact", "river", "waterfall", "wake"],
        "production_hero_foam": "swamp rigid-body impact",
        "limitation": "River, waterfall and wake are contract/field acceptance fixtures, not separately authored full PhysX production shots.",
    },
    "criteria": {
        "four_scene_contract_families_build_collision_fields": not any("collider field" in item for item in errors),
        "flow_profiles_emit_only_policy_allowed_channels": not any("channel policy" in item for item in errors),
        "subcapillary_scaling_is_resolution_invariant": not any("scale normalization" in item for item in errors),
        "wake_churn_requires_a_tagged_hull": scene_rows["wake"]["churn_without_tagged_source"] == 0.0,
        "river_churn_is_disabled": scene_rows["river"]["channel_maxima"]["churn"] == 0.0,
        "waterfall_churn_can_use_global_plunge_support": scene_rows["waterfall"]["channel_maxima"]["churn"] > 0.0,
        "hero_plateau_geometry_and_render_are_audited": hero_passed,
    },
    "scenes": scene_rows,
    "subcapillary_channel_differences": scale_rows,
    "hero_foam": {
        "plateau_audit": str(plateau_audit_path),
        "plateau_audit_sha256": sha256_file(plateau_audit_path),
        "render_audit": str(render_audit_path),
        "render_audit_sha256": sha256_file(render_audit_path),
        "passed": hero_passed,
    },
    "profile_hashes": {
        name: sha256_file(path) for name, path in profile_paths.items()
    },
    "errors": errors,
}
output_path = args.output_report.resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.exists():
    raise FileExistsError(f"Refusing to overwrite {output_path}")
output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"valid": report["valid"], "criteria": report["criteria"], "scenes": scene_rows, "errors": errors}, indent=2))
if errors:
    raise SystemExit(1)
