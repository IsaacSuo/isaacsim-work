"""Independently audit a v6 three-dimensional liquid-field cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from whitewater.domain_partition import (
    load_domain_partition_contract,
    load_particle_classification,
)


REQUIRED_SCALARS = (
    "particle_weight",
    "number_density",
    "fluid_mask",
    "phi",
    "depth",
    "velocity_valid",
    "curvature",
    "surface_valid",
    "divergence",
    "strain_rate",
    "acceleration_valid",
    "collision_sdf",
)
REQUIRED_VECTORS = ("velocity", "normal", "vorticity", "acceleration")
OPTIONAL_COLLISION_SCALARS = (
    "collider_collision_sdf",
    "dynamic_collision_sdf",
    "churn_source_sdf",
    "dynamic_churn_source_sdf",
    "sphere_collision_sdf",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field_directory", type=Path)
    return parser.parse_args()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload, stream, indent=2, sort_keys=True, default=lambda value: value.item()
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


args = parse_args()
field_directory = args.field_directory.resolve()
manifest_path = field_directory / "manifest.json"
audit_path = field_directory / "audit_report.json"
if audit_path.exists():
    raise FileExistsError(f"Refusing to overwrite existing audit: {audit_path}")
manifest = load_json(manifest_path)
errors = []
if manifest.get("schema") != 1 or manifest.get("product") != "whitewater_v6_liquid_fields":
    errors.append("Unsupported manifest schema or product")
if manifest.get("state", {}).get("complete") is not True:
    errors.append("Field manifest is incomplete")

source_manifest_path = Path(manifest["source"]["manifest"])
source_audit_path = Path(manifest["source"]["audit"])
terrain_path = Path(manifest["terrain"]["path"])
provenance_assets = [
    (source_manifest_path, manifest["source"]["manifest_sha256"], "source manifest"),
    (source_audit_path, manifest["source"]["audit_sha256"], "source audit"),
    (terrain_path, manifest["terrain"]["sha256"], "terrain"),
]
if manifest["terrain"].get("selection_path") is not None:
    provenance_assets.append(
        (
            Path(manifest["terrain"]["selection_path"]),
            manifest["terrain"].get("selection_sha256"),
            "terrain selection record",
        )
    )
for path, expected, label in provenance_assets:
    if not path.is_file() or sha256_file(path) != expected:
        errors.append(f"Provenance mismatch: {label}")
scene_contract_metadata = manifest.get("scene_contract", {})
if scene_contract_metadata.get("mode") == "file":
    scene_contract_path = Path(scene_contract_metadata["path"])
    if (
        not scene_contract_path.is_file()
        or sha256_file(scene_contract_path) != scene_contract_metadata.get("sha256")
    ):
        errors.append("Provenance mismatch: scene contract")

static_path = field_directory / manifest["static_fields"]["file"]
if not static_path.is_file() or sha256_file(static_path) != manifest["static_fields"]["sha256"]:
    errors.append("Static grid field hash mismatch")
grid_shape = tuple(int(value) for value in manifest["grid"]["shape"])
grid_origin = np.asarray(manifest["grid"]["origin"], dtype=np.float64)
spacing = float(manifest["grid"]["spacing"])
iso_fraction = float(manifest["configuration"]["iso_fraction"])
role_based_contract = "scene_contract" in manifest
static_terrain_sdf = None
static_collider_sdf = None
static_churn_source_sdf = None
if static_path.is_file():
    with np.load(static_path, allow_pickle=False) as static_cache:
        declared_static_arrays = manifest.get("static_fields", {}).get("arrays")
        if declared_static_arrays is not None and set(declared_static_arrays) != set(
            static_cache.files
        ):
            errors.append("Static-field array declaration does not match the NPZ")
        for axis_name, axis_size in zip(("x", "y", "z"), grid_shape):
            if axis_name not in static_cache or np.asarray(static_cache[axis_name]).shape != (
                axis_size,
            ):
                errors.append(f"Static grid axis mismatch: {axis_name}")
        if "terrain_collision_sdf" not in static_cache:
            errors.append("Static cache is missing terrain_collision_sdf")
        else:
            static_terrain_sdf = np.asarray(
                static_cache["terrain_collision_sdf"], dtype=np.float32
            )
        if "static_collider_collision_sdf" in static_cache:
            static_collider_sdf = np.asarray(
                static_cache["static_collider_collision_sdf"], dtype=np.float32
            )
        if "static_churn_source_sdf" in static_cache:
            static_churn_source_sdf = np.asarray(
                static_cache["static_churn_source_sdf"], dtype=np.float32
            )
        for name, value in (
            ("terrain_collision_sdf", static_terrain_sdf),
            ("static_collider_collision_sdf", static_collider_sdf),
            ("static_churn_source_sdf", static_churn_source_sdf),
        ):
            if value is not None and (
                value.shape != grid_shape or not np.isfinite(value).all()
            ):
                errors.append(f"Invalid static collision field: {name}")

source_manifest = load_json(source_manifest_path)
source_directory = Path(manifest["source"]["directory"])
source_rows = {int(row["sample_index"]): row for row in source_manifest["samples"]}
domain_contract = None
domain_metadata = manifest.get("domain_partition")
if domain_metadata is not None:
    try:
        domain_contract = load_domain_partition_contract(
            domain_metadata["manifest"],
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            requested_spacing=spacing,
            body_id=domain_metadata["body_id"],
        )
        if (
            domain_contract["manifest_sha256"]
            != domain_metadata.get("manifest_sha256")
            or domain_contract["membership_sha256"]
            != domain_metadata.get("membership_sha256")
            or domain_contract["body_index"] != int(domain_metadata.get("body_index", -1))
        ):
            errors.append("Liquid-field domain provenance declaration changed")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"Domain partition validation failed: {exc}")

minimum_normal_length = np.inf
maximum_normal_length = -np.inf
maximum_depth_error = 0.0
maximum_invalid_velocity = 0.0
maximum_invalid_acceleration = 0.0
minimum_particle_support_fraction = 1.0
minimum_collision_clear_fraction = 1.0
minimum_pair_jaccard = 1.0
maximum_phi_change = 0.0
minimum_core_inside_grid_fraction = 1.0
domain_ledger_is_conservative = True
previous_mask = None
previous_phi = None
audited_samples = []

for item in manifest.get("samples", []):
    frame_path = field_directory / item["file"]
    if not frame_path.is_file() or sha256_file(frame_path) != item["sha256"]:
        errors.append(f"Field frame hash mismatch: {frame_path.name}")
        continue
    with np.load(frame_path, allow_pickle=False) as cache:
        names = set(cache.files)
        missing = set(REQUIRED_SCALARS + REQUIRED_VECTORS) - names
        if missing:
            errors.append(f"{frame_path.name} missing fields: {sorted(missing)}")
            continue
        scalars = {name: np.asarray(cache[name]) for name in REQUIRED_SCALARS}
        optional_collision = {
            name: np.asarray(cache[name])
            for name in OPTIONAL_COLLISION_SCALARS
            if name in cache
        }
        vectors = {name: np.asarray(cache[name]) for name in REQUIRED_VECTORS}
        source_index = int(cache["source_sample_index"])
        if int(cache["schema"]) != 1 or source_index != int(item["source_sample_index"]):
            errors.append(f"Metadata mismatch: {frame_path.name}")
        for name, value in scalars.items():
            if value.shape != grid_shape:
                errors.append(f"{frame_path.name}:{name} shape {value.shape}")
        for name, value in optional_collision.items():
            if value.shape != grid_shape:
                errors.append(f"{frame_path.name}:{name} shape {value.shape}")
        for name, value in vectors.items():
            if value.shape != grid_shape + (3,):
                errors.append(f"{frame_path.name}:{name} shape {value.shape}")
        if not all(
            np.isfinite(value).all()
            for value in (
                *scalars.values(),
                *optional_collision.values(),
                *vectors.values(),
            )
        ):
            errors.append(f"Non-finite field values: {frame_path.name}")
            continue

        if role_based_contract and static_terrain_sdf is not None:
            dynamic_collider = optional_collision.get("dynamic_collision_sdf")
            collider_components = [
                value
                for value in (static_collider_sdf, dynamic_collider)
                if value is not None
            ]
            expected_collider = None
            if collider_components:
                expected_collider = collider_components[0].copy()
                for component in collider_components[1:]:
                    np.minimum(expected_collider, component, out=expected_collider)
            actual_collider = optional_collision.get("collider_collision_sdf")
            if expected_collider is None:
                if actual_collider is not None:
                    errors.append(f"Unexpected collider union: {frame_path.name}")
                expected_collision = static_terrain_sdf
            else:
                if actual_collider is None or not np.array_equal(
                    actual_collider, expected_collider
                ):
                    errors.append(f"Collider union mismatch: {frame_path.name}")
                expected_collision = np.minimum(static_terrain_sdf, expected_collider)
            if not np.array_equal(scalars["collision_sdf"], expected_collision):
                errors.append(f"Terrain/collider union mismatch: {frame_path.name}")

            dynamic_churn = optional_collision.get("dynamic_churn_source_sdf")
            churn_components = [
                value
                for value in (static_churn_source_sdf, dynamic_churn)
                if value is not None
            ]
            expected_churn = None
            if churn_components:
                expected_churn = churn_components[0].copy()
                for component in churn_components[1:]:
                    np.minimum(expected_churn, component, out=expected_churn)
            actual_churn = optional_collision.get("churn_source_sdf")
            if expected_churn is None:
                if actual_churn is not None:
                    errors.append(f"Unexpected churn-source union: {frame_path.name}")
            elif actual_churn is None or not np.array_equal(actual_churn, expected_churn):
                errors.append(f"Churn-source union mismatch: {frame_path.name}")

        phi = scalars["phi"]
        fluid = scalars["fluid_mask"] != 0
        expected_fluid = scalars["number_density"] >= iso_fraction
        if not np.array_equal(fluid, expected_fluid):
            errors.append(f"Fluid mask/density mismatch: {frame_path.name}")
        if np.any(phi[fluid] >= 0.0) or np.any(phi[~fluid] <= 0.0):
            errors.append(f"Liquid SDF sign mismatch: {frame_path.name}")
        maximum_depth_error = max(
            maximum_depth_error,
            float(np.max(np.abs(scalars["depth"] - np.maximum(-phi, 0.0)))),
        )

        surface = scalars["surface_valid"] != 0
        normal_lengths = np.linalg.norm(vectors["normal"][surface], axis=1)
        if len(normal_lengths):
            minimum_normal_length = min(minimum_normal_length, float(normal_lengths.min()))
            maximum_normal_length = max(maximum_normal_length, float(normal_lengths.max()))
        invalid_velocity = scalars["velocity_valid"] == 0
        invalid_acceleration = scalars["acceleration_valid"] == 0
        maximum_invalid_velocity = max(
            maximum_invalid_velocity,
            float(np.max(np.abs(vectors["velocity"][invalid_velocity]), initial=0.0)),
        )
        maximum_invalid_acceleration = max(
            maximum_invalid_acceleration,
            float(np.max(np.abs(vectors["acceleration"][invalid_acceleration]), initial=0.0)),
        )

        source_row = source_rows[source_index]
        source_path = source_directory / source_row["file"]
        if sha256_file(source_path) != source_row["sha256"]:
            errors.append(f"Source frame hash mismatch: {source_path.name}")
            continue
        with np.load(source_path, allow_pickle=False) as source:
            all_positions = (
                np.asarray(source["positions"], dtype=np.float64)
                * float(manifest["configuration"].get("metres_per_unit", 1.0))
            )
        positions = all_positions
        classification = None
        if domain_contract is not None:
            try:
                classification = load_particle_classification(
                    domain_contract, source_index, len(all_positions)
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                errors.append(
                    f"Domain classification validation failed for {source_index}: {exc}"
                )
                continue
            core_indices = classification["selected_core_indices"]
            secondary_indices = classification["selected_secondary_indices"]
            stable_indices = domain_contract["body_particle_indices"]
            domain_ledger_is_conservative &= (
                len(np.intersect1d(core_indices, secondary_indices, assume_unique=True)) == 0
                and len(core_indices) + len(secondary_indices) == len(stable_indices)
                and np.array_equal(
                    np.sort(np.concatenate((core_indices, secondary_indices))),
                    stable_indices,
                )
            )
            if (
                int(item.get("liquid_core_input_particles", -1)) != len(core_indices)
                or int(item.get("secondary_handoff_particles", -1))
                != len(secondary_indices)
                or item.get("domain_classification_file")
                != classification["path"].name
                or item.get("domain_classification_sha256")
                != classification["sha256"]
            ):
                errors.append(f"Domain handoff ledger mismatch: {frame_path.name}")
            positions = all_positions[core_indices]
        indices = np.rint((positions - grid_origin) / spacing).astype(np.int64)
        inside_grid = np.all(
            (indices >= 0) & (indices < np.asarray(grid_shape)), axis=1
        )
        clipped = np.clip(indices, 0, np.asarray(grid_shape) - 1)
        particle_phi = phi[clipped[:, 0], clipped[:, 1], clipped[:, 2]]
        particle_collision = scalars["collision_sdf"][
            clipped[:, 0], clipped[:, 1], clipped[:, 2]
        ]
        support_fraction = float(np.mean(inside_grid & (particle_phi <= 1.5 * spacing)))
        collision_clear_fraction = float(
            np.mean(inside_grid & (particle_collision >= -1.5 * spacing))
        )
        minimum_particle_support_fraction = min(
            minimum_particle_support_fraction, support_fraction
        )
        minimum_collision_clear_fraction = min(
            minimum_collision_clear_fraction, collision_clear_fraction
        )
        minimum_core_inside_grid_fraction = min(
            minimum_core_inside_grid_fraction, float(np.mean(inside_grid))
        )

        if previous_mask is not None:
            intersection = np.count_nonzero(previous_mask & fluid)
            union = np.count_nonzero(previous_mask | fluid)
            jaccard = intersection / union if union else 1.0
            minimum_pair_jaccard = min(minimum_pair_jaccard, jaccard)
            maximum_phi_change = max(
                maximum_phi_change, float(np.max(np.abs(phi - previous_phi)))
            )
        previous_mask = fluid
        previous_phi = phi.copy()
        audited_samples.append(
            {
                "source_sample_index": source_index,
                "source_particles": int(len(all_positions)),
                "liquid_core_particles": int(len(positions)),
                "secondary_handoff_particles": (
                    int(len(classification["selected_secondary_indices"]))
                    if classification is not None
                    else 0
                ),
                "particle_support_fraction": support_fraction,
                "collision_clear_fraction": collision_clear_fraction,
            }
        )

if not np.isfinite(minimum_normal_length):
    minimum_normal_length = 0.0
if not np.isfinite(maximum_normal_length):
    maximum_normal_length = 0.0
criteria = {
    "structural_checks_pass": not errors,
    "all_expected_samples_audited": len(audited_samples) == len(manifest.get("samples", [])),
    "surface_normals_are_unit": minimum_normal_length >= 0.99 and maximum_normal_length <= 1.01,
    "depth_matches_liquid_sdf": maximum_depth_error <= 1.0e-6,
    "invalid_velocity_is_zero": maximum_invalid_velocity == 0.0,
    "invalid_acceleration_is_zero": maximum_invalid_acceleration == 0.0,
    "particles_are_supported_by_liquid_sdf": minimum_particle_support_fraction >= 0.995,
    "particles_clear_collision_sdf": minimum_collision_clear_fraction >= 0.999,
    "liquid_core_particles_are_inside_grid": minimum_core_inside_grid_fraction == 1.0,
    "domain_particle_ledger_is_conservative": domain_ledger_is_conservative,
    "consecutive_fluid_masks_are_continuous": minimum_pair_jaccard >= 0.90,
}
report = {
    "schema": 1,
    "product": "whitewater_v6_liquid_fields_audit",
    "created_utc": utc_now_iso(),
    "field_directory": str(field_directory),
    "manifest_sha256": sha256_file(manifest_path),
    "valid": all(criteria.values()),
    "criteria": criteria,
    "thresholds": {
        "minimum_particle_support_fraction": 0.995,
        "minimum_collision_clear_fraction": 0.999,
        "minimum_pair_jaccard": 0.90,
    },
    "metrics": {
        "samples": len(audited_samples),
        "minimum_normal_length": minimum_normal_length,
        "maximum_normal_length": maximum_normal_length,
        "maximum_depth_error": maximum_depth_error,
        "maximum_invalid_velocity": maximum_invalid_velocity,
        "maximum_invalid_acceleration": maximum_invalid_acceleration,
        "minimum_particle_support_fraction": minimum_particle_support_fraction,
        "minimum_collision_clear_fraction": minimum_collision_clear_fraction,
        "minimum_core_inside_grid_fraction": minimum_core_inside_grid_fraction,
        "domain_particle_ledger_is_conservative": domain_ledger_is_conservative,
        "minimum_pair_jaccard": minimum_pair_jaccard,
        "maximum_phi_change": maximum_phi_change,
    },
    "samples": audited_samples,
    "errors": errors,
}
atomic_json(audit_path, report)
print(
    json.dumps(
        report, indent=2, sort_keys=True, default=lambda value: value.item()
    )
)
if not report["valid"]:
    raise SystemExit(1)
