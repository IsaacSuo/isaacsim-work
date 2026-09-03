"""Regression tests for the scene-independent primary-liquid pour source."""

from __future__ import annotations

import copy
import json

import numpy as np

from whitewater.pour_source import PourSource


def contract():
    return {
        "schema": 1,
        "product": "physx_primary_liquid_pour_source",
        "name": "unit_test_pour",
        "coordinate_system": {
            "axes": "xyz",
            "handedness": "right",
            "up_axis": "y",
            "metres_per_unit": 1.0,
        },
        "outlet": {
            "shape": "ellipse",
            "radii_m": [0.075, 0.135],
            "pattern": "staggered_hex",
            "stable_warp_fraction": 0.12,
        },
        "flow": {
            "rate_m3_s": 0.020,
            "speed_m_s": 2.0,
            "velocity_profile": {
                "kind": "power_law",
                "edge_fraction": 0.68,
                "exponent": 1.4,
            },
            "tangent_perturbation_m_s": 0.04,
            "seed": 20260901,
        },
        "timing": {"start_seconds": 0.0, "stop_seconds": 1.2},
        "motion": {
            "kind": "keyframed_rigid",
            "interpolation": "linear_orthonormalized",
            "keyframes": [
                {
                    "time_seconds": 0.0,
                    "centre": [3.2, 1.4, 1.7],
                    "normal": [0.96, -0.28, 0.0],
                    "up_hint": [0.0, 1.0, 0.0],
                },
                {
                    "time_seconds": 1.2,
                    "centre": [3.25, 1.35, 1.72],
                    "normal": [0.94, -0.34, 0.0],
                    "up_hint": [0.0, 1.0, 0.0],
                },
            ],
        },
        "metadata": {"purpose": "unit test"},
    }


source = PourSource.from_mapping(contract())
spacing = 0.03
physics_fps = 240
schedule = source.schedule_for_spacing(physics_fps, spacing)
audit = source.schedule_audit(physics_fps, spacing)
first = source.particle_batch(0, schedule[0], physics_fps, spacing)
repeat = source.particle_batch(0, schedule[0], physics_fps, spacing)
second = source.particle_batch(1, schedule[1], physics_fps, spacing)
pose_mid = source.pose_at(0.6)

phased_contract = copy.deepcopy(contract())
phased_contract["schema"] = 2
phased_contract["emission"] = {
    "kind": "phased_cross_section",
    "phase_count": 4,
}
phased_source = PourSource.from_mapping(phased_contract)
phased_schedule = phased_source.schedule_for_spacing(physics_fps, spacing)
phased_audit = phased_source.schedule_audit(physics_fps, spacing)
first_phased_layer = [
    phased_source.particle_batch(index, phased_schedule[index], physics_fps, spacing)
    for index in range(phased_source.emission.phase_count)
]
phased_layer_coordinates = np.concatenate(
    [batch.local_coordinates for batch in first_phased_layer], axis=0
)

usable = np.asarray(source.outlet.radii_m) - 0.5 * spacing
radius_squared = np.sum((first.local_coordinates / usable) ** 2, axis=1)
axial_speed = first.velocities @ source.pose_at(first.birth_time_seconds).normal

unknown_rejected = False
invalid = copy.deepcopy(contract())
invalid["outlet"]["shortcut_noise"] = True
try:
    PourSource.from_mapping(invalid)
except ValueError:
    unknown_rejected = True

criteria = {
    "strict_schema_rejects_unknown_keys": unknown_rejected,
    "schedule_is_nonempty_and_monotonic": bool(schedule) and list(schedule) == sorted(schedule),
    "flow_volume_quantization_is_bounded_by_one_batch": (
        abs(audit["emitted_volume_m3"] - audit["target_volume_m3"])
        <= audit["batch_volume_m3"] + 1.0e-12
    ),
    "batch_has_nonrectangular_ellipse_support": (
        len(first.positions) >= 8 and float(np.max(radius_squared)) <= 1.0 + 1.0e-6
    ),
    "same_batch_is_bitwise_deterministic": (
        np.array_equal(first.positions, repeat.positions)
        and np.array_equal(first.velocities, repeat.velocities)
    ),
    "successive_birth_layers_are_not_identical_grids": not np.array_equal(
        first.positions, second.positions
    ),
    "velocity_profile_is_faster_in_the_core": float(np.max(axial_speed))
    > float(np.min(axial_speed)),
    "tangent_perturbation_has_no_batch_drift": (
        np.linalg.norm(
            np.mean(first.velocities, axis=0)
            - np.mean(axial_speed) * source.pose_at(first.birth_time_seconds).normal
        )
        <= 1.0e-6
    ),
    "mid_pose_is_rigid_and_interpolated": (
        np.isclose(np.linalg.norm(pose_mid.normal), 1.0)
        and np.isclose(np.dot(pose_mid.normal, pose_mid.tangent_u), 0.0, atol=1.0e-7)
        and np.allclose(pose_mid.centre, [3.225, 1.375, 1.71])
    ),
    "metadata_round_trip_is_physics_equivalent": (
        PourSource.from_mapping(source.metadata()).configuration_sha256()
        == source.configuration_sha256()
    ),
    "schema_2_phases_partition_one_complete_layer": (
        sum(len(batch.positions) for batch in first_phased_layer)
        == phased_source.full_layer_particle_count(spacing)
        and len(np.unique(phased_layer_coordinates, axis=0))
        == len(phased_layer_coordinates)
    ),
    "schema_2_reduces_birth_batch_size": (
        phased_audit["maximum_particles_per_batch"]
        < phased_audit["full_layer_particles"]
    ),
    "schema_2_mass_error_is_bounded_by_one_subbatch": (
        abs(phased_audit["emitted_volume_m3"] - phased_audit["target_volume_m3"])
        <= phased_audit["maximum_batch_volume_m3"] + 1.0e-12
    ),
    "schema_2_emits_at_finer_cadence_than_schema_1": (
        len(phased_schedule) > len(schedule)
        and max(np.diff(phased_schedule)) < max(np.diff(schedule))
    ),
}
criteria = {name: bool(value) for name, value in criteria.items()}
report = {
    "schema": 1,
    "suite": "primary_liquid_pour_source",
    "valid": bool(all(criteria.values())),
    "criteria": criteria,
    "metrics": {
        "particles_per_batch": len(first.positions),
        "birth_batches": len(schedule),
        "first_birth_step": schedule[0],
        "last_birth_step": schedule[-1],
        "schedule_audit": audit,
        "phased_schedule_audit": phased_audit,
        "configuration_sha256": source.configuration_sha256(),
    },
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
