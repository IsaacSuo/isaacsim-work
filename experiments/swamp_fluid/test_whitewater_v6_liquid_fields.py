"""Synthetic gates for the v6 three-dimensional liquid-field contract."""

from __future__ import annotations

import json

import numpy as np

from whitewater.liquid_fields import (
    GridSpec,
    aligned_grid_spec,
    material_acceleration,
    reconstruct_liquid_fields,
    sample_regular_heightfield,
    sphere_collision_sdf,
    splat_particles_cic,
    surface_geometry_from_phi,
    terrain_collision_sdf,
    union_collision_sdf,
    velocity_differentials,
)


def quantile(values, q=0.99):
    values = np.asarray(values)
    return float(np.quantile(values, q)) if values.size else None


spacing = 0.04
axis = np.arange(-0.48, 0.4801, spacing, dtype=np.float64)
x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
interior = (
    (np.abs(x) <= 0.36) & (np.abs(y) <= 0.36) & (np.abs(z) <= 0.36)
)

# Analytic free surfaces validate geometry independently of particle sampling.
plane_phi = y.copy()
plane = surface_geometry_from_phi(plane_phi, spacing, narrow_band=2.0 * spacing)
plane_band = interior & (np.abs(plane_phi) <= spacing)
plane_normal_error = np.linalg.norm(
    plane["normal"][plane_band] - np.asarray((0.0, 1.0, 0.0)), axis=1
)

sphere_radius = 0.32
sphere_phi = np.sqrt(x * x + y * y + z * z) - sphere_radius
sphere = surface_geometry_from_phi(sphere_phi, spacing, narrow_band=spacing)
sphere_band = (np.abs(sphere_phi) <= 0.5 * spacing) & interior
sphere_exact_normal = np.stack((x, y, z), axis=-1)
sphere_exact_normal /= np.maximum(
    np.linalg.norm(sphere_exact_normal, axis=-1, keepdims=True), 1.0e-12
)
sphere_normal_error = np.linalg.norm(
    sphere["normal"][sphere_band] - sphere_exact_normal[sphere_band], axis=1
)
sphere_curvature_error = np.abs(
    sphere["curvature"][sphere_band] - 2.0 / sphere_radius
)

# A general affine velocity field must recover exact first derivatives.
gradient = np.asarray(
    ((0.20, -0.10, 0.05), (0.30, -0.40, 0.10), (-0.20, 0.15, 0.50)),
    dtype=np.float64,
)
offset = np.asarray((0.13, -0.07, 0.04), dtype=np.float64)
points = np.stack((x, y, z), axis=-1)
affine_velocity = np.einsum("...a,ca->...c", points, gradient) + offset
affine = velocity_differentials(affine_velocity, spacing)
expected_divergence = float(np.trace(gradient))
expected_vorticity = np.asarray(
    (
        gradient[2, 1] - gradient[1, 2],
        gradient[0, 2] - gradient[2, 0],
        gradient[1, 0] - gradient[0, 1],
    )
)
symmetry = 0.5 * (gradient + gradient.T)
deviator = symmetry - np.eye(3) * expected_divergence / 3.0
expected_strain = float(np.sqrt(np.sum(deviator * deviator)))

# Solid-body rotation validates curl and the convective acceleration term.
omega = 1.7
rotation_velocity = np.zeros(points.shape, dtype=np.float64)
rotation_velocity[..., 0] = -omega * z
rotation_velocity[..., 2] = omega * x
rotation = velocity_differentials(rotation_velocity, spacing)
rotation_acceleration = material_acceleration(
    rotation_velocity, rotation_velocity, 0.01, spacing
)
expected_rotation_acceleration = np.zeros(points.shape, dtype=np.float64)
expected_rotation_acceleration[..., 0] = -(omega**2) * x
expected_rotation_acceleration[..., 2] = -(omega**2) * z

# Uniform temporal acceleration must not acquire a convective component.
previous_uniform = np.zeros(points.shape, dtype=np.float64)
current_uniform = np.broadcast_to(
    np.asarray((0.2, -0.1, 0.05)), points.shape
).copy()
uniform_dt = 0.02
uniform_acceleration = material_acceleration(
    current_uniform, previous_uniform, uniform_dt, spacing
)
expected_uniform_acceleration = current_uniform[0, 0, 0] / uniform_dt

# CIC conservation and a small slab reconstruction validate particle input.
particle_axis = np.arange(-0.32, 0.3201, 0.08)
px, py, pz = np.meshgrid(
    particle_axis,
    np.arange(-0.32, 0.0001, 0.08),
    particle_axis,
    indexing="ij",
)
particle_positions = np.column_stack((px.ravel(), py.ravel(), pz.ravel()))
particle_velocities = np.tile((0.15, 0.0, -0.05), (len(particle_positions), 1))
particle_spec = aligned_grid_spec(
    particle_positions.min(axis=0),
    particle_positions.max(axis=0),
    spacing=0.08,
    padding=0.24,
)
particle_weight, _ = splat_particles_cic(
    particle_positions, particle_velocities, particle_spec
)
fields = reconstruct_liquid_fields(
    particle_positions,
    particle_velocities,
    particle_spec,
    bulk_weight_reference=1.0,
    iso_fraction=0.3,
    velocity_smoothing_sigma=0.8,
)
gx, gy, gz = particle_spec.axes()
center_x = int(np.argmin(np.abs(gx)))
center_z = int(np.argmin(np.abs(gz)))
inside_y = int(np.argmin(np.abs(gy + 0.16)))
air_y = int(np.argmin(np.abs(gy - 0.16)))
surface_band_mask = (
    (np.abs(fields["phi"]) <= 0.75 * particle_spec.spacing)
    & (fields["surface_valid"] != 0)
)
surface_normal_lengths = np.linalg.norm(
    fields["normal"][surface_band_mask], axis=1
)

# Collision fields use the same sign convention as liquid phi.
terrain_x = np.arange(-0.6, 0.6001, 0.08)
terrain_z = np.arange(-0.6, 0.6001, 0.08)
terrain_height = (
    0.1 * terrain_x[:, None] - 0.05 * terrain_z[None, :] - 0.25
)
terrain_endpoint_sample, terrain_endpoint_valid = sample_regular_heightfield(
    np.asarray((terrain_x[0], terrain_x[-1])),
    np.asarray((terrain_z[0], terrain_z[-1])),
    terrain_x,
    terrain_z,
    terrain_height,
)
terrain_endpoint_expected = (
    0.1 * np.asarray((terrain_x[0], terrain_x[-1]))[:, None]
    - 0.05 * np.asarray((terrain_z[0], terrain_z[-1]))[None, :]
    - 0.25
)
collision_spec = GridSpec((-0.48, -0.48, -0.48), 0.08, (13, 13, 13))
terrain_sdf, terrain_valid = terrain_collision_sdf(
    collision_spec, terrain_x, terrain_z, terrain_height
)
sphere_sdf = sphere_collision_sdf(collision_spec, (0.0, 0.0, 0.0), 0.2)
collision_union = union_collision_sdf(terrain_sdf, sphere_sdf)
cx, cy, cz = [size // 2 for size in collision_spec.shape]

metrics = {
    "plane_normal_error_p99": quantile(plane_normal_error),
    "plane_curvature_absolute_max": float(
        np.max(np.abs(plane["curvature"][plane_band]))
    ),
    "sphere_normal_error_p99": quantile(sphere_normal_error),
    "sphere_curvature_error_p99": quantile(sphere_curvature_error),
    "affine_divergence_error_max": float(
        np.max(np.abs(affine["divergence"][interior] - expected_divergence))
    ),
    "affine_vorticity_error_max": float(
        np.max(
            np.linalg.norm(
                affine["vorticity"][interior] - expected_vorticity, axis=1
            )
        )
    ),
    "affine_strain_error_max": float(
        np.max(np.abs(affine["strain_rate"][interior] - expected_strain))
    ),
    "rotation_divergence_absolute_max": float(
        np.max(np.abs(rotation["divergence"][interior]))
    ),
    "rotation_vorticity_error_max": float(
        np.max(
            np.linalg.norm(
                rotation["vorticity"][interior] - (0.0, -2.0 * omega, 0.0),
                axis=1,
            )
        )
    ),
    "rotation_acceleration_error_max": float(
        np.max(
            np.linalg.norm(
                rotation_acceleration[interior]
                - expected_rotation_acceleration[interior],
                axis=1,
            )
        )
    ),
    "uniform_acceleration_error_max": float(
        np.max(
            np.linalg.norm(
                uniform_acceleration[interior] - expected_uniform_acceleration,
                axis=1,
            )
        )
    ),
    "cic_weight_error": float(abs(np.sum(particle_weight) - len(particle_positions))),
    "reconstruction_inside_phi": float(fields["phi"][center_x, inside_y, center_z]),
    "reconstruction_air_phi": float(fields["phi"][center_x, air_y, center_z]),
    "surface_normal_length_error_p99": quantile(
        np.abs(surface_normal_lengths - 1.0)
    ),
    "terrain_valid_fraction": float(np.mean(terrain_valid)),
    "terrain_endpoint_error_max": float(
        np.max(np.abs(terrain_endpoint_sample - terrain_endpoint_expected))
    ),
    "sphere_center_sdf": float(sphere_sdf[cx, cy, cz]),
    "collision_union_center_sdf": float(collision_union[cx, cy, cz]),
}

finite_outputs = (
    plane["normal"],
    plane["curvature"],
    sphere["normal"],
    sphere["curvature"],
    affine["divergence"],
    affine["vorticity"],
    affine["strain_rate"],
    rotation_acceleration,
    uniform_acceleration,
    fields["phi"],
    fields["velocity"],
    collision_union,
)
criteria = {
    "all_outputs_finite": all(np.isfinite(value).all() for value in finite_outputs),
    "plane_normal_is_exact": metrics["plane_normal_error_p99"] <= 1.0e-6,
    "plane_curvature_is_zero": metrics["plane_curvature_absolute_max"] <= 1.0e-6,
    "sphere_normal_is_accurate": metrics["sphere_normal_error_p99"] <= 0.015,
    "sphere_curvature_is_accurate": metrics["sphere_curvature_error_p99"] <= 0.7,
    "affine_divergence_is_exact": metrics["affine_divergence_error_max"] <= 1.0e-5,
    "affine_vorticity_is_exact": metrics["affine_vorticity_error_max"] <= 1.0e-5,
    "affine_strain_is_exact": metrics["affine_strain_error_max"] <= 1.0e-5,
    "rotation_divergence_is_zero": metrics["rotation_divergence_absolute_max"] <= 1.0e-5,
    "rotation_vorticity_is_exact": metrics["rotation_vorticity_error_max"] <= 1.0e-5,
    "material_acceleration_is_exact": metrics["rotation_acceleration_error_max"] <= 1.0e-5,
    "temporal_acceleration_is_exact": metrics["uniform_acceleration_error_max"] <= 1.0e-5,
    "cic_is_conservative": metrics["cic_weight_error"] <= 1.0e-6,
    "reconstruction_has_correct_sign": (
        metrics["reconstruction_inside_phi"] < 0.0
        and metrics["reconstruction_air_phi"] > 0.0
    ),
    "reconstruction_normals_are_unit": metrics["surface_normal_length_error_p99"] <= 1.0e-5,
    "terrain_covers_collision_grid": metrics["terrain_valid_fraction"] == 1.0,
    "terrain_endpoints_are_exact": (
        bool(np.all(terrain_endpoint_valid))
        and metrics["terrain_endpoint_error_max"] <= 1.0e-7
    ),
    "sphere_sdf_has_correct_sign": metrics["sphere_center_sdf"] < 0.0,
    "collision_union_has_correct_sign": metrics["collision_union_center_sdf"] < 0.0,
}
report = {
    "schema": 1,
    "suite": "whitewater_v6_liquid_fields",
    "valid": all(criteria.values()),
    "criteria": criteria,
    "metrics": metrics,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["valid"]:
    raise SystemExit(1)
