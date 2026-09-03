"""Deterministic synthetic validation for the Warp whitewater feature kernels."""

from __future__ import annotations

import json

import numpy as np

from whitewater.features_warp import WarpFeatureComputer


spacing = 0.008
support_radius = 2.5 * spacing
axis = np.arange(12, dtype=np.float32) * spacing
grid_x, grid_y, grid_z = np.meshgrid(axis, axis, axis, indexing="ij")
positions = np.column_stack(
    (grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1))
).astype(np.float32)

expected_gradient = np.asarray(
    [
        [0.20, -0.10, 0.05],
        [0.30, -0.40, 0.10],
        [-0.20, 0.15, 0.50],
    ],
    dtype=np.float32,
)
velocity_offset = np.asarray((0.13, -0.07, 0.04), dtype=np.float32)
velocities = positions @ expected_gradient.T + velocity_offset

computer = WarpFeatureComputer(
    particle_count=len(positions),
    radius=support_radius,
    device="cuda:0",
    grid_dimension=32,
)
local = computer.compute_local(positions, velocities)

minimum = positions.min(axis=0)
maximum = positions.max(axis=0)
interior = np.all(
    (positions >= minimum + support_radius)
    & (positions <= maximum - support_radius),
    axis=1,
)
top_central = (
    (positions[:, 1] == maximum[1])
    & (positions[:, 0] >= minimum[0] + 2.0 * support_radius)
    & (positions[:, 0] <= maximum[0] - 2.0 * support_radius)
    & (positions[:, 2] >= minimum[2] + 2.0 * support_radius)
    & (positions[:, 2] <= maximum[2] - 2.0 * support_radius)
)

expected_divergence = float(np.trace(expected_gradient))
expected_vorticity = np.asarray(
    (
        expected_gradient[2, 1] - expected_gradient[1, 2],
        expected_gradient[0, 2] - expected_gradient[2, 0],
        expected_gradient[1, 0] - expected_gradient[0, 1],
    ),
    dtype=np.float32,
)
expected_symmetric = 0.5 * (expected_gradient + expected_gradient.T)
expected_deviator = expected_symmetric - np.eye(3) * expected_divergence / 3.0
expected_strain_rate = float(np.sqrt(np.sum(expected_deviator**2)))

divergence_error = np.abs(
    local["velocity_divergence"][interior] - expected_divergence
)
vorticity_error = np.linalg.norm(
    local["vorticity"][interior] - expected_vorticity, axis=1
)
strain_error = np.abs(local["strain_rate"][interior] - expected_strain_rate)
top_normals = local["surface_normal"][top_central]

surface_confidence = np.zeros(len(positions), dtype=np.float32)
surface_confidence[
    positions[:, 1] >= maximum[1] - 1.1 * spacing
] = 1.0
curvature_result = computer.compute_curvature(
    local["surface_normal"], surface_confidence
)
top_curvature = np.abs(curvature_result["curvature"][top_central])

metrics = {
    "particles": len(positions),
    "interior_particles": int(np.count_nonzero(interior)),
    "top_central_particles": int(np.count_nonzero(top_central)),
    "expected_divergence": expected_divergence,
    "divergence_error_p99": float(np.quantile(divergence_error, 0.99)),
    "vorticity_error_p99": float(np.quantile(vorticity_error, 0.99)),
    "strain_error_p99": float(np.quantile(strain_error, 0.99)),
    "top_normal_y_median": float(np.median(top_normals[:, 1])),
    "top_normal_horizontal_p99": float(
        np.quantile(np.linalg.norm(top_normals[:, [0, 2]], axis=1), 0.99)
    ),
    "flat_top_curvature_p99": float(np.quantile(top_curvature, 0.99)),
    "minimum_interior_neighbors": int(
        local["neighbor_count"][interior].min()
    ),
    "minimum_interior_gradient_quality": float(
        local["gradient_quality"][interior].min()
    ),
}
criteria = {
    "all_outputs_finite": all(
        np.isfinite(value).all()
        for value in (
            local["kernel_sum"],
            local["centroid_offset"],
            local["surface_normal"],
            local["gradient_quality"],
            local["velocity_divergence"],
            local["vorticity"],
            local["strain_rate"],
            local["velocity_dispersion"],
            curvature_result["curvature"],
            curvature_result["curvature_quality"],
        )
    ),
    "linear_divergence_recovered": metrics["divergence_error_p99"] <= 0.01,
    "linear_vorticity_recovered": metrics["vorticity_error_p99"] <= 0.01,
    "linear_strain_recovered": metrics["strain_error_p99"] <= 0.01,
    "top_normal_points_outward": metrics["top_normal_y_median"] >= 0.99,
    "top_normal_is_vertical": metrics["top_normal_horizontal_p99"] <= 0.01,
    "flat_surface_curvature_is_zero": metrics["flat_top_curvature_p99"] <= 0.5,
    "interior_has_sufficient_neighbors": metrics["minimum_interior_neighbors"] >= 40,
    "interior_gradient_is_well_conditioned": (
        metrics["minimum_interior_gradient_quality"] >= 0.8
    ),
}
report = {"valid": all(criteria.values()), "criteria": criteria, "metrics": metrics}
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(1)
