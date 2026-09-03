"""Warp GPU kernels for full-particle whitewater source feature extraction."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def load_warp():
    """Load the Warp version bundled with Isaac Sim without starting Kit."""
    try:
        import warp as wp

        return wp
    except ModuleNotFoundError:
        isaac_root = Path(os.environ.get("ISAACSIM_ROOT", r"Y:\isaacsim"))
        candidates = sorted(
            (isaac_root / "extscache").glob("omni.warp.core-*+wx64")
        )
        if not candidates:
            raise RuntimeError(
                f"Could not locate Isaac Sim's bundled Warp under {isaac_root}"
            )
        sys.path.insert(0, str(candidates[-1]))
        import warp as wp

        return wp


wp = load_warp()
wp.init()


@wp.func
def compact_weight(distance: float, radius: float):
    q = distance / radius
    if q >= 1.0:
        return 0.0
    one_minus_q2 = 1.0 - q * q
    return one_minus_q2 * one_minus_q2 * one_minus_q2


@wp.kernel
def compute_local_features(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    radius: float,
    neighbor_count: wp.array(dtype=wp.int32),
    kernel_sum: wp.array(dtype=float),
    centroid_offset: wp.array(dtype=wp.vec3),
    surface_normal: wp.array(dtype=wp.vec3),
    gradient_quality: wp.array(dtype=float),
    velocity_divergence: wp.array(dtype=float),
    vorticity: wp.array(dtype=wp.vec3),
    strain_rate: wp.array(dtype=float),
    velocity_dispersion: wp.array(dtype=float),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    xi = positions[i]
    vi = velocities[i]

    count = int(0)
    weight_sum = float(0.0)
    weighted_offset = wp.vec3()
    weighted_velocity_variance = float(0.0)
    moment = wp.mat33()
    velocity_moment = wp.mat33()

    neighbors = wp.hash_grid_query(grid, xi, radius)
    for j in neighbors:
        displacement = positions[j] - xi
        distance = wp.length(displacement)
        if distance <= radius:
            weight = compact_weight(distance, radius)
            count += 1
            weight_sum += weight
            if j != i:
                velocity_delta = velocities[j] - vi
                weighted_offset += weight * displacement
                weighted_velocity_variance += weight * wp.dot(
                    velocity_delta, velocity_delta
                )
                moment += weight * wp.outer(displacement, displacement)
                velocity_moment += weight * wp.outer(
                    velocity_delta, displacement
                )

    normal = wp.vec3()
    offset = wp.vec3()
    divergence = float(0.0)
    curl = wp.vec3()
    strain_norm = float(0.0)
    quality = float(0.0)
    dispersion = float(0.0)
    if weight_sum > 0.0:
        offset = weighted_offset / weight_sum
        offset_length = wp.length(offset)
        if offset_length > radius * 1.0e-6:
            normal = -offset / offset_length
        dispersion = wp.sqrt(
            wp.max(weighted_velocity_variance / weight_sum, 0.0)
        )

    moment_trace = moment[0, 0] + moment[1, 1] + moment[2, 2]
    determinant = wp.determinant(moment)
    denominator = wp.max(
        (moment_trace / 3.0) * (moment_trace / 3.0) * (moment_trace / 3.0),
        1.0e-30,
    )
    quality = wp.max(determinant / denominator, 0.0)
    if count >= 8 and moment_trace > radius * radius * 1.0e-5:
        regularization = wp.max(weight_sum, 1.0) * radius * radius * 1.0e-6
        corrected_moment = moment + wp.identity(n=3, dtype=float) * regularization
        velocity_gradient = velocity_moment * wp.inverse(corrected_moment)
        divergence = (
            velocity_gradient[0, 0]
            + velocity_gradient[1, 1]
            + velocity_gradient[2, 2]
        )
        curl = wp.vec3(
            velocity_gradient[2, 1] - velocity_gradient[1, 2],
            velocity_gradient[0, 2] - velocity_gradient[2, 0],
            velocity_gradient[1, 0] - velocity_gradient[0, 1],
        )
        symmetric = 0.5 * (
            velocity_gradient + wp.transpose(velocity_gradient)
        )
        mean_rate = divergence / 3.0
        d00 = symmetric[0, 0] - mean_rate
        d11 = symmetric[1, 1] - mean_rate
        d22 = symmetric[2, 2] - mean_rate
        strain_norm = wp.sqrt(
            wp.max(
                d00 * d00
                + d11 * d11
                + d22 * d22
                + 2.0
                * (
                    symmetric[0, 1] * symmetric[0, 1]
                    + symmetric[0, 2] * symmetric[0, 2]
                    + symmetric[1, 2] * symmetric[1, 2]
                ),
                0.0,
            )
        )

    neighbor_count[i] = count
    kernel_sum[i] = weight_sum
    centroid_offset[i] = offset
    surface_normal[i] = normal
    gradient_quality[i] = quality
    velocity_divergence[i] = divergence
    vorticity[i] = curl
    strain_rate[i] = strain_norm
    velocity_dispersion[i] = dispersion


@wp.kernel
def compute_surface_curvature(
    grid: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    surface_confidence: wp.array(dtype=float),
    radius: float,
    curvature: wp.array(dtype=float),
    curvature_quality: wp.array(dtype=float),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    confidence_i = surface_confidence[i]
    if confidence_i < 0.05:
        curvature[i] = 0.0
        curvature_quality[i] = 0.0
        return

    xi = positions[i]
    ni = normals[i]
    count = int(0)
    weight_sum = float(0.0)
    moment = wp.mat33()
    normal_moment = wp.mat33()
    tangent_projector = wp.identity(n=3, dtype=float) - wp.outer(ni, ni)
    neighbors = wp.hash_grid_query(grid, xi, radius)
    for j in neighbors:
        if j != i and surface_confidence[j] >= 0.025:
            displacement = positions[j] - xi
            distance = wp.length(displacement)
            if distance <= radius:
                weight = (
                    compact_weight(distance, radius)
                    * wp.min(surface_confidence[j], confidence_i)
                )
                if weight > 0.0:
                    tangent_displacement = tangent_projector * displacement
                    normal_separation = wp.abs(wp.dot(displacement, ni))
                    normal_delta = normals[j] - ni
                    if wp.length(normal_delta) < 1.0e-4:
                        normal_delta = wp.vec3()
                    if (
                        wp.length(tangent_displacement) > radius * 1.0e-6
                        and normal_separation <= 0.15 * radius
                        and wp.dot(normals[j], ni) > 0.25
                    ):
                        count += 1
                        weight_sum += weight
                        moment += weight * wp.outer(
                            tangent_displacement, tangent_displacement
                        )
                        normal_moment += weight * wp.outer(
                            normal_delta, tangent_displacement
                        )

    value = float(0.0)
    quality = float(0.0)
    moment_trace = moment[0, 0] + moment[1, 1] + moment[2, 2]
    moment_square_trace = (
        moment[0, 0] * moment[0, 0]
        + moment[1, 1] * moment[1, 1]
        + moment[2, 2] * moment[2, 2]
        + 2.0
        * (
            moment[0, 1] * moment[1, 0]
            + moment[0, 2] * moment[2, 0]
            + moment[1, 2] * moment[2, 1]
        )
    )
    tangent_determinant = 0.5 * (
        moment_trace * moment_trace - moment_square_trace
    )
    quality = wp.clamp(
        4.0
        * tangent_determinant
        / wp.max(moment_trace * moment_trace, 1.0e-30),
        0.0,
        1.0,
    )
    if count >= 6 and weight_sum > 0.0:
        regularization = wp.max(weight_sum, 1.0) * radius * radius * 1.0e-6
        normal_regularization = wp.max(0.5 * moment_trace, regularization)
        corrected_moment = (
            moment
            + wp.identity(n=3, dtype=float) * regularization
            + wp.outer(ni, ni) * normal_regularization
        )
        normal_gradient = normal_moment * wp.inverse(corrected_moment)
        tangent_gradient = normal_gradient * tangent_projector
        value = -(
            tangent_gradient[0, 0]
            + tangent_gradient[1, 1]
            + tangent_gradient[2, 2]
        )
    curvature[i] = value
    curvature_quality[i] = quality


class WarpFeatureComputer:
    """Reusable GPU buffers and hash grid for one fixed-size particle cache."""

    def __init__(
        self,
        particle_count: int,
        radius: float,
        device: str = "cuda:0",
        grid_dimension: int = 128,
    ):
        self.particle_count = int(particle_count)
        self.radius = float(radius)
        self.device = wp.get_device(device)
        self.grid = wp.HashGrid(
            grid_dimension,
            grid_dimension,
            grid_dimension,
            device=self.device,
        )
        self.neighbor_count = wp.empty(
            self.particle_count, dtype=wp.int32, device=self.device
        )
        self.kernel_sum = wp.empty(
            self.particle_count, dtype=float, device=self.device
        )
        self.centroid_offset = wp.empty(
            self.particle_count, dtype=wp.vec3, device=self.device
        )
        self.surface_normal = wp.empty(
            self.particle_count, dtype=wp.vec3, device=self.device
        )
        self.gradient_quality = wp.empty(
            self.particle_count, dtype=float, device=self.device
        )
        self.velocity_divergence = wp.empty(
            self.particle_count, dtype=float, device=self.device
        )
        self.vorticity = wp.empty(
            self.particle_count, dtype=wp.vec3, device=self.device
        )
        self.strain_rate = wp.empty(
            self.particle_count, dtype=float, device=self.device
        )
        self.velocity_dispersion = wp.empty(
            self.particle_count, dtype=float, device=self.device
        )
        self.curvature = wp.empty(
            self.particle_count, dtype=float, device=self.device
        )
        self.curvature_quality = wp.empty(
            self.particle_count, dtype=float, device=self.device
        )
        self.positions = None

    def compute_local(self, positions: np.ndarray, velocities: np.ndarray):
        if positions.shape != (self.particle_count, 3):
            raise ValueError(f"Unexpected positions shape {positions.shape}")
        if velocities.shape != (self.particle_count, 3):
            raise ValueError(f"Unexpected velocities shape {velocities.shape}")
        self.positions = wp.array(
            np.ascontiguousarray(positions, dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        velocity_array = wp.array(
            np.ascontiguousarray(velocities, dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        self.grid.build(self.positions, self.radius)
        wp.launch(
            compute_local_features,
            dim=self.particle_count,
            inputs=[
                self.grid.id,
                self.positions,
                velocity_array,
                self.radius,
                self.neighbor_count,
                self.kernel_sum,
                self.centroid_offset,
                self.surface_normal,
                self.gradient_quality,
                self.velocity_divergence,
                self.vorticity,
                self.strain_rate,
                self.velocity_dispersion,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return {
            "neighbor_count": self.neighbor_count.numpy(),
            "kernel_sum": self.kernel_sum.numpy(),
            "centroid_offset": self.centroid_offset.numpy(),
            "surface_normal": self.surface_normal.numpy(),
            "gradient_quality": self.gradient_quality.numpy(),
            "velocity_divergence": self.velocity_divergence.numpy(),
            "vorticity": self.vorticity.numpy(),
            "strain_rate": self.strain_rate.numpy(),
            "velocity_dispersion": self.velocity_dispersion.numpy(),
        }

    def compute_curvature(
        self, normals: np.ndarray, surface_confidence: np.ndarray
    ):
        if self.positions is None:
            raise RuntimeError("compute_local must run before compute_curvature")
        normal_array = wp.array(
            np.ascontiguousarray(normals, dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        confidence_array = wp.array(
            np.ascontiguousarray(surface_confidence, dtype=np.float32),
            dtype=float,
            device=self.device,
        )
        wp.launch(
            compute_surface_curvature,
            dim=self.particle_count,
            inputs=[
                self.grid.id,
                self.positions,
                normal_array,
                confidence_array,
                self.radius,
                self.curvature,
                self.curvature_quality,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        return {
            "curvature": self.curvature.numpy(),
            "curvature_quality": self.curvature_quality.numpy(),
        }
