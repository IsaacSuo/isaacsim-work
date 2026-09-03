"""Warp kernels for conservative whitewater surface-density deposition."""

from __future__ import annotations

import numpy as np

from .features_warp import wp


@wp.kernel
def deposit_surface_density(
    marker_grid: wp.uint64,
    marker_positions: wp.array(dtype=wp.vec3),
    marker_directions: wp.array(dtype=wp.vec2),
    sigma_major: wp.array(dtype=float),
    sigma_minor: wp.array(dtype=float),
    peak_optical_depth: wp.array(dtype=float),
    marker_age: wp.array(dtype=float),
    marker_radius: wp.array(dtype=float),
    marker_kind: wp.array(dtype=wp.int32),
    minimum_x: float,
    minimum_z: float,
    cell_size: float,
    z_cells: int,
    maximum_support: float,
    foam_tau: wp.array(dtype=float),
    foam_age_weighted: wp.array(dtype=float),
    foam_orientation_xx: wp.array(dtype=float),
    foam_orientation_xz: wp.array(dtype=float),
    foam_orientation_zz: wp.array(dtype=float),
    bubble_tau: wp.array(dtype=float),
    bubble_radius_weighted: wp.array(dtype=float),
):
    tid = wp.tid()
    ix = tid // z_cells
    iz = tid - ix * z_cells
    x = minimum_x + float(ix) * cell_size
    z = minimum_z + float(iz) * cell_size
    query_position = wp.vec3(x, 0.0, z)

    local_foam_tau = float(0.0)
    local_foam_age = float(0.0)
    local_xx = float(0.0)
    local_xz = float(0.0)
    local_zz = float(0.0)
    local_bubble_tau = float(0.0)
    local_bubble_radius = float(0.0)
    neighbors = wp.hash_grid_query(marker_grid, query_position, maximum_support)
    for marker_index in neighbors:
        displacement = query_position - marker_positions[marker_index]
        direction = marker_directions[marker_index]
        perpendicular = wp.vec2(-direction[1], direction[0])
        displacement_xz = wp.vec2(displacement[0], displacement[2])
        major_distance = wp.dot(displacement_xz, direction)
        minor_distance = wp.dot(displacement_xz, perpendicular)
        major_sigma = sigma_major[marker_index]
        minor_sigma = sigma_minor[marker_index]
        q2 = (
            major_distance * major_distance / (major_sigma * major_sigma)
            + minor_distance * minor_distance / (minor_sigma * minor_sigma)
        )
        if q2 <= 9.0:
            contribution = peak_optical_depth[marker_index] * wp.exp(-0.5 * q2)
            if marker_kind[marker_index] == 0:
                local_foam_tau += contribution
                local_foam_age += contribution * marker_age[marker_index]
                local_xx += contribution * direction[0] * direction[0]
                local_xz += contribution * direction[0] * direction[1]
                local_zz += contribution * direction[1] * direction[1]
            else:
                local_bubble_tau += contribution
                local_bubble_radius += contribution * marker_radius[marker_index]

    foam_tau[tid] = local_foam_tau
    foam_age_weighted[tid] = local_foam_age
    foam_orientation_xx[tid] = local_xx
    foam_orientation_xz[tid] = local_xz
    foam_orientation_zz[tid] = local_zz
    bubble_tau[tid] = local_bubble_tau
    bubble_radius_weighted[tid] = local_bubble_radius


class FoamDensityComputer:
    """Deposit anisotropic marker kernels onto a fixed world-space X/Z atlas."""

    def __init__(
        self,
        x_values,
        z_values,
        maximum_markers,
        maximum_support,
        device="cuda:0",
        grid_dimension=128,
    ):
        self.x_values = np.asarray(x_values, dtype=np.float64)
        self.z_values = np.asarray(z_values, dtype=np.float64)
        if len(self.x_values) < 2 or len(self.z_values) < 2:
            raise ValueError("Density atlas requires at least two cells per axis")
        self.cell_size = float(np.median(np.diff(self.x_values)))
        coordinate_tolerance = max(1.0e-9, abs(self.cell_size) * 1.0e-4)
        if not np.allclose(
            np.diff(self.x_values), self.cell_size, rtol=0.0, atol=coordinate_tolerance
        ):
            raise ValueError("x_values are not regularly spaced")
        if not np.allclose(
            np.diff(self.z_values), self.cell_size, rtol=0.0, atol=coordinate_tolerance
        ):
            raise ValueError("z_values do not match the atlas spacing")
        self.maximum_markers = int(maximum_markers)
        self.maximum_support = float(maximum_support)
        self.device = wp.get_device(device)
        self.grid = wp.HashGrid(
            grid_dimension,
            grid_dimension,
            grid_dimension,
            device=self.device,
        )
        self.cell_count = len(self.x_values) * len(self.z_values)
        self.outputs = {
            "foam_tau": wp.empty(self.cell_count, dtype=float, device=self.device),
            "foam_age_weighted": wp.empty(
                self.cell_count, dtype=float, device=self.device
            ),
            "foam_orientation_xx": wp.empty(
                self.cell_count, dtype=float, device=self.device
            ),
            "foam_orientation_xz": wp.empty(
                self.cell_count, dtype=float, device=self.device
            ),
            "foam_orientation_zz": wp.empty(
                self.cell_count, dtype=float, device=self.device
            ),
            "bubble_tau": wp.empty(self.cell_count, dtype=float, device=self.device),
            "bubble_radius_weighted": wp.empty(
                self.cell_count, dtype=float, device=self.device
            ),
        }

    def compute(
        self,
        positions,
        directions,
        sigma_major,
        sigma_minor,
        peak_optical_depth,
        age,
        radius,
        kind,
    ):
        positions = np.asarray(positions, dtype=np.float32)
        count = len(positions)
        if count > self.maximum_markers:
            raise ValueError(f"Marker count {count} exceeds {self.maximum_markers}")
        expected = {
            "positions": (positions, (count, 3)),
            "directions": (np.asarray(directions, dtype=np.float32), (count, 2)),
            "sigma_major": (np.asarray(sigma_major, dtype=np.float32), (count,)),
            "sigma_minor": (np.asarray(sigma_minor, dtype=np.float32), (count,)),
            "peak_optical_depth": (
                np.asarray(peak_optical_depth, dtype=np.float32),
                (count,),
            ),
            "age": (np.asarray(age, dtype=np.float32), (count,)),
            "radius": (np.asarray(radius, dtype=np.float32), (count,)),
            "kind": (np.asarray(kind, dtype=np.int32), (count,)),
        }
        for name, (array, shape) in expected.items():
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape}, expected {shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if count == 0:
            shape = (len(self.x_values), len(self.z_values))
            return {name: np.zeros(shape, dtype=np.float32) for name in self.outputs}
        major = expected["sigma_major"][0]
        minor = expected["sigma_minor"][0]
        if np.any(major <= 0.0) or np.any(minor <= 0.0):
            raise ValueError("Gaussian sigma values must be strictly positive")
        if np.any(expected["peak_optical_depth"][0] < 0.0):
            raise ValueError("Peak optical depth cannot be negative")
        direction_norm = np.linalg.norm(expected["directions"][0], axis=1)
        if not np.allclose(direction_norm, 1.0, rtol=1.0e-4, atol=1.0e-5):
            raise ValueError("Marker directions must be unit vectors")
        required_support = float(3.0 * np.max(major))
        if required_support > self.maximum_support * (1.0 + 1.0e-6):
            raise ValueError(
                f"maximum_support {self.maximum_support} is smaller than "
                f"the required 3*sigma_major {required_support}"
            )
        if not np.all(np.isin(expected["kind"][0], (0, 1))):
            raise ValueError("Marker kind must be 0 (foam) or 1 (surface bubble)")
        marker_positions = positions.copy()
        marker_positions[:, 1] = 0.0
        positions_wp = wp.array(marker_positions, dtype=wp.vec3, device=self.device)
        self.grid.build(positions_wp, self.maximum_support)
        wp.launch(
            deposit_surface_density,
            dim=self.cell_count,
            inputs=[
                self.grid.id,
                positions_wp,
                wp.array(expected["directions"][0], dtype=wp.vec2, device=self.device),
                wp.array(expected["sigma_major"][0], dtype=float, device=self.device),
                wp.array(expected["sigma_minor"][0], dtype=float, device=self.device),
                wp.array(expected["peak_optical_depth"][0], dtype=float, device=self.device),
                wp.array(expected["age"][0], dtype=float, device=self.device),
                wp.array(expected["radius"][0], dtype=float, device=self.device),
                wp.array(expected["kind"][0], dtype=wp.int32, device=self.device),
                float(self.x_values[0]),
                float(self.z_values[0]),
                self.cell_size,
                len(self.z_values),
                self.maximum_support,
                *self.outputs.values(),
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        shape = (len(self.x_values), len(self.z_values))
        return {name: array.numpy().reshape(shape) for name, array in self.outputs.items()}
