"""Reference reconstruction of three-dimensional liquid fields.

The v6 whitewater solver consumes continuous fields rather than attaching
secondary markers to individual PhysX particles.  This module intentionally
contains a NumPy/SciPy reference implementation first: it defines the data
contract and provides deterministic validation before a GPU implementation is
allowed to replace it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


CANONICAL_COLLISION_FIELDS = (
    "collision_sdf",
    "collider_collision_sdf",
    "dynamic_collision_sdf",
    "churn_source_sdf",
    "dynamic_churn_source_sdf",
)
LEGACY_COLLISION_FIELDS = ("sphere_collision_sdf",)


def load_collision_fields(cache, destination):
    """Copy canonical collision arrays and recognized read-only legacy aliases."""

    for name in CANONICAL_COLLISION_FIELDS + LEGACY_COLLISION_FIELDS:
        if name in cache:
            destination[name] = np.asarray(cache[name])
    if "collision_sdf" not in destination:
        raise KeyError("Liquid-field cache is missing canonical collision_sdf")
    return destination


@dataclass(frozen=True)
class GridSpec:
    """A regular node-centred Cartesian grid in Isaac world coordinates."""

    origin: tuple[float, float, float]
    spacing: float
    shape: tuple[int, int, int]

    def __post_init__(self):
        origin = tuple(float(value) for value in self.origin)
        shape = tuple(int(value) for value in self.shape)
        spacing = float(self.spacing)
        if len(origin) != 3 or len(shape) != 3:
            raise ValueError("Grid origin and shape must each have three values")
        if not np.isfinite(origin).all() or not np.isfinite(spacing):
            raise ValueError("Grid coordinates must be finite")
        if spacing <= 0.0 or any(size < 3 for size in shape):
            raise ValueError("Grid spacing must be positive and every axis >= 3")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", shape)

    @property
    def cell_count(self):
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def maximum(self):
        return tuple(
            self.origin[axis] + self.spacing * (self.shape[axis] - 1)
            for axis in range(3)
        )

    def axes(self, dtype=np.float64):
        return tuple(
            np.asarray(self.origin[axis], dtype=dtype)
            + np.arange(self.shape[axis], dtype=dtype)
            * np.asarray(self.spacing, dtype=dtype)
            for axis in range(3)
        )

    def metadata(self):
        return {
            "origin": list(self.origin),
            "spacing": self.spacing,
            "shape": list(self.shape),
            "maximum": list(self.maximum),
            "centering": "node",
            "axis_order": "xyz",
        }


def aligned_grid_spec(minimum, maximum, spacing, padding=0.0):
    """Return a fixed grid whose endpoints are aligned to ``spacing``."""

    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    spacing = float(spacing)
    padding = np.broadcast_to(np.asarray(padding, dtype=np.float64), (3,))
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("Bounds must be three-vectors")
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("Bounds must be finite")
    if spacing <= 0.0 or np.any(padding < 0.0) or np.any(maximum <= minimum):
        raise ValueError("Invalid grid spacing, padding, or bounds")
    origin = np.floor((minimum - padding) / spacing) * spacing
    top = np.ceil((maximum + padding) / spacing) * spacing
    shape = np.rint((top - origin) / spacing).astype(np.int64) + 1
    return GridSpec(tuple(origin), spacing, tuple(shape))


def _validated_particle_arrays(positions, velocities, spec):
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    velocities = np.ascontiguousarray(velocities, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Unexpected particle positions shape {positions.shape}")
    if velocities.shape != positions.shape:
        raise ValueError(f"Unexpected particle velocities shape {velocities.shape}")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("Particle positions and velocities must be finite")
    coordinates = (positions - np.asarray(spec.origin)) / spec.spacing
    return positions, velocities, coordinates


def splat_particles_cic(positions, velocities, spec: GridSpec, chunk_size=250_000):
    """Conservatively splat unit particle weights and momentum with CIC."""

    _, velocities, coordinates = _validated_particle_arrays(
        positions, velocities, spec
    )
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    shape = np.asarray(spec.shape, dtype=np.int64)
    weights = np.zeros(spec.cell_count, dtype=np.float64)
    momentum = np.zeros((3, spec.cell_count), dtype=np.float64)

    for start in range(0, len(coordinates), chunk_size):
        stop = min(start + chunk_size, len(coordinates))
        xyz = coordinates[start:stop]
        velocity = velocities[start:stop]
        base = np.floor(xyz).astype(np.int64)
        fraction = xyz - base
        for ox in (0, 1):
            wx = fraction[:, 0] if ox else 1.0 - fraction[:, 0]
            ix = base[:, 0] + ox
            for oy in (0, 1):
                wy = fraction[:, 1] if oy else 1.0 - fraction[:, 1]
                iy = base[:, 1] + oy
                for oz in (0, 1):
                    wz = fraction[:, 2] if oz else 1.0 - fraction[:, 2]
                    iz = base[:, 2] + oz
                    contribution = wx * wy * wz
                    valid = (
                        (ix >= 0)
                        & (ix < shape[0])
                        & (iy >= 0)
                        & (iy < shape[1])
                        & (iz >= 0)
                        & (iz < shape[2])
                        & (contribution > 0.0)
                    )
                    if not np.any(valid):
                        continue
                    flat = (
                        (ix[valid] * shape[1] + iy[valid]) * shape[2]
                        + iz[valid]
                    )
                    corner_weight = contribution[valid]
                    weights += np.bincount(
                        flat, weights=corner_weight, minlength=spec.cell_count
                    )
                    for component in range(3):
                        momentum[component] += np.bincount(
                            flat,
                            weights=corner_weight * velocity[valid, component],
                            minlength=spec.cell_count,
                        )

    grid_shape = spec.shape
    return (
        weights.reshape(grid_shape).astype(np.float32),
        np.moveaxis(momentum.reshape((3,) + grid_shape), 0, -1).astype(
            np.float32
        ),
    )


def estimate_bulk_weight(particle_weight, quantile=0.75):
    """Estimate a stable bulk CIC weight from occupied grid nodes."""

    particle_weight = np.asarray(particle_weight, dtype=np.float64)
    positive = particle_weight[particle_weight > 1.0e-6]
    if not len(positive):
        raise ValueError("Cannot estimate bulk density from an empty splat")
    value = float(np.quantile(positive, quantile))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Estimated bulk particle weight is invalid")
    return value


def signed_distance_from_mask(fluid_mask, spacing):
    """Build a signed distance with negative values inside the liquid."""

    fluid_mask = np.asarray(fluid_mask, dtype=bool)
    if fluid_mask.ndim != 3 or not np.any(fluid_mask) or np.all(fluid_mask):
        raise ValueError("Fluid mask must be a non-empty, non-full 3D array")
    spacing = float(spacing)
    outside = distance_transform_edt(~fluid_mask, sampling=spacing)
    inside = distance_transform_edt(fluid_mask, sampling=spacing)
    phi = outside.astype(np.float64)
    phi[fluid_mask] = -inside[fluid_mask]
    # Move the zero crossing from grid nodes to the boundary between nodes.
    phi += np.where(fluid_mask, 0.5 * spacing, -0.5 * spacing)
    return phi.astype(np.float32)


def _gradient_scalar(field, spacing):
    return np.stack(
        np.gradient(field, spacing, spacing, spacing, edge_order=2), axis=-1
    )


def surface_geometry_from_phi(phi, spacing, narrow_band=None):
    """Compute outward normals and mean-curvature convention ``div(n)``."""

    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 3 or min(phi.shape) < 3 or not np.isfinite(phi).all():
        raise ValueError("phi must be a finite 3D field with every axis >= 3")
    gradient = _gradient_scalar(phi, spacing)
    magnitude = np.linalg.norm(gradient, axis=-1)
    normal = np.zeros_like(gradient)
    np.divide(
        gradient,
        magnitude[..., None],
        out=normal,
        where=magnitude[..., None] > 1.0e-8,
    )
    derivatives = [
        np.gradient(normal[..., axis], spacing, axis=axis, edge_order=2)
        for axis in range(3)
    ]
    curvature = derivatives[0] + derivatives[1] + derivatives[2]
    valid = magnitude > 0.25
    if narrow_band is not None:
        valid &= np.abs(phi) <= float(narrow_band)
    normal[~valid] = 0.0
    curvature[~valid] = 0.0
    return {
        "normal": normal.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "surface_valid": valid.astype(np.uint8),
        "gradient_magnitude": magnitude.astype(np.float32),
    }


def velocity_jacobian(velocity, spacing):
    """Return ``J[..., component, derivative_axis]``."""

    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 4 or velocity.shape[-1] != 3 or min(velocity.shape[:3]) < 3:
        raise ValueError("velocity must have shape (nx, ny, nz, 3)")
    if not np.isfinite(velocity).all():
        raise ValueError("velocity must be finite")
    jacobian = np.empty(velocity.shape[:3] + (3, 3), dtype=np.float64)
    for component in range(3):
        derivatives = np.gradient(
            velocity[..., component], spacing, spacing, spacing, edge_order=2
        )
        for axis in range(3):
            jacobian[..., component, axis] = derivatives[axis]
    return jacobian


def velocity_differentials(velocity, spacing, valid_mask=None):
    """Compute divergence, curl and deviatoric strain magnitude."""

    jacobian = velocity_jacobian(velocity, spacing)
    divergence = np.trace(jacobian, axis1=-2, axis2=-1)
    vorticity = np.empty(velocity.shape, dtype=np.float64)
    vorticity[..., 0] = jacobian[..., 2, 1] - jacobian[..., 1, 2]
    vorticity[..., 1] = jacobian[..., 0, 2] - jacobian[..., 2, 0]
    vorticity[..., 2] = jacobian[..., 1, 0] - jacobian[..., 0, 1]
    symmetric = 0.5 * (jacobian + np.swapaxes(jacobian, -1, -2))
    identity = np.eye(3, dtype=np.float64)
    deviator = symmetric - divergence[..., None, None] * identity / 3.0
    strain_rate = np.sqrt(np.sum(deviator * deviator, axis=(-2, -1)))
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != velocity.shape[:3]:
            raise ValueError("valid_mask shape does not match velocity grid")
        divergence = np.where(valid_mask, divergence, 0.0)
        vorticity = np.where(valid_mask[..., None], vorticity, 0.0)
        strain_rate = np.where(valid_mask, strain_rate, 0.0)
    return {
        "divergence": divergence.astype(np.float32),
        "vorticity": vorticity.astype(np.float32),
        "strain_rate": strain_rate.astype(np.float32),
        "jacobian": jacobian,
    }


def material_acceleration(current_velocity, previous_velocity, dt, spacing, valid_mask=None):
    """Compute ``Du/Dt = du/dt + (u dot grad)u`` on a fixed grid."""

    current = np.asarray(current_velocity, dtype=np.float64)
    previous = np.asarray(previous_velocity, dtype=np.float64)
    if current.shape != previous.shape or dt <= 0.0:
        raise ValueError("Velocity fields must match and dt must be positive")
    jacobian = velocity_jacobian(current, spacing)
    convective = np.einsum("...ca,...a->...c", jacobian, current)
    acceleration = (current - previous) / float(dt) + convective
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        acceleration = np.where(valid_mask[..., None], acceleration, 0.0)
    return acceleration.astype(np.float32)


def reconstruct_liquid_fields(
    positions,
    velocities,
    spec: GridSpec,
    *,
    bulk_weight_reference=None,
    iso_fraction=0.30,
    velocity_smoothing_sigma=0.85,
    narrow_band_cells=4.0,
):
    """Reconstruct one deterministic liquid-field snapshot from particles."""

    iso_fraction = float(iso_fraction)
    velocity_smoothing_sigma = float(velocity_smoothing_sigma)
    narrow_band_cells = float(narrow_band_cells)
    if not 0.05 <= iso_fraction <= 0.95:
        raise ValueError("iso_fraction must lie within 0.05..0.95")
    if velocity_smoothing_sigma <= 0.0 or narrow_band_cells < 1.0:
        raise ValueError("Smoothing and narrow-band widths must be positive")

    particle_weight, momentum = splat_particles_cic(positions, velocities, spec)
    if bulk_weight_reference is None:
        bulk_weight_reference = estimate_bulk_weight(particle_weight)
    bulk_weight_reference = float(bulk_weight_reference)
    if not np.isfinite(bulk_weight_reference) or bulk_weight_reference <= 0.0:
        raise ValueError("bulk_weight_reference must be finite and positive")

    number_density = particle_weight / bulk_weight_reference
    fluid_mask = number_density >= iso_fraction
    phi = signed_distance_from_mask(fluid_mask, spec.spacing)
    narrow_band = narrow_band_cells * spec.spacing

    sigma = velocity_smoothing_sigma
    smooth_weight = gaussian_filter(
        particle_weight.astype(np.float64), sigma=sigma, mode="constant"
    )
    smooth_momentum = np.empty(momentum.shape, dtype=np.float64)
    for component in range(3):
        smooth_momentum[..., component] = gaussian_filter(
            momentum[..., component].astype(np.float64),
            sigma=sigma,
            mode="constant",
        )
    velocity = np.zeros(momentum.shape, dtype=np.float64)
    velocity_valid = (
        (smooth_weight >= 0.01 * bulk_weight_reference)
        & (phi <= narrow_band)
    )
    np.divide(
        smooth_momentum,
        smooth_weight[..., None],
        out=velocity,
        where=smooth_weight[..., None] > 1.0e-12,
    )
    velocity[~velocity_valid] = 0.0

    geometry = surface_geometry_from_phi(phi, spec.spacing, narrow_band)
    differentials = velocity_differentials(
        velocity, spec.spacing, valid_mask=velocity_valid
    )
    return {
        "particle_weight": particle_weight,
        "number_density": number_density.astype(np.float32),
        "fluid_mask": fluid_mask.astype(np.uint8),
        "phi": phi,
        "depth": np.maximum(-phi, 0.0).astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "velocity_valid": velocity_valid.astype(np.uint8),
        "normal": geometry["normal"],
        "curvature": geometry["curvature"],
        "surface_valid": geometry["surface_valid"],
        "divergence": differentials["divergence"],
        "vorticity": differentials["vorticity"],
        "strain_rate": differentials["strain_rate"],
        "bulk_weight_reference": bulk_weight_reference,
        "narrow_band": narrow_band,
    }


def sample_regular_heightfield(x, z, x_values, z_values, height):
    """Bilinearly sample a regular X/Z terrain heightfield."""

    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x_values = np.asarray(x_values, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    if height.shape != (len(x_values), len(z_values)):
        raise ValueError("Terrain height shape does not match its axes")
    if len(x_values) < 2 or len(z_values) < 2:
        raise ValueError("Terrain axes must contain at least two values")
    dx = np.diff(x_values)
    dz = np.diff(z_values)
    if not np.allclose(dx, dx[0]) or not np.allclose(dz, dz[0]):
        raise ValueError("Terrain axes must be regular")
    fx = (x - x_values[0]) / dx[0]
    fz = (z - z_values[0]) / dz[0]
    ix = np.floor(fx).astype(np.int64)
    iz = np.floor(fz).astype(np.int64)
    valid_x = (fx >= 0.0) & (fx <= len(x_values) - 1)
    valid_z = (fz >= 0.0) & (fz <= len(z_values) - 1)
    ix = np.clip(ix, 0, len(x_values) - 2)
    iz = np.clip(iz, 0, len(z_values) - 2)
    # Compute fractions after clipping so an exact final-axis sample uses the
    # right endpoint of the last interval rather than its left endpoint.
    tx = np.clip(fx - ix, 0.0, 1.0)
    tz = np.clip(fz - iz, 0.0, 1.0)
    h00 = height[ix[:, None], iz[None, :]]
    h10 = height[ix[:, None] + 1, iz[None, :]]
    h01 = height[ix[:, None], iz[None, :] + 1]
    h11 = height[ix[:, None] + 1, iz[None, :] + 1]
    sampled = (
        (1.0 - tx[:, None]) * (1.0 - tz[None, :]) * h00
        + tx[:, None] * (1.0 - tz[None, :]) * h10
        + (1.0 - tx[:, None]) * tz[None, :] * h01
        + tx[:, None] * tz[None, :] * h11
    )
    valid = valid_x[:, None] & valid_z[None, :] & np.isfinite(sampled)
    return sampled.astype(np.float32), valid


def terrain_collision_sdf(spec: GridSpec, x_values, z_values, terrain_height):
    """Return positive distance above terrain and negative below it."""

    x, y, z = spec.axes()
    terrain_xz, valid_xz = sample_regular_heightfield(
        x, z, x_values, z_values, terrain_height
    )
    sdf = y[None, :, None] - terrain_xz[:, None, :]
    valid = np.broadcast_to(valid_xz[:, None, :], spec.shape)
    sdf = np.where(valid, sdf, np.inf)
    return sdf.astype(np.float32), valid.astype(np.uint8)


def sphere_collision_sdf(spec: GridSpec, center, radius):
    """Return the exact node-sampled signed distance to a sphere."""

    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    if center.shape != (3,) or not np.isfinite(center).all() or radius <= 0.0:
        raise ValueError("Invalid sphere center or radius")
    x, y, z = spec.axes()
    squared = (
        (x[:, None, None] - center[0]) ** 2
        + (y[None, :, None] - center[1]) ** 2
        + (z[None, None, :] - center[2]) ** 2
    )
    return (np.sqrt(squared) - radius).astype(np.float32)


def union_collision_sdf(*fields):
    if not fields:
        raise ValueError("At least one collision field is required")
    result = np.asarray(fields[0], dtype=np.float32).copy()
    for field in fields[1:]:
        field = np.asarray(field, dtype=np.float32)
        if field.shape != result.shape:
            raise ValueError("Collision field shapes differ")
        np.minimum(result, field, out=result)
    return result
