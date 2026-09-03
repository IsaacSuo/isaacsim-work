"""CUDA neighborhood queries matching SPlisHSPlasH FoamGenerator 2.18.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


FOAM = 0
SPRAY = 1
BUBBLES = 2


@wp.kernel
def _query_neighborhood(
    grid: wp.uint64,
    primary_positions: wp.array(dtype=wp.vec3),
    primary_velocities: wp.array(dtype=wp.vec3),
    secondary_positions: wp.array(dtype=wp.vec3),
    support_radius: float,
    neighbor_counts: wp.array(dtype=wp.int32),
    particle_types: wp.array(dtype=wp.int32),
    fluid_velocities: wp.array(dtype=wp.vec3),
    weight_sums: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    position = secondary_positions[index]
    radius_squared = support_radius * support_radius
    count = int(0)
    velocity_sum = wp.vec3(0.0, 0.0, 0.0)
    weight_sum = float(0.0)
    query = wp.hash_grid_query(grid, position, support_radius)
    for primary_index in query:
        delta = position - primary_positions[primary_index]
        distance_squared = wp.dot(delta, delta)
        if distance_squared <= radius_squared:
            count += 1
            q = wp.sqrt(distance_squared) / support_radius
            weight = float(0.0)
            if q <= 0.5:
                q_squared = q * q
                weight = 6.0 * q_squared * q - 6.0 * q_squared + 1.0
            elif q <= 1.0:
                one_minus_q = 1.0 - q
                weight = 2.0 * one_minus_q * one_minus_q * one_minus_q
            velocity_sum += primary_velocities[primary_index] * weight
            weight_sum += weight

    particle_type = int(FOAM)
    if count < 6:
        particle_type = SPRAY
    elif count > 20:
        particle_type = BUBBLES
    neighbor_counts[index] = count
    particle_types[index] = particle_type
    weight_sums[index] = weight_sum
    if weight_sum > 0.0:
        fluid_velocities[index] = velocity_sum / weight_sum
    else:
        fluid_velocities[index] = wp.vec3(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class NeighborhoodResult:
    particle_types: np.ndarray
    neighbor_counts: np.ndarray
    fluid_velocities: np.ndarray
    weight_sums: np.ndarray


class FoamGeneratorWarpNeighborhood:
    """GPU port of FoamGenerator classification and local velocity sampling."""

    def __init__(
        self,
        particle_radius: float,
        device: str = "cuda:0",
        grid_dimensions: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        if particle_radius <= 0.0:
            raise ValueError("particle_radius must be positive")
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError(f"CUDA Warp device required, got {self.device}")
        self.particle_radius = float(particle_radius)
        self.support_radius = float(np.float32(4.0 * particle_radius))
        self.grid_dimensions = tuple(int(value) for value in grid_dimensions)
        if len(self.grid_dimensions) != 3 or min(self.grid_dimensions) <= 0:
            raise ValueError("grid_dimensions must contain three positive integers")

    def query(
        self,
        primary_positions: np.ndarray,
        primary_velocities: np.ndarray,
        secondary_positions: np.ndarray,
    ) -> NeighborhoodResult:
        primary_positions = np.ascontiguousarray(primary_positions, dtype=np.float32)
        primary_velocities = np.ascontiguousarray(primary_velocities, dtype=np.float32)
        secondary_positions = np.ascontiguousarray(secondary_positions, dtype=np.float32)
        if (
            primary_positions.shape != primary_velocities.shape
            or primary_positions.ndim != 2
            or primary_positions.shape[1] != 3
        ):
            raise ValueError("Primary positions and velocities must have shape (N, 3)")
        if secondary_positions.ndim != 2 or secondary_positions.shape[1] != 3:
            raise ValueError("Secondary positions must have shape (M, 3)")
        if not (
            np.all(np.isfinite(primary_positions))
            and np.all(np.isfinite(primary_velocities))
            and np.all(np.isfinite(secondary_positions))
        ):
            raise ValueError("Neighborhood query received non-finite state")
        count = len(secondary_positions)
        if count == 0:
            return NeighborhoodResult(
                particle_types=np.empty(0, dtype=np.int32),
                neighbor_counts=np.empty(0, dtype=np.int32),
                fluid_velocities=np.empty((0, 3), dtype=np.float32),
                weight_sums=np.empty(0, dtype=np.float32),
            )

        primary_positions_wp = wp.array(
            primary_positions, dtype=wp.vec3, device=self.device
        )
        primary_velocities_wp = wp.array(
            primary_velocities, dtype=wp.vec3, device=self.device
        )
        secondary_positions_wp = wp.array(
            secondary_positions, dtype=wp.vec3, device=self.device
        )
        neighbor_counts_wp = wp.zeros(count, dtype=wp.int32, device=self.device)
        particle_types_wp = wp.zeros(count, dtype=wp.int32, device=self.device)
        fluid_velocities_wp = wp.zeros(count, dtype=wp.vec3, device=self.device)
        weight_sums_wp = wp.zeros(count, dtype=wp.float32, device=self.device)
        grid = wp.HashGrid(
            *self.grid_dimensions, device=self.device, dtype=wp.float32
        )
        grid.build(primary_positions_wp, self.support_radius)
        wp.launch(
            _query_neighborhood,
            dim=count,
            inputs=[
                grid.id,
                primary_positions_wp,
                primary_velocities_wp,
                secondary_positions_wp,
                self.support_radius,
            ],
            outputs=[
                neighbor_counts_wp,
                particle_types_wp,
                fluid_velocities_wp,
                weight_sums_wp,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        particle_types = particle_types_wp.numpy()
        neighbor_counts = neighbor_counts_wp.numpy()
        fluid_velocities = fluid_velocities_wp.numpy()
        weight_sums = weight_sums_wp.numpy()
        if np.any((particle_types < FOAM) | (particle_types > BUBBLES)):
            raise RuntimeError("Warp returned an invalid secondary particle type")
        if np.any((particle_types != SPRAY) & (weight_sums <= 0.0)):
            raise RuntimeError("Foam/bubble particle has no weighted fluid neighbors")
        if not np.all(np.isfinite(fluid_velocities)):
            raise RuntimeError("Warp returned a non-finite local fluid velocity")
        return NeighborhoodResult(
            particle_types=particle_types,
            neighbor_counts=neighbor_counts,
            fluid_velocities=fluid_velocities,
            weight_sums=weight_sums,
        )


def apply_physx_velocity_control(
    velocities: np.ndarray,
    particle_types: np.ndarray,
    fluid_velocities: np.ndarray,
    gravity: np.ndarray,
    physics_dt: float,
    source_substeps: int,
    buoyancy: float,
    drag: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare velocities so the following PhysX substep applies mature type dynamics.

    PhysX remains responsible for integration and contacts. The explicit control
    compensates PhysX gravity only where FoamGenerator's foam/bubble equations do.
    """

    velocities = np.ascontiguousarray(velocities, dtype=np.float32)
    particle_types = np.ascontiguousarray(particle_types, dtype=np.int32)
    fluid_velocities = np.ascontiguousarray(fluid_velocities, dtype=np.float32)
    gravity = np.asarray(gravity, dtype=np.float32)
    count = len(velocities)
    if (
        velocities.shape != (count, 3)
        or particle_types.shape != (count,)
        or fluid_velocities.shape != (count, 3)
        or gravity.shape != (3,)
    ):
        raise ValueError("Type-dynamics arrays are not aligned")
    if physics_dt <= 0.0 or source_substeps < 1:
        raise ValueError("Invalid PhysX step configuration")
    if buoyancy < 0.0 or not 0.0 <= drag <= 1.0:
        raise ValueError("Buoyancy must be non-negative and drag must be in [0, 1]")

    controlled = velocities.copy()
    foam_mask = particle_types == FOAM
    bubble_mask = particle_types == BUBBLES
    # FoamGenerator advects foam directly with the local fluid velocity. Subtract
    # one PhysX gravity impulse so the post-step free velocity remains the target.
    controlled[foam_mask] = (
        fluid_velocities[foam_mask] - gravity[None, :] * np.float32(physics_dt)
    )
    # FoamGenerator applies drag once at source FPS. Convert it to an equivalent
    # per-substep relaxation so two 240 Hz steps do not double a 120 Hz coefficient.
    drag_per_substep = 1.0 - (1.0 - float(drag)) ** (1.0 / source_substeps)
    if np.any(bubble_mask):
        bubble_velocity = controlled[bubble_mask]
        bubble_velocity += np.float32(drag_per_substep) * (
            fluid_velocities[bubble_mask] - bubble_velocity
        )
        # Original FoamGenerator bubble acceleration is -buoyancy*g and contains
        # no ordinary gravity term. Cancel PhysX gravity and add that buoyancy.
        bubble_velocity -= (
            np.float32((1.0 + buoyancy) * physics_dt) * gravity[None, :]
        )
        controlled[bubble_mask] = bubble_velocity
    delta = controlled - velocities
    if not np.all(np.isfinite(controlled)):
        raise RuntimeError("Type dynamics produced non-finite velocities")
    return controlled, delta
