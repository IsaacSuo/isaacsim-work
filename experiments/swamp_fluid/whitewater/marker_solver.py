"""Reference dynamics for persistent v6 spray and bubble markers.

The solver consumes immutable birth events and time-interpolated liquid fields.
It preserves marker identity and represented phase volume across state changes,
and emits explicit terminal events for the later surface-foam solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .liquid_fields import GridSpec
from .marker_birth import BIRTH_DTYPE, BirthChannel, PhaseKind, sample_scalar_trilinear
from .state_machine import WhitewaterState, bubble_shape, bubble_terminal_velocity


class TerminalEventKind(IntEnum):
    SPRAY_REENTRY = 1
    BUBBLE_BURST = 2
    DOMAIN_ESCAPE = 3


class NonterminalTransitionKind(IntEnum):
    SURFACE_SUPPORT_LOSS_REENTRAINMENT = 1


SOLVER_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("state", "u1"),
        ("channel", "u1"),
        ("phase", "u1"),
        ("source_node_id", "<u4"),
        ("source_emission_index", "<u4"),
        ("source_sample", "<i4"),
        ("birth_time", "<f8"),
        ("state_age", "<f4"),
        ("position", "<f4", (3,)),
        ("velocity", "<f4", (3,)),
        ("physical_radius", "<f4"),
        ("representative_count", "<f4"),
        ("phase_volume", "<f4"),
        ("shape", "<f4", (3,)),
        ("random_key", "<u8"),
    ]
)


EVENT_DTYPE = np.dtype(
    [
        ("event_id", "<u8"),
        ("marker_id", "<u8"),
        ("kind", "u1"),
        ("channel", "u1"),
        ("phase", "u1"),
        ("event_time", "<f8"),
        ("position", "<f4", (3,)),
        ("normal", "<f4", (3,)),
        ("phase_volume", "<f4"),
        ("physical_radius", "<f4"),
        ("representative_count", "<f4"),
        ("random_key", "<u8"),
        ("impact_speed", "<f4"),
    ]
)


SUPPORT_LOSS_DTYPE = np.dtype(
    [
        ("marker_id", "<u8"),
        ("kind", "u1"),
        ("event_time", "<f8"),
        ("position", "<f4", (3,)),
        ("phase_volume", "<f4"),
        ("sampled_support", "<f4"),
    ]
)


@dataclass(frozen=True)
class MarkerSolverModel:
    spacing: float
    gravity: float = 9.81
    maximum_cfl: float = 0.40
    maximum_substeps: int = 16
    maximum_solid_projection_iterations: int = 24
    solid_clearance_cells: float = 0.04
    solid_projection_tolerance_m: float = 1.0e-7
    solid_projection_guard_m: float = 8.0e-6
    spray_reentry_cells: float = 0.12
    maximum_surface_projection_cells: float = 2.0
    surface_constraint_tolerance_cells: float = 0.25
    secondary_domain_padding_cells: float = 0.0

    def __post_init__(self):
        values = (
            self.spacing,
            self.gravity,
            self.maximum_cfl,
            self.solid_clearance_cells,
            self.solid_projection_tolerance_m,
            self.solid_projection_guard_m,
            self.spray_reentry_cells,
            self.maximum_surface_projection_cells,
            self.surface_constraint_tolerance_cells,
        )
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("Marker solver scales must be finite and positive")
        if not 1 <= int(self.maximum_substeps) <= 64:
            raise ValueError("maximum_substeps must lie within 1..64")
        if not 1 <= int(self.maximum_solid_projection_iterations) <= 64:
            raise ValueError(
                "maximum_solid_projection_iterations must lie within 1..64"
            )
        if (
            not np.isfinite(self.secondary_domain_padding_cells)
            or self.secondary_domain_padding_cells < 0.0
        ):
            raise ValueError("secondary_domain_padding_cells must be finite and non-negative")

    def metadata(self):
        return {
            "spacing": self.spacing,
            "gravity": self.gravity,
            "maximum_cfl": self.maximum_cfl,
            "maximum_substeps": self.maximum_substeps,
            "maximum_solid_projection_iterations": self.maximum_solid_projection_iterations,
            "solid_clearance_cells": self.solid_clearance_cells,
            "solid_projection_tolerance_m": self.solid_projection_tolerance_m,
            "solid_projection_guard_m": self.solid_projection_guard_m,
            "spray_reentry_cells": self.spray_reentry_cells,
            "maximum_surface_projection_cells": self.maximum_surface_projection_cells,
            "surface_constraint_tolerance_cells": self.surface_constraint_tolerance_cells,
            "secondary_domain_padding_cells": self.secondary_domain_padding_cells,
            "secondary_domain_policy": (
                "spray outside the carrier grid advances ballistically with aerodynamic "
                "drag until the padded secondary envelope; an optional sparse scene "
                "query supplies authored open-terrain and collider contact outside the "
                "audited liquid grid"
            ),
            "spray_aerodynamic_drag": "Schiller-Naumann",
            "bubble_carrier_coupling": "exponential relaxation to liquid plus terminal rise",
            "surface_bubble": "surface-constrained tangential advection then deterministic burst",
        }


def births_to_solver_markers(births):
    births = np.asarray(births)
    if births.dtype != BIRTH_DTYPE:
        raise ValueError("Birth array has the wrong dtype")
    markers = np.zeros(len(births), dtype=SOLVER_DTYPE)
    for name in (
        "id",
        "channel",
        "phase",
        "source_node_id",
        "source_emission_index",
        "source_sample",
        "birth_time",
        "position",
        "velocity",
        "physical_radius",
        "representative_count",
        "phase_volume",
        "random_key",
    ):
        markers[name] = births[name]
    spray = births["channel"] == np.uint8(BirthChannel.SPRAY)
    markers["state"][spray] = np.uint8(WhitewaterState.SPRAY)
    markers["state"][~spray] = np.uint8(WhitewaterState.ENTRAINED_BUBBLE)
    markers["shape"] = 1.0
    if np.any(~spray):
        rise = bubble_terminal_velocity(markers["physical_radius"][~spray])
        markers["shape"][~spray] = bubble_shape(
            markers["physical_radius"][~spray], rise
        )
    return markers


def sample_vector_trilinear(field, positions, spec: GridSpec, outside=0.0):
    field = np.asarray(field)
    if field.shape != spec.shape + (3,):
        raise ValueError("Vector field does not match the grid")
    return np.column_stack(
        [
            sample_scalar_trilinear(field[..., axis], positions, spec, outside)
            for axis in range(3)
        ]
    )


def sample_scalar_gradient(field, positions, spec: GridSpec):
    positions = np.asarray(positions, dtype=np.float64)
    # Stay within the local trilinear cell.  A half-voxel stencil can cross the
    # non-differentiable branch where terrain and sphere union SDFs exchange
    # ownership, producing a direction that belongs to neither collider.
    epsilon = 0.10 * spec.spacing
    gradient = np.empty((len(positions), 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = epsilon
        upper = sample_scalar_trilinear(field, positions + offset, spec)
        lower = sample_scalar_trilinear(field, positions - offset, spec)
        gradient[:, axis] = (upper - lower) / (2.0 * epsilon)
    return gradient


def _normalized(vectors, fallback=(0.0, 1.0, 0.0)):
    vectors = np.asarray(vectors, dtype=np.float64).copy()
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1.0e-7)
    vectors[valid] /= lengths[valid, None]
    vectors[~valid] = fallback
    return vectors


def _sample_interpolated(fields0, fields1, name, positions, alpha, spec, vector=False):
    sampler = sample_vector_trilinear if vector else sample_scalar_trilinear
    first = sampler(fields0[name], positions, spec)
    if fields1 is None:
        return first
    second = sampler(fields1[name], positions, spec)
    return (1.0 - alpha) * first + alpha * second


def _sample_gradient_interpolated(fields0, fields1, name, positions, alpha, spec):
    first = sample_scalar_gradient(fields0[name], positions, spec)
    if fields1 is None:
        return first
    second = sample_scalar_gradient(fields1[name], positions, spec)
    return (1.0 - alpha) * first + alpha * second


def _spray_acceleration(velocity, radius, gravity):
    """Gravity plus still-air aerodynamic drag on spherical water droplets."""

    velocity = np.asarray(velocity, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=1)
    air_density = 1.204
    air_viscosity = 1.825e-5
    liquid_density = 998.2
    reynolds = np.maximum(air_density * speed * (2.0 * radius) / air_viscosity, 1.0e-8)
    drag_coefficient = np.where(
        reynolds < 1000.0,
        24.0 / reynolds * (1.0 + 0.15 * reynolds**0.687),
        0.44,
    )
    coefficient = 3.0 * air_density * drag_coefficient / (8.0 * liquid_density * radius)
    acceleration = -coefficient[:, None] * speed[:, None] * velocity
    acceleration[:, 1] -= gravity
    return acceleration


def surface_bubble_lifetime(radius, random_key):
    radius = np.asarray(radius, dtype=np.float64)
    key = np.asarray(random_key, dtype=np.uint64)
    jitter = ((key >> np.uint64(11)).astype(np.float64)) / float(1 << 53)
    size = np.clip((radius - 0.00012) / (0.008 - 0.00012), 0.0, 1.0)
    return np.clip(0.30 - 0.18 * np.sqrt(size) + 0.08 * jitter, 0.06, 0.38)


def _event_records(markers, selection, kind, event_time, normal, impact_speed):
    ids = np.flatnonzero(selection)
    events = np.zeros(len(ids), dtype=EVENT_DTYPE)
    if not len(ids):
        return events
    marker_ids = markers["id"][ids]
    mask = (1 << 64) - 1
    event_salt = np.uint64((int(kind) * 0x9E3779B97F4A7C15) & mask)
    events["event_id"] = marker_ids ^ event_salt
    events["marker_id"] = marker_ids
    events["kind"] = np.uint8(kind)
    events["channel"] = markers["channel"][ids]
    events["phase"] = markers["phase"][ids]
    events["event_time"] = float(event_time)
    events["position"] = markers["position"][ids]
    events["normal"] = np.asarray(normal, dtype=np.float32)[ids]
    events["phase_volume"] = markers["phase_volume"][ids]
    events["physical_radius"] = markers["physical_radius"][ids]
    events["representative_count"] = markers["representative_count"][ids]
    events["random_key"] = markers["random_key"][ids]
    events["impact_speed"] = np.asarray(impact_speed, dtype=np.float32)[ids]
    return events


def advance_marker_interval(
    markers,
    step_start_time,
    dt,
    fields0,
    fields1,
    spec: GridSpec,
    model: MarkerSolverModel,
    secondary_collision_query=None,
):
    """Return survivors, terminal events, support losses, and step metrics."""

    markers = np.asarray(markers).copy()
    if markers.dtype != SOLVER_DTYPE:
        raise ValueError("Solver marker array has the wrong dtype")
    step_start_time = float(step_start_time)
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("Solver interval dt must be positive")
    if not len(markers):
        return (
            markers,
            np.empty(0, dtype=EVENT_DTYPE),
            np.empty(0, dtype=SUPPORT_LOSS_DTYPE),
            {
                "substeps": 1,
                "maximum_cfl": 0.0,
                "solid_projection_count": 0,
                "surface_transition_count": 0,
                "surface_transition_support_rejection_count": 0,
                "surface_support_loss_count": 0,
                "secondary_collision_query_count": 0,
                "secondary_solid_projection_count": 0,
                "secondary_open_boundary_invalid_count": 0,
                "secondary_to_carrier_reconciliation_count": 0,
            },
        )
    maximum_speed = float(np.linalg.norm(markers["velocity"], axis=1).max())
    # Gravity can increase spray speed during the interval. Include that exact
    # acceleration bound rather than sizing substeps from the initial velocity
    # alone and discovering a CFL violation after integration.
    bounded_speed = maximum_speed + model.gravity * dt
    requested = int(
        np.ceil(bounded_speed * dt / (model.maximum_cfl * spec.spacing))
    )
    substeps = int(np.clip(max(1, requested), 1, model.maximum_substeps))
    sub_dt = dt / substeps
    events = []
    support_loss_records = []
    solid_projection_count = 0
    surface_transition_count = 0
    surface_transition_support_rejection_count = 0
    surface_support_loss_count = 0
    secondary_collision_query_count = 0
    secondary_solid_projection_count = 0
    secondary_open_boundary_invalid_count = 0
    secondary_to_carrier_reconciliation_count = 0
    maximum_observed_cfl = 0.0

    for substep in range(substeps):
        sub_start = step_start_time + substep * sub_dt
        sub_end = sub_start + sub_dt
        alive = markers["state"] != np.uint8(WhitewaterState.DEAD)
        active_dt = np.clip(
            sub_end - np.maximum(markers["birth_time"], sub_start), 0.0, sub_dt
        )
        active = alive & (active_dt > 0.0)
        if not np.any(active):
            continue
        active_ids = np.flatnonzero(active)
        positions = markers["position"][active_ids].astype(np.float64)
        velocities = markers["velocity"][active_ids].astype(np.float64)
        old_velocities = velocities.copy()
        local_dt = active_dt[active_ids].astype(np.float64)
        dynamic_alpha = np.clip(
            (sub_start + 0.5 * sub_dt - step_start_time) / dt, 0.0, 1.0
        )
        constraint_alpha = np.clip((sub_end - step_start_time) / dt, 0.0, 1.0)
        carrier = _sample_interpolated(
            fields0, fields1, "velocity", positions, dynamic_alpha, spec, vector=True
        )
        normal = _normalized(
            _sample_interpolated(
                fields0,
                fields1,
                "normal",
                positions,
                dynamic_alpha,
                spec,
                vector=True,
            )
        )
        state = markers["state"][active_ids]
        spray = state == np.uint8(WhitewaterState.SPRAY)
        bubbles = state == np.uint8(WhitewaterState.ENTRAINED_BUBBLE)
        surface_bubbles = state == np.uint8(WhitewaterState.SURFACE_BUBBLE)
        displacement = np.empty_like(velocities)

        if np.any(spray):
            radius = markers["physical_radius"][active_ids[spray]].astype(np.float64)
            acceleration = _spray_acceleration(
                velocities[spray], radius, model.gravity
            )
            velocities[spray] += acceleration * local_dt[spray, None]
        if np.any(bubbles):
            bubble_ids = active_ids[bubbles]
            radius = markers["physical_radius"][bubble_ids].astype(np.float64)
            rise = bubble_terminal_velocity(radius)
            target = carrier[bubbles].copy()
            target[:, 1] += rise
            response = np.clip(0.012 + 5.0 * radius, 0.012, 0.060)
            exponential = np.exp(-local_dt[bubbles] / response)
            velocities[bubbles] = target + (
                old_velocities[bubbles] - target
            ) * exponential[:, None]
            displacement[bubbles] = (
                target * local_dt[bubbles, None]
                + (old_velocities[bubbles] - target)
                * response[:, None]
                * (1.0 - exponential)[:, None]
            )
            markers["shape"][bubble_ids] = bubble_shape(radius, rise)
        if np.any(surface_bubbles):
            tangent_carrier = carrier[surface_bubbles] - np.sum(
                carrier[surface_bubbles] * normal[surface_bubbles], axis=1
            )[:, None] * normal[surface_bubbles]
            exponential = np.exp(-local_dt[surface_bubbles] / 0.025)
            velocities[surface_bubbles] = tangent_carrier + (
                old_velocities[surface_bubbles] - tangent_carrier
            ) * exponential[:, None]
            displacement[surface_bubbles] = (
                tangent_carrier * local_dt[surface_bubbles, None]
                + (old_velocities[surface_bubbles] - tangent_carrier)
                * 0.025
                * (1.0 - exponential)[:, None]
            )

        old_positions = positions.copy()
        displacement[spray] = velocities[spray] * local_dt[spray, None]
        positions += displacement
        step_speed = np.linalg.norm(positions - old_positions, axis=1) / np.maximum(
            local_dt, 1.0e-12
        )
        maximum_observed_cfl = max(
            maximum_observed_cfl,
            float(np.max(step_speed * local_dt / spec.spacing)),
        )
        markers["position"][active_ids] = positions.astype(np.float32)
        markers["velocity"][active_ids] = velocities.astype(np.float32)
        markers["state_age"][active_ids] += local_dt.astype(np.float32)

        # Resolve solid penetration by projecting along the union-SDF gradient.
        radius = markers["physical_radius"][active_ids].astype(np.float64)
        required_clearance = (
            radius
            + model.solid_clearance_cells * spec.spacing
            + model.solid_projection_guard_m
        )
        damped = np.zeros(len(active_ids), dtype=bool)
        for _ in range(model.maximum_solid_projection_iterations):
            dynamic_collision_name = (
                "dynamic_collision_sdf"
                if "dynamic_collision_sdf" in fields0
                else "sphere_collision_sdf"
                if "sphere_collision_sdf" in fields0
                else None
            )
            if dynamic_collision_name is not None:
                dynamic_collision = _sample_interpolated(
                    fields0,
                    fields1,
                    dynamic_collision_name,
                    positions,
                    constraint_alpha,
                    spec,
                )
                dynamic_overlap = required_clearance - dynamic_collision
                dynamic_colliding = np.isfinite(dynamic_overlap) & (
                    dynamic_overlap > model.solid_projection_tolerance_m
                )
                if np.any(dynamic_colliding):
                    dynamic_indices = np.flatnonzero(dynamic_colliding)
                    dynamic_gradient = _sample_gradient_interpolated(
                        fields0,
                        fields1,
                        dynamic_collision_name,
                        positions[dynamic_colliding],
                        constraint_alpha,
                        spec,
                    )
                    dynamic_gradient_length = np.linalg.norm(
                        dynamic_gradient, axis=1
                    )
                    valid_gradient = np.isfinite(dynamic_gradient_length) & (
                        dynamic_gradient_length > 1.0e-8
                    )
                    dynamic_indices = dynamic_indices[valid_gradient]
                    dynamic_gradient = dynamic_gradient[valid_gradient]
                    dynamic_gradient_length = dynamic_gradient_length[valid_gradient]
                    if len(dynamic_indices):
                        dynamic_normal = _normalized(dynamic_gradient)
                        dynamic_distance = dynamic_overlap[dynamic_colliding][
                            valid_gradient
                        ] / np.clip(dynamic_gradient_length, 0.50, 2.0)
                        positions[dynamic_indices] += (
                            dynamic_distance[:, None] * dynamic_normal
                        )
                        velocity_collision = velocities[dynamic_indices]
                        inward = np.sum(velocity_collision * dynamic_normal, axis=1)
                        inward_selection = inward < 0.0
                        velocity_collision[inward_selection] -= inward[
                            inward_selection, None
                        ] * dynamic_normal[inward_selection]
                        first_collision = ~damped[dynamic_indices]
                        velocity_collision[first_collision] *= 0.85
                        damped[dynamic_indices] = True
                        velocities[dynamic_indices] = velocity_collision
                        solid_projection_count += int(len(dynamic_indices))
            collision = _sample_interpolated(
                fields0,
                fields1,
                "collision_sdf",
                positions,
                constraint_alpha,
                spec,
            )
            overlap = required_clearance - collision
            colliding = np.isfinite(overlap) & (
                overlap > model.solid_projection_tolerance_m
            )
            if not np.any(colliding):
                break
            collision_gradient = _sample_gradient_interpolated(
                fields0,
                fields1,
                "collision_sdf",
                positions[colliding],
                constraint_alpha,
                spec,
            )
            collision_gradient_length = np.linalg.norm(collision_gradient, axis=1)
            valid_gradient = np.isfinite(collision_gradient_length) & (
                collision_gradient_length > 1.0e-8
            )
            colliding_indices = np.flatnonzero(colliding)[valid_gradient]
            if not len(colliding_indices):
                break
            collision_gradient = collision_gradient[valid_gradient]
            collision_gradient_length = collision_gradient_length[valid_gradient]
            collision_normal = _normalized(collision_gradient)
            collision_distance = overlap[colliding][valid_gradient] / np.clip(
                collision_gradient_length, 0.50, 2.0
            )
            positions[colliding_indices] += (
                collision_distance[:, None] * collision_normal
            )
            velocity_collision = velocities[colliding_indices]
            inward = np.sum(velocity_collision * collision_normal, axis=1)
            inward_selection = inward < 0.0
            velocity_collision[inward_selection] -= inward[inward_selection, None] * collision_normal[
                inward_selection
            ]
            first_collision = ~damped[colliding_indices]
            velocity_collision[first_collision] *= 0.85
            damped[colliding_indices] = True
            velocities[colliding_indices] = velocity_collision
            solid_projection_count += int(len(colliding_indices))

        # The dense carrier grid is intentionally finite. Sparse spray outside
        # it still collides with authored open terrain and scene colliders via
        # exact point queries; invalid open edges remain open and never become
        # extrapolated walls.
        if secondary_collision_query is not None:
            carrier_minimum = np.asarray(spec.origin, dtype=np.float64)
            carrier_maximum = np.asarray(spec.maximum, dtype=np.float64)
            outside_carrier = spray & (
                np.any(positions < carrier_minimum, axis=1)
                | np.any(positions > carrier_maximum, axis=1)
            )
            gradient_guard = 0.11 * spec.spacing
            sparse_region = spray & (
                np.any(positions < carrier_minimum + gradient_guard, axis=1)
                | np.any(positions > carrier_maximum - gradient_guard, axis=1)
            )
            for _ in range(model.maximum_solid_projection_iterations):
                outside_indices = np.flatnonzero(sparse_region)
                if not len(outside_indices):
                    break
                query = secondary_collision_query(
                    positions[outside_indices], constraint_alpha
                )
                query_distance = np.asarray(query["distance"], dtype=np.float64)
                query_normal = np.asarray(query["normal"], dtype=np.float64)
                query_valid = np.asarray(query["valid"], dtype=bool)
                boundary_invalid = np.asarray(
                    query.get(
                        "open_boundary_invalid",
                        np.zeros(len(outside_indices), dtype=bool),
                    ),
                    dtype=bool,
                )
                if (
                    query_distance.shape != (len(outside_indices),)
                    or query_normal.shape != (len(outside_indices), 3)
                    or query_valid.shape != (len(outside_indices),)
                ):
                    raise ValueError("Secondary collision query returned wrong shapes")
                secondary_collision_query_count += len(outside_indices)
                secondary_open_boundary_invalid_count += int(
                    np.count_nonzero(boundary_invalid)
                )
                overlap = required_clearance[outside_indices] - query_distance
                colliding = query_valid & np.isfinite(overlap) & (
                    overlap > model.solid_projection_tolerance_m
                )
                if not np.any(colliding):
                    break
                local_ids = np.flatnonzero(colliding)
                global_ids = outside_indices[local_ids]
                normals = _normalized(query_normal[local_ids])
                positions[global_ids] += overlap[local_ids, None] * normals
                velocity_collision = velocities[global_ids]
                inward = np.sum(velocity_collision * normals, axis=1)
                inward_selection = inward < 0.0
                velocity_collision[inward_selection] -= inward[
                    inward_selection, None
                ] * normals[inward_selection]
                first_collision = ~damped[global_ids]
                velocity_collision[first_collision] *= 0.85
                damped[global_ids] = True
                velocities[global_ids] = velocity_collision
                solid_projection_count += len(global_ids)
                secondary_solid_projection_count += len(global_ids)

            # A sparse terrain projection can legitimately move a marker back
            # across the finite carrier-grid boundary. Reconcile that endpoint
            # against the dense union SDF before accepting it; otherwise the
            # last substep could end between the two constraint systems.
            inside_after_sparse = np.all(
                (positions >= carrier_minimum) & (positions <= carrier_maximum),
                axis=1,
            )
            reconcile = outside_carrier & inside_after_sparse
            for _ in range(model.maximum_solid_projection_iterations):
                reconcile_ids = np.flatnonzero(reconcile)
                if not len(reconcile_ids):
                    break
                collision = _sample_interpolated(
                    fields0,
                    fields1,
                    "collision_sdf",
                    positions[reconcile_ids],
                    constraint_alpha,
                    spec,
                )
                overlap = required_clearance[reconcile_ids] - collision
                colliding = np.isfinite(overlap) & (
                    overlap > model.solid_projection_tolerance_m
                )
                if not np.any(colliding):
                    break
                local_ids = np.flatnonzero(colliding)
                global_ids = reconcile_ids[local_ids]
                gradient = _sample_gradient_interpolated(
                    fields0,
                    fields1,
                    "collision_sdf",
                    positions[global_ids],
                    constraint_alpha,
                    spec,
                )
                gradient_length = np.linalg.norm(gradient, axis=1)
                valid_gradient = np.isfinite(gradient_length) & (
                    gradient_length > 1.0e-8
                )
                global_ids = global_ids[valid_gradient]
                if not len(global_ids):
                    break
                normals = _normalized(gradient[valid_gradient])
                distance = overlap[local_ids][valid_gradient] / np.clip(
                    gradient_length[valid_gradient], 0.50, 2.0
                )
                positions[global_ids] += distance[:, None] * normals
                velocity_collision = velocities[global_ids]
                inward = np.sum(velocity_collision * normals, axis=1)
                inward_selection = inward < 0.0
                velocity_collision[inward_selection] -= inward[
                    inward_selection, None
                ] * normals[inward_selection]
                velocities[global_ids] = velocity_collision
                solid_projection_count += len(global_ids)
                secondary_to_carrier_reconciliation_count += len(global_ids)
        markers["position"][active_ids] = positions.astype(np.float32)
        markers["velocity"][active_ids] = velocities.astype(np.float32)

        phi = _sample_interpolated(
            fields0, fields1, "phi", positions, constraint_alpha, spec
        )
        normal = _normalized(
            _sample_interpolated(
                fields0,
                fields1,
                "normal",
                positions,
                constraint_alpha,
                spec,
                vector=True,
            )
        )
        speed_normal = np.abs(np.sum(velocities * normal, axis=1))

        spray_reentry_local = spray & (
            phi <= -np.maximum(
                model.spray_reentry_cells * spec.spacing,
                0.25 * markers["physical_radius"][active_ids],
            )
        )
        if np.any(spray_reentry_local):
            global_selection = np.zeros(len(markers), dtype=bool)
            ids = active_ids[spray_reentry_local]
            global_selection[ids] = True
            markers["position"][ids] = (
                positions[spray_reentry_local]
                - phi[spray_reentry_local, None] * normal[spray_reentry_local]
            ).astype(np.float32)
            full_normal = np.zeros((len(markers), 3), dtype=np.float32)
            full_speed = np.zeros(len(markers), dtype=np.float32)
            full_normal[ids] = normal[spray_reentry_local]
            full_speed[ids] = speed_normal[spray_reentry_local]
            events.append(
                _event_records(
                    markers,
                    global_selection,
                    TerminalEventKind.SPRAY_REENTRY,
                    sub_end,
                    full_normal,
                    full_speed,
                )
            )
            markers["state"][ids] = np.uint8(WhitewaterState.DEAD)

        bubble_surface_candidate = bubbles & (
            phi >= -0.50 * markers["physical_radius"][active_ids]
        )
        if "render_surface_support" in fields0:
            surface_support = _sample_interpolated(
                fields0,
                fields1,
                "render_surface_support",
                positions,
                constraint_alpha,
                spec,
            ) >= 0.5
        else:
            surface_support = np.ones(len(active_ids), dtype=bool)
        bubble_surface_local = bubble_surface_candidate & surface_support
        surface_transition_support_rejection_count += int(
            np.count_nonzero(bubble_surface_candidate & ~surface_support)
        )
        if np.any(bubble_surface_local):
            ids = active_ids[bubble_surface_local]
            markers["state"][ids] = np.uint8(WhitewaterState.SURFACE_BUBBLE)
            markers["state_age"][ids] = 0.0
            markers["position"][ids] = (
                positions[bubble_surface_local]
                - phi[bubble_surface_local, None] * normal[bubble_surface_local]
                + 0.20
                * markers["physical_radius"][ids, None]
                * normal[bubble_surface_local]
            ).astype(np.float32)
            surface_transition_count += len(ids)

        # Reproject already-surface bubbles and burst them after a size-aware,
        # deterministic residence time.
        current_surface = (
            markers["state"] == np.uint8(WhitewaterState.SURFACE_BUBBLE)
        ) & active
        if np.any(current_surface) and "render_surface_support" in fields0:
            ids = np.flatnonzero(current_surface)
            sampled_support = _sample_interpolated(
                fields0,
                fields1,
                "render_surface_support",
                markers["position"][ids].astype(np.float64),
                constraint_alpha,
                spec,
            )
            current_support = sampled_support >= 0.5
            if np.any(~current_support):
                unsupported_ids = ids[~current_support]
                records = np.zeros(len(unsupported_ids), dtype=SUPPORT_LOSS_DTYPE)
                records["marker_id"] = markers["id"][unsupported_ids]
                records["kind"] = np.uint8(
                    NonterminalTransitionKind.SURFACE_SUPPORT_LOSS_REENTRAINMENT
                )
                records["event_time"] = sub_end
                records["position"] = markers["position"][unsupported_ids]
                records["phase_volume"] = markers["phase_volume"][unsupported_ids]
                records["sampled_support"] = sampled_support[~current_support]
                support_loss_records.append(records)
                markers["state"][unsupported_ids] = np.uint8(
                    WhitewaterState.ENTRAINED_BUBBLE
                )
                surface_support_loss_count += len(unsupported_ids)
                current_surface[unsupported_ids] = False
        if np.any(current_surface):
            ids = np.flatnonzero(current_surface)
            surface_position = markers["position"][ids].astype(np.float64)
            surface_radius = markers["physical_radius"][ids].astype(np.float64)
            initial_surface_phi = _sample_interpolated(
                fields0,
                fields1,
                "phi",
                surface_position,
                constraint_alpha,
                spec,
            )
            initial_surface_normal = _normalized(
                _sample_interpolated(
                    fields0,
                    fields1,
                    "normal",
                    surface_position,
                    constraint_alpha,
                    spec,
                    vector=True,
                )
            )
            topology_lost = (
                ~np.isfinite(initial_surface_phi)
                | (
                    np.abs(initial_surface_phi)
                    > model.maximum_surface_projection_cells * spec.spacing
                )
            )
            if np.any(topology_lost):
                lost_ids = ids[topology_lost]
                global_selection = np.zeros(len(markers), dtype=bool)
                global_selection[lost_ids] = True
                full_normal = np.zeros((len(markers), 3), dtype=np.float32)
                full_speed = np.zeros(len(markers), dtype=np.float32)
                full_normal[lost_ids] = initial_surface_normal[topology_lost]
                events.append(
                    _event_records(
                        markers,
                        global_selection,
                        TerminalEventKind.BUBBLE_BURST,
                        sub_end,
                        full_normal,
                        full_speed,
                    )
                )
                markers["state"][lost_ids] = np.uint8(WhitewaterState.DEAD)
                keep = ~topology_lost
                ids = ids[keep]
                surface_position = surface_position[keep]
                surface_radius = surface_radius[keep]
            # The free surface and a moving solid can intersect.  Alternating
            # endpoint projections avoids satisfying one constraint by
            # violating the other after the sphere has moved for the rest of
            # the substep.
            for _ in range(3):
                surface_phi = _sample_interpolated(
                    fields0,
                    fields1,
                    "phi",
                    surface_position,
                    constraint_alpha,
                    spec,
                )
                surface_normal = _normalized(
                    _sample_interpolated(
                        fields0,
                        fields1,
                        "normal",
                        surface_position,
                        constraint_alpha,
                        spec,
                        vector=True,
                    )
                )
                surface_position += (
                    -surface_phi + 0.20 * surface_radius
                )[:, None] * surface_normal
                surface_collision = _sample_interpolated(
                    fields0,
                    fields1,
                    "collision_sdf",
                    surface_position,
                    constraint_alpha,
                    spec,
                )
                surface_overlap = (
                    surface_radius
                    + model.solid_clearance_cells * spec.spacing
                    + model.solid_projection_guard_m
                    - surface_collision
                )
                surface_colliding = np.isfinite(surface_overlap) & (
                    surface_overlap > 0.0
                )
                if np.any(surface_colliding):
                    surface_collision_gradient = _sample_gradient_interpolated(
                        fields0,
                        fields1,
                        "collision_sdf",
                        surface_position[surface_colliding],
                        constraint_alpha,
                        spec,
                    )
                    surface_collision_gradient_length = np.linalg.norm(
                        surface_collision_gradient, axis=1
                    )
                    surface_collision_normal = _normalized(
                        surface_collision_gradient
                    )
                    surface_collision_distance = surface_overlap[
                        surface_colliding
                    ] / np.clip(surface_collision_gradient_length, 0.50, 2.0)
                    surface_position[surface_colliding] += surface_collision_distance[
                        :, None
                    ] * surface_collision_normal
                    solid_projection_count += int(
                        np.count_nonzero(surface_colliding)
                    )
            final_surface_phi = _sample_interpolated(
                fields0,
                fields1,
                "phi",
                surface_position,
                constraint_alpha,
                spec,
            )
            final_surface_collision = _sample_interpolated(
                fields0,
                fields1,
                "collision_sdf",
                surface_position,
                constraint_alpha,
                spec,
            )
            constraint_conflict = (
                ~np.isfinite(final_surface_phi)
                | ~np.isfinite(final_surface_collision)
                | (
                    np.abs(final_surface_phi)
                    > model.surface_constraint_tolerance_cells * spec.spacing
                )
                | (
                    final_surface_collision
                    < surface_radius
                    + model.solid_clearance_cells * spec.spacing
                    + model.solid_projection_guard_m
                    - 2.0e-6
                )
            )
            markers["position"][ids] = surface_position.astype(np.float32)
            if np.any(constraint_conflict):
                conflict_ids = ids[constraint_conflict]
                global_selection = np.zeros(len(markers), dtype=bool)
                global_selection[conflict_ids] = True
                full_normal = np.zeros((len(markers), 3), dtype=np.float32)
                full_speed = np.zeros(len(markers), dtype=np.float32)
                full_normal[conflict_ids] = surface_normal[constraint_conflict]
                events.append(
                    _event_records(
                        markers,
                        global_selection,
                        TerminalEventKind.BUBBLE_BURST,
                        sub_end,
                        full_normal,
                        full_speed,
                    )
                )
                markers["state"][conflict_ids] = np.uint8(WhitewaterState.DEAD)
                keep = ~constraint_conflict
                ids = ids[keep]
                surface_radius = surface_radius[keep]
                surface_normal = surface_normal[keep]
            lifetime = surface_bubble_lifetime(
                markers["physical_radius"][ids], markers["random_key"][ids]
            )
            bursting = markers["state_age"][ids] >= lifetime
            if np.any(bursting):
                burst_ids = ids[bursting]
                global_selection = np.zeros(len(markers), dtype=bool)
                global_selection[burst_ids] = True
                full_normal = np.zeros((len(markers), 3), dtype=np.float32)
                full_speed = np.zeros(len(markers), dtype=np.float32)
                full_normal[burst_ids] = surface_normal[bursting]
                events.append(
                    _event_records(
                        markers,
                        global_selection,
                        TerminalEventKind.BUBBLE_BURST,
                        sub_end,
                        full_normal,
                        full_speed,
                    )
                )
                markers["state"][burst_ids] = np.uint8(WhitewaterState.DEAD)

        positions_now = markers["position"].astype(np.float64)
        secondary_padding = model.secondary_domain_padding_cells * spec.spacing
        if secondary_padding == 0.0:
            secondary_minimum = np.asarray(spec.origin) + spec.spacing
            secondary_maximum = np.asarray(spec.maximum) - spec.spacing
        else:
            secondary_minimum = np.asarray(spec.origin) - secondary_padding
            secondary_maximum = np.asarray(spec.maximum) + secondary_padding
        escaped = (
            (markers["state"] != np.uint8(WhitewaterState.DEAD))
            & (
                np.any(positions_now < secondary_minimum, axis=1)
                | np.any(positions_now > secondary_maximum, axis=1)
            )
        )
        if np.any(escaped):
            normal_full = np.zeros((len(markers), 3), dtype=np.float32)
            speed_full = np.linalg.norm(markers["velocity"], axis=1)
            events.append(
                _event_records(
                    markers,
                    escaped,
                    TerminalEventKind.DOMAIN_ESCAPE,
                    sub_end,
                    normal_full,
                    speed_full,
                )
            )
            markers["state"][escaped] = np.uint8(WhitewaterState.DEAD)

    event_array = (
        np.concatenate(events) if events else np.empty(0, dtype=EVENT_DTYPE)
    )
    survivors = markers[markers["state"] != np.uint8(WhitewaterState.DEAD)]
    if len(survivors):
        survivors = survivors[np.argsort(survivors["id"])]
    if len(event_array):
        event_array = event_array[np.argsort(event_array["event_id"])]
    support_loss_array = (
        np.concatenate(support_loss_records)
        if support_loss_records
        else np.empty(0, dtype=SUPPORT_LOSS_DTYPE)
    )
    return survivors, event_array, support_loss_array, {
        "substeps": substeps,
        "maximum_cfl": maximum_observed_cfl,
        "solid_projection_count": solid_projection_count,
        "surface_transition_count": surface_transition_count,
        "surface_transition_support_rejection_count": (
            surface_transition_support_rejection_count
        ),
        "surface_support_loss_count": surface_support_loss_count,
        "secondary_collision_query_count": secondary_collision_query_count,
        "secondary_solid_projection_count": secondary_solid_projection_count,
        "secondary_open_boundary_invalid_count": (
            secondary_open_boundary_invalid_count
        ),
        "secondary_to_carrier_reconciliation_count": (
            secondary_to_carrier_reconciliation_count
        ),
    }
