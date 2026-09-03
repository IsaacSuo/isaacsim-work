"""Conservative surface-foam parcels and burst repellents for v6 whitewater."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .emission_fields import smoothstep
from .liquid_fields import GridSpec
from .marker_solver import (
    EVENT_DTYPE,
    TerminalEventKind,
    _normalized,
    _sample_gradient_interpolated,
    _sample_interpolated,
    sample_vector_trilinear,
)
from .state_machine import splitmix64, uniform01_from_keys


FOAM_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("source_event_id", "<u8"),
        ("source_kind", "u1"),
        ("birth_time", "<f8"),
        ("age", "<f4"),
        ("position", "<f4", (3,)),
        ("velocity", "<f4", (3,)),
        ("film_area", "<f4"),
        ("support_area", "<f4"),
        ("drainage_time", "<f4"),
        ("principal_direction", "<f4", (3,)),
        ("anisotropy", "<f4"),
        ("random_key", "<u8"),
    ]
)


REPELLENT_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("source_event_id", "<u8"),
        ("birth_time", "<f8"),
        ("age", "<f4"),
        ("position", "<f4", (3,)),
        ("velocity", "<f4", (3,)),
        ("radius", "<f4"),
        ("strength", "<f4"),
        ("lifetime", "<f4"),
        ("random_key", "<u8"),
    ]
)


@dataclass(frozen=True)
class SurfaceFoamModel:
    spacing: float
    liquid_density: float = 998.2
    surface_tension: float = 0.0728
    surface_velocity_response_seconds: float = 0.030
    maximum_surface_projection_cells: float = 2.0
    surface_constraint_tolerance_cells: float = 0.25
    solid_clearance_cells: float = 0.02
    minimum_film_area_m2: float = 1.0e-11
    maximum_surface_divergence_per_second: float = 20.0
    surface_support_storage_guard: float = 1.0e-5

    def __post_init__(self):
        values = (
            self.spacing,
            self.liquid_density,
            self.surface_tension,
            self.surface_velocity_response_seconds,
            self.maximum_surface_projection_cells,
            self.surface_constraint_tolerance_cells,
            self.solid_clearance_cells,
            self.minimum_film_area_m2,
            self.maximum_surface_divergence_per_second,
            self.surface_support_storage_guard,
        )
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("Surface foam model values must be finite and positive")

    def metadata(self):
        return {
            "spacing": self.spacing,
            "liquid_density": self.liquid_density,
            "surface_tension": self.surface_tension,
            "surface_velocity_response_seconds": self.surface_velocity_response_seconds,
            "maximum_surface_projection_cells": self.maximum_surface_projection_cells,
            "surface_constraint_tolerance_cells": self.surface_constraint_tolerance_cells,
            "solid_clearance_cells": self.solid_clearance_cells,
            "minimum_film_area_m2": self.minimum_film_area_m2,
            "maximum_surface_divergence_per_second": self.maximum_surface_divergence_per_second,
            "surface_support_storage_guard": self.surface_support_storage_guard,
            "foam_state": {
                "film_area": "visible foam-film area; drains exponentially",
                "support_area": "material surface footprint; changes with surface divergence",
                "coverage": "clip(film_area/support_area, 0, 1)",
            },
            "spray_source": {
                "weber_activation": [6.0, 60.0],
                "projected_area_survival": 0.25,
                "drainage_seconds": [0.8, 3.5],
            },
            "bubble_source": {
                "surface_area_survival": 0.12,
                "drainage_seconds": [1.5, 6.0],
            },
        }


def _uniform(keys, stream):
    mask = (1 << 64) - 1
    salt = np.uint64((int(stream) * 0x9E3779B97F4A7C15) & mask)
    return uniform01_from_keys(splitmix64(np.asarray(keys, np.uint64) ^ salt))


def _source_ids(event_ids, salt):
    mask = (1 << 64) - 1
    return splitmix64(
        np.asarray(event_ids, dtype=np.uint64)
        ^ np.uint64(int(salt) & mask)
    )


def source_surface_foam(events, model: SurfaceFoamModel):
    """Convert terminal spray/bubble events into foam and repellent sources."""

    events = np.asarray(events)
    if events.dtype != EVENT_DTYPE:
        raise ValueError("Terminal event array has the wrong dtype")
    if not len(events):
        return (
            np.empty(0, dtype=FOAM_DTYPE),
            np.empty(0, dtype=REPELLENT_DTYPE),
            {
                "eligible_events": 0,
                "foam_sources": 0,
                "repellent_sources": 0,
                "rejected_low_weber": 0,
                "injected_film_area_m2": 0.0,
            },
        )
    spray = events["kind"] == np.uint8(TerminalEventKind.SPRAY_REENTRY)
    burst = events["kind"] == np.uint8(TerminalEventKind.BUBBLE_BURST)
    eligible = spray | burst
    selected = events[eligible]
    selected_spray = spray[eligible]
    selected_burst = burst[eligible]
    radius = selected["physical_radius"].astype(np.float64)
    representative = selected["representative_count"].astype(np.float64)
    impact_speed = selected["impact_speed"].astype(np.float64)
    weber = (
        model.liquid_density
        * impact_speed**2
        * (2.0 * radius)
        / model.surface_tension
    )
    activation = smoothstep(6.0, 60.0, weber)
    spray_film = (
        representative * np.pi * radius**2 * 0.25 * activation
    )
    bubble_film = representative * 4.0 * np.pi * radius**2 * 0.12
    film_area = np.where(selected_spray, spray_film, bubble_film)
    rejected_low_weber = int(np.count_nonzero(selected_spray & (activation <= 0.0)))
    source_ok = selected_burst | (selected_spray & (activation > 0.0))
    source_ok &= film_area >= model.minimum_film_area_m2
    selected = selected[source_ok]
    selected_spray = selected_spray[source_ok]
    selected_burst = selected_burst[source_ok]
    film_area = film_area[source_ok]
    radius = radius[source_ok]
    keys = selected["random_key"].astype(np.uint64)
    random_a = _uniform(keys, 101)
    random_b = _uniform(keys, 102)

    foam = np.zeros(len(selected), dtype=FOAM_DTYPE)
    foam["id"] = _source_ids(selected["event_id"], 0xF0A65A11)
    foam["source_event_id"] = selected["event_id"]
    foam["source_kind"] = selected["kind"]
    foam["birth_time"] = selected["event_time"]
    foam["position"] = selected["position"]
    foam["velocity"] = 0.0
    foam["film_area"] = film_area.astype(np.float32)
    initial_coverage = np.where(
        selected_spray,
        0.30 + 0.20 * random_a,
        0.55 + 0.25 * random_a,
    )
    foam["support_area"] = (film_area / initial_coverage).astype(np.float32)
    foam["drainage_time"] = np.where(
        selected_spray,
        0.8 * (3.5 / 0.8) ** random_b,
        1.5 * (6.0 / 1.5) ** random_b,
    ).astype(np.float32)
    event_normal = _normalized(selected["normal"])
    reference = np.tile((1.0, 0.0, 0.0), (len(foam), 1))
    parallel = np.abs(np.sum(reference * event_normal, axis=1)) > 0.90
    reference[parallel] = (0.0, 0.0, 1.0)
    principal = _normalized(
        reference - np.sum(reference * event_normal, axis=1)[:, None] * event_normal
    )
    foam["principal_direction"] = principal.astype(np.float32)
    foam["anisotropy"] = 1.0
    foam["random_key"] = keys
    if len(foam):
        foam = foam[np.argsort(foam["id"])]

    burst_events = selected[selected_burst]
    burst_radius = radius[selected_burst]
    burst_film = film_area[selected_burst]
    burst_keys = keys[selected_burst]
    repellents = np.zeros(len(burst_events), dtype=REPELLENT_DTYPE)
    repellents["id"] = _source_ids(burst_events["event_id"], 0xB057A11E)
    repellents["source_event_id"] = burst_events["event_id"]
    repellents["birth_time"] = burst_events["event_time"]
    repellents["position"] = burst_events["position"]
    repellents["radius"] = np.maximum(
        1.5 * burst_radius, 0.45 * np.sqrt(burst_film / np.pi)
    ).astype(np.float32)
    repellents["strength"] = (0.55 + 0.40 * _uniform(burst_keys, 103)).astype(
        np.float32
    )
    size = np.clip(burst_radius / 0.008, 0.0, 1.0)
    repellents["lifetime"] = (
        0.15 + 0.45 * np.sqrt(size) + 0.15 * _uniform(burst_keys, 104)
    ).astype(np.float32)
    repellents["random_key"] = burst_keys
    if len(repellents):
        repellents = repellents[np.argsort(repellents["id"])]

    return foam, repellents, {
        "eligible_events": int(np.count_nonzero(eligible)),
        "foam_sources": len(foam),
        "repellent_sources": len(repellents),
        "rejected_low_weber": rejected_low_weber,
        "injected_film_area_m2": float(film_area.sum(dtype=np.float64)),
    }


def sample_surface_divergence(fields0, fields1, positions, normal, alpha, spec):
    positions = np.asarray(positions, dtype=np.float64)
    normal = _normalized(normal)
    epsilon = 0.25 * spec.spacing
    jacobian = np.empty((len(positions), 3, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = epsilon
        first_upper = sample_vector_trilinear(
            fields0["velocity"], positions + offset, spec
        )
        first_lower = sample_vector_trilinear(
            fields0["velocity"], positions - offset, spec
        )
        derivative = (first_upper - first_lower) / (2.0 * epsilon)
        if fields1 is not None:
            second_upper = sample_vector_trilinear(
                fields1["velocity"], positions + offset, spec
            )
            second_lower = sample_vector_trilinear(
                fields1["velocity"], positions - offset, spec
            )
            second = (second_upper - second_lower) / (2.0 * epsilon)
            derivative = (1.0 - alpha) * derivative + alpha * second
        jacobian[..., :, axis] = derivative
    full_divergence = np.trace(jacobian, axis1=1, axis2=2)
    normal_strain = np.einsum("ni,nij,nj->n", normal, jacobian, normal)
    surface_divergence = full_divergence - normal_strain
    symmetric = 0.5 * (jacobian + np.swapaxes(jacobian, 1, 2))
    tangent_strain = symmetric - np.einsum(
        "ni,nj,njk->nik", normal, normal, symmetric
    )
    shear = np.sqrt(np.maximum(np.sum(tangent_strain**2, axis=(1, 2)), 0.0))
    return surface_divergence, shear


def _advance_surface_positions(records, local_dt, fields0, fields1, alpha, spec, response):
    positions = records["position"].astype(np.float64)
    old_velocity = records["velocity"].astype(np.float64)
    carrier = _sample_interpolated(
        fields0, fields1, "velocity", positions, alpha, spec, vector=True
    )
    normal = _normalized(
        _sample_interpolated(
            fields0, fields1, "normal", positions, alpha, spec, vector=True
        )
    )
    tangent = carrier - np.sum(carrier * normal, axis=1)[:, None] * normal
    exponential = np.exp(-local_dt / response)
    velocity = tangent + (old_velocity - tangent) * exponential[:, None]
    displacement = (
        tangent * local_dt[:, None]
        + (old_velocity - tangent)
        * response
        * (1.0 - exponential)[:, None]
    )
    return positions + displacement, velocity


def constrain_surface_positions(positions, fields0, fields1, spec, model):
    """Alternately satisfy liquid-surface and moving-solid constraints."""

    positions = np.asarray(positions, dtype=np.float64).copy()
    if not len(positions):
        return positions, np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=bool)
    initial_phi = _sample_interpolated(fields0, fields1, "phi", positions, 1.0, spec)
    lost = (~np.isfinite(initial_phi)) | (
        np.abs(initial_phi) > model.maximum_surface_projection_cells * spec.spacing
    )
    active = ~lost
    clearance = model.solid_clearance_cells * spec.spacing
    for _ in range(5):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        current = positions[ids]
        phi = _sample_interpolated(fields0, fields1, "phi", current, 1.0, spec)
        normal = _normalized(
            _sample_interpolated(
                fields0, fields1, "normal", current, 1.0, spec, vector=True
            )
        )
        current -= phi[:, None] * normal
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
                current,
                1.0,
                spec,
            )
            overlap = clearance - dynamic_collision
            colliding = np.isfinite(overlap) & (overlap > 1.0e-7)
            if np.any(colliding):
                gradient = _sample_gradient_interpolated(
                    fields0,
                    fields1,
                    dynamic_collision_name,
                    current[colliding],
                    1.0,
                    spec,
                )
                length = np.linalg.norm(gradient, axis=1)
                direction = _normalized(gradient)
                distance = overlap[colliding] / np.clip(length, 0.50, 2.0)
                current[colliding] += distance[:, None] * direction
        collision = _sample_interpolated(
            fields0, fields1, "collision_sdf", current, 1.0, spec
        )
        overlap = clearance - collision
        colliding = np.isfinite(overlap) & (overlap > 1.0e-7)
        if np.any(colliding):
            gradient = _sample_gradient_interpolated(
                fields0,
                fields1,
                "collision_sdf",
                current[colliding],
                1.0,
                spec,
            )
            length = np.linalg.norm(gradient, axis=1)
            direction = _normalized(gradient)
            distance = overlap[colliding] / np.clip(length, 0.50, 2.0)
            current[colliding] += distance[:, None] * direction
        positions[ids] = current

    final_phi = _sample_interpolated(fields0, fields1, "phi", positions, 1.0, spec)
    final_collision = _sample_interpolated(
        fields0, fields1, "collision_sdf", positions, 1.0, spec
    )
    if "render_surface_support" in fields0:
        final_support = _sample_interpolated(
            fields0,
            fields1,
            "render_surface_support",
            positions,
            1.0,
            spec,
        )
    else:
        final_support = np.ones(len(positions), dtype=np.float64)
    lost |= (
        ~np.isfinite(final_phi)
        | ~np.isfinite(final_collision)
        | (
            np.abs(final_phi)
            > model.surface_constraint_tolerance_cells * spec.spacing
        )
        | (final_collision < clearance - 2.0e-6)
        | (final_support < 0.5 + model.surface_support_storage_guard)
    )
    final_normal = _normalized(
        _sample_interpolated(
            fields0, fields1, "normal", positions, 1.0, spec, vector=True
        )
    )
    return positions, final_normal, lost


def advance_surface_layer(
    foam,
    repellents,
    step_start_time,
    dt,
    fields0,
    fields1,
    spec: GridSpec,
    model: SurfaceFoamModel,
):
    """Advance foam parcels/repellents and report film-area sinks."""

    foam = np.asarray(foam).copy()
    repellents = np.asarray(repellents).copy()
    if foam.dtype != FOAM_DTYPE or repellents.dtype != REPELLENT_DTYPE:
        raise ValueError("Surface layer arrays have the wrong dtype")
    if dt <= 0.0:
        raise ValueError("Surface layer dt must be positive")
    drained_area = 0.0
    topology_lost_area = 0.0
    topology_lost_foam = 0
    topology_lost_repellents = 0
    expired_foam = 0
    expired_repellents = 0
    step_end = float(step_start_time) + float(dt)
    alpha = 0.5

    if len(foam):
        local_dt = np.clip(
            step_end - np.maximum(foam["birth_time"], step_start_time), 0.0, dt
        )
        # Events can occur exactly at the interval endpoint.  They receive no
        # advection or drainage yet, but must still pass endpoint surface and
        # solid constraints before entering a snapshot.
        active = foam["birth_time"] <= step_end + 1.0e-12
        if np.any(active):
            ids = np.flatnonzero(active)
            positions, velocities = _advance_surface_positions(
                foam[ids],
                local_dt[ids],
                fields0,
                fields1,
                alpha,
                spec,
                model.surface_velocity_response_seconds,
            )
            normal = _normalized(
                _sample_interpolated(
                    fields0,
                    fields1,
                    "normal",
                    positions,
                    1.0,
                    spec,
                    vector=True,
                )
            )
            divergence, shear = sample_surface_divergence(
                fields0, fields1, positions, normal, alpha, spec
            )
            divergence = np.clip(
                divergence,
                -model.maximum_surface_divergence_per_second,
                model.maximum_surface_divergence_per_second,
            )
            old_film = foam["film_area"][ids].astype(np.float64)
            decay = np.exp(
                -local_dt[ids] / foam["drainage_time"][ids].astype(np.float64)
            )
            new_film = old_film * decay
            drained_area += float((old_film - new_film).sum(dtype=np.float64))
            foam["film_area"][ids] = new_film.astype(np.float32)
            foam["support_area"][ids] *= np.exp(
                divergence * local_dt[ids]
            ).astype(np.float32)
            foam["anisotropy"][ids] = np.clip(
                foam["anisotropy"][ids]
                * np.exp(np.clip(shear, 0.0, 20.0) * local_dt[ids]),
                1.0,
                8.0,
            ).astype(np.float32)
            speed = np.linalg.norm(velocities, axis=1)
            moving = speed > 1.0e-5
            direction = velocities - np.sum(velocities * normal, axis=1)[:, None] * normal
            direction[moving] /= np.linalg.norm(direction[moving], axis=1)[:, None]
            foam["principal_direction"][ids[moving]] = direction[moving].astype(np.float32)
            positions, final_normal, lost = constrain_surface_positions(
                positions, fields0, fields1, spec, model
            )
            velocities -= np.sum(velocities * final_normal, axis=1)[:, None] * final_normal
            direction = foam["principal_direction"][ids].astype(np.float64)
            moving = np.linalg.norm(velocities, axis=1) > 1.0e-5
            direction[moving] = velocities[moving]
            direction -= np.sum(direction * final_normal, axis=1)[:, None] * final_normal
            direction_length = np.linalg.norm(direction, axis=1)
            degenerate = direction_length < 1.0e-7
            if np.any(degenerate):
                reference = np.tile(
                    (1.0, 0.0, 0.0), (np.count_nonzero(degenerate), 1)
                )
                parallel = np.abs(final_normal[degenerate, 0]) > 0.90
                reference[parallel] = (0.0, 0.0, 1.0)
                direction[degenerate] = reference - np.sum(
                    reference * final_normal[degenerate], axis=1
                )[:, None] * final_normal[degenerate]
            direction = _normalized(direction)
            foam["principal_direction"][ids] = direction.astype(np.float32)
            foam["position"][ids] = positions.astype(np.float32)
            foam["velocity"][ids] = velocities.astype(np.float32)
            foam["age"][ids] += local_dt[ids].astype(np.float32)
            if np.any(lost):
                lost_ids = ids[lost]
                topology_lost_area += float(
                    foam["film_area"][lost_ids].sum(dtype=np.float64)
                )
                topology_lost_foam += len(lost_ids)
                keep = np.ones(len(foam), dtype=bool)
                keep[lost_ids] = False
                foam = foam[keep]
        expired = foam["film_area"] < model.minimum_film_area_m2
        if np.any(expired):
            expired_foam = int(np.count_nonzero(expired))
            drained_area += float(foam["film_area"][expired].sum(dtype=np.float64))
            foam = foam[~expired]

    if len(repellents):
        local_dt = np.clip(
            step_end - np.maximum(repellents["birth_time"], step_start_time), 0.0, dt
        )
        active = repellents["birth_time"] <= step_end + 1.0e-12
        if np.any(active):
            ids = np.flatnonzero(active)
            positions, velocities = _advance_surface_positions(
                repellents[ids],
                local_dt[ids],
                fields0,
                fields1,
                alpha,
                spec,
                model.surface_velocity_response_seconds,
            )
            normal = _normalized(
                _sample_interpolated(
                    fields0,
                    fields1,
                    "normal",
                    positions,
                    1.0,
                    spec,
                    vector=True,
                )
            )
            positions, final_normal, lost = constrain_surface_positions(
                positions, fields0, fields1, spec, model
            )
            velocities -= np.sum(velocities * final_normal, axis=1)[:, None] * final_normal
            repellents["position"][ids] = positions.astype(np.float32)
            repellents["velocity"][ids] = velocities.astype(np.float32)
            repellents["age"][ids] += local_dt[ids].astype(np.float32)
            if np.any(lost):
                lost_ids = ids[lost]
                topology_lost_repellents += len(lost_ids)
                keep = np.ones(len(repellents), dtype=bool)
                keep[lost_ids] = False
                repellents = repellents[keep]
        expired = repellents["age"] >= repellents["lifetime"]
        expired_repellents = int(np.count_nonzero(expired))
        repellents = repellents[~expired]

    if len(foam):
        foam = foam[np.argsort(foam["id"])]
    if len(repellents):
        repellents = repellents[np.argsort(repellents["id"])]
    return foam, repellents, {
        "drained_film_area_m2": drained_area,
        "topology_lost_film_area_m2": topology_lost_area,
        "topology_lost_foam": topology_lost_foam,
        "topology_lost_repellents": topology_lost_repellents,
        "expired_foam": expired_foam,
        "expired_repellents": expired_repellents,
    }
