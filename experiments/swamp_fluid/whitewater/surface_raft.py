"""Neighbour-aware, conservative surface-bubble raft reconstruction.

This module is deliberately a derived visual dynamics layer.  It never mutates
the audited marker trajectory: every surface-bubble row is copied one-for-one,
and only ``raft_position``/diagnostic fields are added.  Pair constraints use
the physical bubble radii; representative count and gas volume are accounting
quantities and are never converted into a larger collision radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .liquid_fields import GridSpec
from .marker_solver import SOLVER_DTYPE, _normalized, _sample_interpolated
from .state_machine import WhitewaterState
from .surface_foam import constrain_surface_positions


RAFT_DTYPE = np.dtype(
    [
        ("marker_id", "<u8"),
        ("physical_radius", "<f4"),
        ("representative_count", "<f4"),
        ("phase_volume", "<f4"),
        ("shape", "<f4", (3,)),
        ("raw_anchor_position", "<f4", (3,)),
        ("raft_position", "<f4", (3,)),
        ("raft_velocity", "<f4", (3,)),
        ("state_age", "<f4"),
        ("random_key", "<u8"),
        ("neighbor_count", "<u2"),
        ("contact_count", "<u2"),
        ("maximum_contact_compression", "<f4"),
        ("packing_fraction", "<f4"),
        ("cluster_id", "<u8"),
        ("cluster_size", "<u4"),
        ("cluster_representative_count", "<f4"),
        ("anchor_displacement", "<f4"),
        ("support_value", "<f4"),
        ("support_fallback", "u1"),
        ("anchor_support_repair", "u1"),
    ]
)


@dataclass(frozen=True)
class SurfaceRaftModel:
    spacing: float
    liquid_density: float = 998.2
    surface_tension: float = 0.0728
    gravity: float = 9.81
    interaction_capillary_lengths: float = 3.0
    attraction_neighbor_limit: int = 4
    capillary_relaxation_seconds: float = 0.18
    anchor_relaxation_seconds: float = 0.75
    maximum_anchor_displacement_m: float = 0.012
    solver_iterations: int = 12
    nonpenetration_iterations: int = 32
    contact_recovery_iterations: int = 128
    pair_relaxation: float = 0.90
    nonpenetration_relaxation: float = 1.0
    maximum_contact_compression_fraction: float = 0.30
    contact_compression_guard_fraction: float = 0.001
    minimum_normal_alignment: float = 0.50
    maximum_normal_separation_m: float = 0.002
    maximum_surface_projection_cells: float = 2.0
    surface_constraint_tolerance_cells: float = 0.25
    solid_clearance_cells: float = 0.02
    surface_support_storage_guard: float = 0.0

    def __post_init__(self):
        values = (
            self.spacing,
            self.liquid_density,
            self.surface_tension,
            self.gravity,
            self.interaction_capillary_lengths,
            self.capillary_relaxation_seconds,
            self.anchor_relaxation_seconds,
            self.maximum_anchor_displacement_m,
            self.pair_relaxation,
            self.nonpenetration_relaxation,
            self.maximum_contact_compression_fraction,
            self.contact_compression_guard_fraction,
            self.maximum_normal_separation_m,
            self.maximum_surface_projection_cells,
            self.surface_constraint_tolerance_cells,
            self.solid_clearance_cells,
        )
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("Surface raft model values must be finite and positive")
        if (
            not np.isfinite(self.surface_support_storage_guard)
            or self.surface_support_storage_guard < 0.0
        ):
            raise ValueError("surface_support_storage_guard must be finite and nonnegative")
        if not 1 <= int(self.solver_iterations) <= 64:
            raise ValueError("solver_iterations must lie within 1..64")
        if not 1 <= int(self.attraction_neighbor_limit) <= 12:
            raise ValueError("attraction_neighbor_limit must lie within 1..12")
        if not 1 <= int(self.nonpenetration_iterations) <= 64:
            raise ValueError("nonpenetration_iterations must lie within 1..64")
        if not 1 <= int(self.contact_recovery_iterations) <= 128:
            raise ValueError("contact_recovery_iterations must lie within 1..128")
        if not 0.0 < self.pair_relaxation <= 1.0:
            raise ValueError("pair_relaxation must lie within (0, 1]")
        if not 0.0 < self.nonpenetration_relaxation <= 1.0:
            raise ValueError("nonpenetration_relaxation must lie within (0, 1]")
        if not 0.0 < self.maximum_contact_compression_fraction < 0.5:
            raise ValueError(
                "maximum_contact_compression_fraction must lie within (0, 0.5)"
            )
        if not 0.0 <= self.contact_compression_guard_fraction < 0.05:
            raise ValueError(
                "contact_compression_guard_fraction must lie within [0, 0.05)"
            )
        if self.contact_compression_guard_fraction >= self.maximum_contact_compression_fraction:
            raise ValueError("contact compression guard must be below the maximum")
        if not -1.0 <= self.minimum_normal_alignment <= 1.0:
            raise ValueError("minimum_normal_alignment must lie within [-1, 1]")

    @property
    def capillary_length_m(self):
        return float(
            np.sqrt(self.surface_tension / (self.liquid_density * self.gravity))
        )

    @property
    def interaction_distance_m(self):
        return self.interaction_capillary_lengths * self.capillary_length_m

    def metadata(self):
        return {
            "spacing": self.spacing,
            "liquid_density": self.liquid_density,
            "surface_tension": self.surface_tension,
            "gravity": self.gravity,
            "capillary_length_m": self.capillary_length_m,
            "interaction_capillary_lengths": self.interaction_capillary_lengths,
            "interaction_distance_m": self.interaction_distance_m,
            "attraction_neighbor_limit": self.attraction_neighbor_limit,
            "capillary_relaxation_seconds": self.capillary_relaxation_seconds,
            "anchor_relaxation_seconds": self.anchor_relaxation_seconds,
            "maximum_anchor_displacement_m": self.maximum_anchor_displacement_m,
            "solver_iterations": self.solver_iterations,
            "nonpenetration_iterations": self.nonpenetration_iterations,
            "contact_recovery_iterations": self.contact_recovery_iterations,
            "pair_relaxation": self.pair_relaxation,
            "nonpenetration_relaxation": self.nonpenetration_relaxation,
            "maximum_contact_compression_fraction": self.maximum_contact_compression_fraction,
            "contact_compression_guard_fraction": self.contact_compression_guard_fraction,
            "minimum_normal_alignment": self.minimum_normal_alignment,
            "maximum_normal_separation_m": self.maximum_normal_separation_m,
            "surface_projection": {
                "maximum_cells": self.maximum_surface_projection_cells,
                "tolerance_cells": self.surface_constraint_tolerance_cells,
                "solid_clearance_cells": self.solid_clearance_cells,
                "support_threshold": 0.5 + self.surface_support_storage_guard,
            },
            "contracts": {
                "contact_distance": "physical_radius_i + physical_radius_j",
                "contact_model": (
                    "deformable bubble: contact begins at the real-radius sum; "
                    "linear centre compression is bounded by the configured "
                    "maximum_contact_compression_fraction"
                ),
                "mobility": "inverse physical radius; smaller bubbles move more",
                "gas_fields": "bitwise copy by marker ID; no coalescence or deletion",
                "anchor_tether": "soft exponential relaxation plus hard 12 mm bound",
            },
        }


def _surface_markers(markers):
    markers = np.asarray(markers)
    if markers.dtype != SOLVER_DTYPE:
        raise ValueError("Marker array has the wrong dtype")
    selected = markers[
        markers["state"] == np.uint8(WhitewaterState.SURFACE_BUBBLE)
    ].copy()
    if len(selected):
        selected = selected[np.argsort(selected["id"])]
        if np.any(selected["id"][1:] == selected["id"][:-1]):
            raise ValueError("Surface marker IDs are not unique")
    return selected


def _previous_matches(ids, previous):
    matched = np.zeros(len(ids), dtype=bool)
    previous_index = np.zeros(len(ids), dtype=np.int64)
    if previous is None or not len(previous) or not len(ids):
        return matched, previous_index
    previous = np.asarray(previous)
    if previous.dtype != RAFT_DTYPE:
        raise ValueError("Previous raft array has the wrong dtype")
    previous_ids = previous["marker_id"]
    if np.any(previous_ids[1:] <= previous_ids[:-1]):
        raise ValueError("Previous raft rows must be strictly ID-sorted")
    candidate = np.searchsorted(previous_ids, ids)
    inside = candidate < len(previous_ids)
    matched[inside] = previous_ids[candidate[inside]] == ids[inside]
    previous_index[inside] = candidate[inside]
    return matched, previous_index


def _hard_tether(positions, anchors, maximum):
    offset = positions - anchors
    length = np.linalg.norm(offset, axis=1)
    outside = length > maximum
    if np.any(outside):
        positions[outside] = anchors[outside] + offset[outside] * (
            maximum / length[outside]
        )[:, None]
    return outside


def _position_validity(positions, fields, spec, model):
    """Validate a position without moving it (important after hard tethering)."""
    phi = _sample_interpolated(fields, None, "phi", positions, 1.0, spec)
    collision = _sample_interpolated(
        fields, None, "collision_sdf", positions, 1.0, spec
    )
    support = _sample_interpolated(
        fields, None, "render_surface_support", positions, 1.0, spec
    )
    clearance = model.solid_clearance_cells * spec.spacing
    valid = (
        np.isfinite(phi)
        & np.isfinite(collision)
        & np.isfinite(support)
        & (
            np.abs(phi)
            <= model.surface_constraint_tolerance_cells * spec.spacing
        )
        & (collision >= clearance - 2.0e-6)
        & (support >= 0.5 + model.surface_support_storage_guard)
    )
    dynamic_collision_name = (
        "dynamic_collision_sdf"
        if "dynamic_collision_sdf" in fields
        else "sphere_collision_sdf" if "sphere_collision_sdf" in fields else None
    )
    if dynamic_collision_name is not None:
        dynamic_collision = _sample_interpolated(
            fields, None, dynamic_collision_name, positions, 1.0, spec
        )
        valid &= np.isfinite(dynamic_collision) & (
            dynamic_collision >= clearance - 2.0e-6
        )
    return valid, support


def _compatible_pairs(positions, normals, radii, model):
    if len(positions) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0)
    search_distance = max(model.interaction_distance_m, 2.0 * float(radii.max()))
    pairs = cKDTree(positions).query_pairs(search_distance, output_type="ndarray")
    if not len(pairs):
        return pairs.astype(np.int64), np.empty(0), np.empty(0)
    pairs = pairs.astype(np.int64, copy=False)
    i, j = pairs[:, 0], pairs[:, 1]
    delta = positions[j] - positions[i]
    distance = np.linalg.norm(delta, axis=1)
    contact = radii[i] + radii[j]
    overlap = distance < contact
    interaction = distance <= np.maximum(contact, model.interaction_distance_m)
    interaction &= np.sum(normals[i] * normals[j], axis=1) >= model.minimum_normal_alignment
    average_normal = _normalized(normals[i] + normals[j])
    normal_separation = np.abs(np.sum(delta * average_normal, axis=1))
    interaction &= normal_separation <= np.maximum(
        model.maximum_normal_separation_m, 0.5 * contact
    )
    # Attraction is sheet-aware, but physical spheres may never interpenetrate
    # merely because their sampled surface normals disagree.
    allowed = overlap | interaction
    return pairs[allowed], distance[allowed], contact[allowed]


def _deterministic_tangent(normal, first_id, second_id):
    reference = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(normal[0])) > 0.85:
        reference[:] = (0.0, 0.0, 1.0)
    first = _normalized((reference - np.dot(reference, normal) * normal)[None, :])[0]
    second = np.cross(normal, first)
    mixed = (
        (int(first_id) * 0x9E3779B97F4A7C15)
        ^ (int(second_id) * 0xD6E8FEB86659FD93)
    ) & ((1 << 64) - 1)
    angle = 2.0 * np.pi * ((mixed >> 11) / float(1 << 53))
    return np.cos(angle) * first + np.sin(angle) * second


def _solve_pairs(positions, normals, radii, ids, dt, model, locked):
    pairs, distance, contact = _compatible_pairs(positions, normals, radii, model)
    if not len(pairs):
        return positions, 0, 0
    total_attraction = 1.0 - np.exp(-dt / model.capillary_relaxation_seconds)
    attraction = 1.0 - (1.0 - total_attraction) ** (1.0 / model.solver_iterations)
    corrections = np.zeros_like(positions)
    counts = np.zeros(len(positions), dtype=np.int32)
    attraction_candidate = np.ones(len(pairs), dtype=bool)
    incident = [[] for _ in range(len(positions))]
    for pair_index, (first, second) in enumerate(pairs):
        incident[first].append(pair_index)
        incident[second].append(pair_index)
    selected_by_node = np.zeros((len(pairs), 2), dtype=bool)
    for node, pair_indices in enumerate(incident):
        if not pair_indices:
            continue
        pair_indices = np.asarray(pair_indices, dtype=np.int64)
        chosen = pair_indices[
            np.argsort(distance[pair_indices], kind="stable")[
                : model.attraction_neighbor_limit
            ]
        ]
        first_endpoint = pairs[chosen, 0] == node
        selected_by_node[chosen[first_endpoint], 0] = True
        selected_by_node[chosen[~first_endpoint], 1] = True
    attraction_candidate = selected_by_node[:, 0] & selected_by_node[:, 1]
    overlap_count = 0
    attraction_count = 0
    for pair_index, (first, second) in enumerate(pairs):
        if locked[first] or locked[second]:
            continue
        delta = positions[second] - positions[first]
        average_normal = _normalized((normals[first] + normals[second])[None, :])[0]
        signed_normal_separation = float(np.dot(delta, average_normal))
        tangent = delta - signed_normal_separation * average_normal
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1.0e-10:
            direction = _deterministic_tangent(
                average_normal, ids[first], ids[second]
            )
        else:
            direction = tangent / tangent_length
        current_distance = float(distance[pair_index])
        contact_distance = float(contact[pair_index])
        solver_compression = (
            model.maximum_contact_compression_fraction
            - model.contact_compression_guard_fraction
        )
        minimum_distance = (1.0 - solver_compression) * contact_distance
        constraint = current_distance - minimum_distance
        if constraint < 0.0:
            strength = model.pair_relaxation
            overlap_count += 1
        elif (
            attraction_candidate[pair_index]
            and current_distance > contact_distance
            and current_distance < model.interaction_distance_m
        ):
            constraint = current_distance - contact_distance
            denominator = max(
                model.interaction_distance_m - contact_distance, 1.0e-9
            )
            kernel = np.clip(
                (model.interaction_distance_m - current_distance) / denominator,
                0.0,
                1.0,
            )
            strength = attraction * kernel * kernel
            attraction_count += 1
        else:
            continue
        # Overdamped interfacial mobility scales inversely with bubble size.
        mobility_first = 1.0 / max(float(radii[first]), 5.0e-5)
        mobility_second = 1.0 / max(float(radii[second]), 5.0e-5)
        mobility_sum = mobility_first + mobility_second
        correction = strength * constraint * direction
        corrections[first] += correction * (mobility_first / mobility_sum)
        corrections[second] -= correction * (mobility_second / mobility_sum)
        counts[first] += 1
        counts[second] += 1
    active = counts > 0
    positions[active] += corrections[active] / counts[active, None]
    return positions, overlap_count, attraction_count


def _solve_nonpenetration(positions, anchors, normals, radii, ids, model, locked):
    """One deterministic Gauss-Seidel physical-radius contact sweep."""
    pairs, _, _ = _compatible_pairs(positions, normals, radii, model)
    corrections = 0
    for first, second in pairs:
        delta = positions[second] - positions[first]
        distance = float(np.linalg.norm(delta))
        physical_contact = float(radii[first] + radii[second])
        solver_compression = (
            model.maximum_contact_compression_fraction
            - model.contact_compression_guard_fraction
        )
        contact = (1.0 - solver_compression) * physical_contact
        overlap = contact - distance
        if overlap <= 1.0e-7:
            continue
        average_normal = _normalized((normals[first] + normals[second])[None, :])[0]
        signed_normal_separation = float(np.dot(delta, average_normal))
        tangent = delta - signed_normal_separation * average_normal
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1.0e-10:
            direction = _deterministic_tangent(
                average_normal, ids[first], ids[second]
            )
        else:
            direction = tangent / tangent_length
        mobile_first = not locked[first]
        mobile_second = not locked[second]
        if not mobile_first and not mobile_second:
            continue
        mobility_first = (
            1.0 / max(float(radii[first]), 5.0e-5) if mobile_first else 0.0
        )
        mobility_second = (
            1.0 / max(float(radii[second]), 5.0e-5) if mobile_second else 0.0
        )
        mobility_sum = mobility_first + mobility_second
        # The correction is constrained to the local tangent plane.  Solving
        # by the 3-D overlap itself under-corrects whenever the two projected
        # centres have nonzero normal separation; solve the required tangent
        # leg of the contact triangle exactly instead.
        target_tangent_length = np.sqrt(
            max(contact * contact - signed_normal_separation**2, 0.0)
        )
        tangent_overlap = target_tangent_length - tangent_length
        if tangent_overlap <= 1.0e-10:
            continue
        motion_first = -direction * (mobility_first / mobility_sum)
        motion_second = direction * (mobility_second / mobility_sum)
        motion_first -= np.dot(motion_first, normals[first]) * normals[first]
        motion_second -= np.dot(motion_second, normals[second]) * normals[second]
        # At the hard tether boundary, redirect an outward contact correction
        # into a slide along that boundary instead of applying it and then
        # losing most of it to radial clipping.
        for marker, motion in ((first, motion_first), (second, motion_second)):
            offset = positions[marker] - anchors[marker]
            offset_length = float(np.linalg.norm(offset))
            if offset_length >= model.maximum_anchor_displacement_m - 2.0e-5:
                outward = offset / max(offset_length, 1.0e-12)
                outward_component = float(np.dot(motion, outward))
                if outward_component > 0.0:
                    motion -= outward_component * outward
        separation_gain = float(np.dot(motion_second - motion_first, direction))
        if separation_gain <= 1.0e-8:
            continue
        scale = (
            model.nonpenetration_relaxation
            * tangent_overlap
            / separation_gain
        )
        positions[first] += scale * motion_first
        positions[second] += scale * motion_second
        corrections += 1
    return positions, corrections


def _repair_unsupported_anchors(positions, anchors, bad, fields, spec, model):
    """Find the nearest valid top-shell point when a locked raw anchor is stale."""
    repaired = np.zeros(len(positions), dtype=bool)
    repair_distance = np.zeros(len(positions), dtype=np.float64)
    for index in np.flatnonzero(bad):
        anchor = anchors[index]
        normal = _normalized(
            _sample_interpolated(
                fields, None, "normal", anchor[None, :], 1.0, spec, vector=True
            )
        )[0]
        reference = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(normal[0])) > 0.85:
            reference[:] = (0.0, 0.0, 1.0)
        tangent_first = _normalized(
            (reference - np.dot(reference, normal) * normal)[None, :]
        )[0]
        tangent_second = np.cross(normal, tangent_first)
        radii = np.linspace(0.00025, 0.012, 48, dtype=np.float64)
        angles = np.arange(32, dtype=np.float64) * (2.0 * np.pi / 32.0)
        directions = (
            np.cos(angles)[:, None] * tangent_first
            + np.sin(angles)[:, None] * tangent_second
        )
        candidates = anchor + (
            radii[:, None, None] * directions[None, :, :]
        ).reshape(-1, 3)
        candidates, _, lost = constrain_surface_positions(
            candidates, fields, None, spec, model
        )
        _hard_tether(
            candidates,
            np.broadcast_to(anchor, candidates.shape),
            model.maximum_anchor_displacement_m,
        )
        valid, _ = _position_validity(candidates, fields, spec, model)
        valid &= ~lost
        if not np.any(valid):
            continue
        valid_ids = np.flatnonzero(valid)
        distance = np.linalg.norm(candidates[valid_ids] - anchor, axis=1)
        chosen = valid_ids[np.argmin(distance)]
        positions[index] = candidates[chosen]
        repaired[index] = True
        repair_distance[index] = distance.min()
    return positions, repaired, repair_distance


def _recenter_internal_cluster_offsets(
    positions, anchors, normals, radii, representative, model
):
    """Remove drag-weighted translation caused solely by internal pair forces."""
    pairs, distance, contact = _compatible_pairs(positions, normals, radii, model)
    count = len(positions)
    parent = np.arange(count, dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for pair_index, (first, second) in enumerate(pairs):
        if distance[pair_index] <= max(
            contact[pair_index], model.capillary_length_m
        ):
            union(int(first), int(second))
    if not count:
        return positions, 0, 0.0
    roots = np.array([find(index) for index in range(count)], dtype=np.int64)
    recentered = 0
    maximum_shift = 0.0
    # This derived row tracks an ensemble centroid, while representative_count
    # remains an accounting field.  Use the same radius-proportional drag as
    # the pair mobility so internal corrections have exactly zero weighted
    # translation in the solver's discrete mechanics.
    drag = np.maximum(radii, 1.0e-12)
    offset = positions - anchors
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root)
        if len(members) < 2:
            continue
        mean_offset = np.average(offset[members], axis=0, weights=drag[members])
        positions[members] -= mean_offset
        recentered += 1
        maximum_shift = max(maximum_shift, float(np.linalg.norm(mean_offset)))
    return positions, recentered, maximum_shift


def _recenter_excess_contact_offsets(positions, anchors, normals, radii, model):
    """Free tether capacity only in components that exceed soft-contact strain."""
    pairs, distance, contact = _compatible_pairs(positions, normals, radii, model)
    if not len(pairs):
        return positions, 0, 0.0
    compression = np.divide(
        contact - distance,
        contact,
        out=np.zeros_like(contact),
        where=contact > 0.0,
    )
    excessive = compression > model.maximum_contact_compression_fraction + 1.0e-4
    if not np.any(excessive):
        return positions, 0, 0.0
    count = len(positions)
    parent = np.arange(count, dtype=np.int64)
    involved = np.zeros(count, dtype=bool)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in pairs[excessive]:
        involved[first] = True
        involved[second] = True
        union(int(first), int(second))
    roots = np.array([find(index) for index in range(count)], dtype=np.int64)
    offset = positions - anchors
    component_count = 0
    maximum_shift = 0.0
    for root in np.unique(roots[involved]):
        members = np.flatnonzero(involved & (roots == root))
        mean_offset = np.average(offset[members], axis=0, weights=radii[members])
        positions[members] -= mean_offset
        component_count += 1
        maximum_shift = max(maximum_shift, float(np.linalg.norm(mean_offset)))
    return positions, component_count, maximum_shift


def _cluster_diagnostics(positions, radii, representative, ids, model):
    count = len(positions)
    neighbors = np.zeros(count, dtype=np.uint16)
    packing = np.zeros(count, dtype=np.float64)
    cluster_id = ids.copy()
    cluster_size = np.ones(count, dtype=np.uint32)
    cluster_representative = representative.astype(np.float64).copy()
    if not count:
        return neighbors, packing, cluster_id, cluster_size, cluster_representative
    normals = np.zeros_like(positions)
    normals[:, 1] = 1.0
    # Clustering is based on geometric distance only here. Compatibility was
    # already enforced by the solver and all final positions are top-shell bound.
    pairs = cKDTree(positions).query_pairs(
        model.interaction_distance_m, output_type="ndarray"
    ).astype(np.int64, copy=False)
    parent = np.arange(count, dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            if ids[first_root] <= ids[second_root]:
                parent[second_root] = first_root
            else:
                parent[first_root] = second_root

    interaction_area = np.pi * model.interaction_distance_m**2
    projected_area = representative * np.pi * radii**2
    packing += projected_area / interaction_area
    for first, second in pairs:
        distance = float(np.linalg.norm(positions[second] - positions[first]))
        contact = float(radii[first] + radii[second])
        neighbors[first] = np.uint16(min(int(neighbors[first]) + 1, 65535))
        neighbors[second] = np.uint16(min(int(neighbors[second]) + 1, 65535))
        kernel = max(0.0, 1.0 - distance / model.interaction_distance_m)
        packing[first] += kernel * projected_area[second] / interaction_area
        packing[second] += kernel * projected_area[first] / interaction_area
        if distance <= max(contact, model.capillary_length_m):
            union(int(first), int(second))
    roots = np.array([find(index) for index in range(count)], dtype=np.int64)
    for root in np.unique(roots):
        members = roots == root
        cluster_id[members] = ids[members].min()
        cluster_size[members] = np.uint32(np.count_nonzero(members))
        cluster_representative[members] = representative[members].sum(dtype=np.float64)
    return neighbors, packing, cluster_id, cluster_size, cluster_representative


def build_surface_raft(markers, previous, dt, fields, spec: GridSpec, model):
    """Build one conservative raft snapshot from one marker snapshot."""

    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("Raft dt must be finite and positive")
    selected = _surface_markers(markers)
    raft = np.zeros(len(selected), dtype=RAFT_DTYPE)
    if not len(selected):
        return raft, {
            "surface_bubbles": 0,
            "inherited": 0,
            "new": 0,
            "gas_volume_input_m3": 0.0,
            "gas_volume_output_m3": 0.0,
            "gas_volume_residual_m3": 0.0,
            "support_fallbacks": 0,
            "anchor_support_repairs": 0,
            "maximum_anchor_support_repair_m": 0.0,
            "support_bad": 0,
            "hard_tether_hits": 0,
            "recentered_clusters": 0,
            "maximum_cluster_recentering_m": 0.0,
            "maximum_anchor_displacement_m": 0.0,
            "mean_anchor_displacement_m": 0.0,
            "pair_attraction_constraints": 0,
            "pair_overlap_constraints": 0,
            "remaining_overlap_pairs": 0,
            "maximum_overlap_m": 0.0,
            "deformed_contact_pairs": 0,
            "excess_compression_pairs": 0,
            "maximum_contact_compression_fraction": 0.0,
            "post_repair_recovery_components": 0,
            "maximum_post_repair_recentering_m": 0.0,
            "post_repair_recovery_corrections": 0,
            "mean_neighbor_count": 0.0,
            "maximum_neighbor_count": 0,
            "packing_fraction_quantiles": [0.0] * 5,
            "clusters": 0,
            "largest_cluster": 0,
            "compatible_final_pairs": 0,
        }

    ids = selected["id"]
    anchors = selected["position"].astype(np.float64)
    radii = selected["physical_radius"].astype(np.float64)
    matched, previous_index = _previous_matches(ids, previous)
    positions = anchors.copy()
    if np.any(matched):
        prior = previous[previous_index[matched]]
        anchor_motion = anchors[matched] - prior["raw_anchor_position"].astype(np.float64)
        positions[matched] = prior["raft_position"].astype(np.float64) + anchor_motion
    _hard_tether(positions, anchors, model.maximum_anchor_displacement_m)
    positions, normals, initial_lost = constrain_surface_positions(
        positions, fields, None, spec, model
    )
    support_fallback = initial_lost.copy()
    immobile = np.zeros(len(selected), dtype=bool)
    positions[initial_lost] = anchors[initial_lost]
    total_overlap_constraints = 0
    total_attraction_constraints = 0
    hard_tether_hits = 0
    soft_total = 1.0 - np.exp(-dt / model.anchor_relaxation_seconds)
    soft_iteration = 1.0 - (1.0 - soft_total) ** (1.0 / model.solver_iterations)
    for _ in range(model.solver_iterations):
        normals = _normalized(
            _sample_interpolated(
                fields, None, "normal", positions, 1.0, spec, vector=True
            )
        )
        positions, overlaps, attractions = _solve_pairs(
            positions, normals, radii, ids, dt, model, immobile
        )
        total_overlap_constraints += overlaps
        total_attraction_constraints += attractions
        positions += soft_iteration * (anchors - positions)
        hard_tether_hits += int(
            np.count_nonzero(
                _hard_tether(
                    positions, anchors, model.maximum_anchor_displacement_m
                )
            )
        )

    normals = _normalized(
        _sample_interpolated(
            fields, None, "normal", positions, 1.0, spec, vector=True
        )
    )
    positions, recentered_clusters, maximum_recentering = (
        _recenter_internal_cluster_offsets(
            positions,
            anchors,
            normals,
            radii,
            selected["representative_count"].astype(np.float64),
            model,
        )
    )
    hard_tether_hits += int(
        np.count_nonzero(
            _hard_tether(positions, anchors, model.maximum_anchor_displacement_m)
        )
    )
    positions, normals, recenter_lost = constrain_surface_positions(
        positions, fields, None, spec, model
    )
    support_fallback |= recenter_lost
    positions[recenter_lost] = anchors[recenter_lost]
    _hard_tether(positions, anchors, model.maximum_anchor_displacement_m)

    nonpenetration_corrections = 0
    for _ in range(model.nonpenetration_iterations):
        normals = _normalized(
            _sample_interpolated(
                fields, None, "normal", positions, 1.0, spec, vector=True
            )
        )
        positions, corrections = _solve_nonpenetration(
            positions, anchors, normals, radii, ids, model, immobile
        )
        nonpenetration_corrections += corrections
        hard_tether_hits += int(
            np.count_nonzero(
                _hard_tether(
                    positions, anchors, model.maximum_anchor_displacement_m
                )
            )
        )
        positions, normals, lost = constrain_surface_positions(
            positions, fields, None, spec, model
        )
        support_fallback |= lost
        positions[lost] = anchors[lost]
        hard_tether_hits += int(
            np.count_nonzero(
                _hard_tether(
                    positions, anchors, model.maximum_anchor_displacement_m
                )
            )
        )
        if corrections == 0:
            break

    normals = _normalized(
        _sample_interpolated(
            fields, None, "normal", positions, 1.0, spec, vector=True
        )
    )
    positions, recovery_components, maximum_recovery_recentering = (
        _recenter_excess_contact_offsets(
            positions, anchors, normals, radii, model
        )
    )
    recovery_corrections = 0
    if recovery_components:
        _hard_tether(positions, anchors, model.maximum_anchor_displacement_m)
        positions, normals, lost = constrain_surface_positions(
            positions, fields, None, spec, model
        )
        support_fallback |= lost
        positions[lost] = anchors[lost]
        for _ in range(model.contact_recovery_iterations):
            normals = _normalized(
                _sample_interpolated(
                    fields, None, "normal", positions, 1.0, spec, vector=True
                )
            )
            positions, corrections = _solve_nonpenetration(
                positions, anchors, normals, radii, ids, model, immobile
            )
            recovery_corrections += corrections
            _hard_tether(
                positions, anchors, model.maximum_anchor_displacement_m
            )
            positions, normals, lost = constrain_surface_positions(
                positions, fields, None, spec, model
            )
            support_fallback |= lost
            positions[lost] = anchors[lost]
            _hard_tether(
                positions, anchors, model.maximum_anchor_displacement_m
            )
            if corrections == 0:
                break

    positions, normals, final_lost = constrain_surface_positions(
        positions, fields, None, spec, model
    )
    support_fallback |= final_lost
    positions[final_lost] = anchors[final_lost]
    hard_tether_hits += int(
        np.count_nonzero(
            _hard_tether(positions, anchors, model.maximum_anchor_displacement_m)
        )
    )
    valid_after_tether, support = _position_validity(positions, fields, spec, model)
    newly_invalid = ~valid_after_tether
    support_fallback |= newly_invalid
    positions[newly_invalid] = anchors[newly_invalid]
    anchor_valid, support = _position_validity(positions, fields, spec, model)
    positions, anchor_repair, anchor_repair_distance = _repair_unsupported_anchors(
        positions, anchors, ~anchor_valid, fields, spec, model
    )
    safe_support_anchor = anchors.copy()
    safe_support_anchor[anchor_repair] = positions[anchor_repair]
    normals = _normalized(
        _sample_interpolated(
            fields, None, "normal", positions, 1.0, spec, vector=True
        )
    )
    positions, post_repair_components, post_repair_recentering = (
        _recenter_excess_contact_offsets(
            positions, anchors, normals, radii, model
        )
    )
    post_repair_corrections = 0
    if post_repair_components:
        for _ in range(model.contact_recovery_iterations):
            normals = _normalized(
                _sample_interpolated(
                    fields, None, "normal", positions, 1.0, spec, vector=True
                )
            )
            positions, corrections = _solve_nonpenetration(
                positions, anchors, normals, radii, ids, model, immobile
            )
            post_repair_corrections += corrections
            _hard_tether(
                positions, anchors, model.maximum_anchor_displacement_m
            )
            positions, normals, lost = constrain_surface_positions(
                positions, fields, None, spec, model
            )
            support_fallback |= lost
            positions[lost] = safe_support_anchor[lost]
            _hard_tether(
                positions, anchors, model.maximum_anchor_displacement_m
            )
            if corrections == 0:
                break
    final_valid, _ = _position_validity(positions, fields, spec, model)
    support_fallback |= ~final_valid
    positions[~final_valid] = safe_support_anchor[~final_valid]
    anchor_valid, support = _position_validity(positions, fields, spec, model)
    support_bad = ~anchor_valid
    normals = _normalized(
        _sample_interpolated(
            fields, None, "normal", positions, 1.0, spec, vector=True
        )
    )

    compatible, final_distance, final_contact = _compatible_pairs(
        positions,
        _normalized(
            _sample_interpolated(
                fields, None, "normal", positions, 1.0, spec, vector=True
            )
        ),
        radii,
        model,
    )
    physical_overlap = final_contact - final_distance
    compression = np.divide(
        physical_overlap,
        final_contact,
        out=np.zeros_like(physical_overlap),
        where=final_contact > 0.0,
    )
    compression = np.maximum(compression, 0.0)
    deformed_contact = compression > 1.0e-7
    excess_compression = (
        compression > model.maximum_contact_compression_fraction + 1.0e-4
    )
    contact_count = np.zeros(len(raft), dtype=np.uint16)
    maximum_compression_by_row = np.zeros(len(raft), dtype=np.float64)
    if len(compatible):
        for endpoint in (0, 1):
            row = compatible[:, endpoint]
            np.add.at(contact_count, row[deformed_contact], np.uint16(1))
            np.maximum.at(maximum_compression_by_row, row, compression)
    neighbors, packing, cluster_id, cluster_size, cluster_representative = (
        _cluster_diagnostics(
            positions,
            radii,
            selected["representative_count"].astype(np.float64),
            ids,
            model,
        )
    )

    raft["marker_id"] = ids
    for name in (
        "physical_radius",
        "representative_count",
        "phase_volume",
        "shape",
        "state_age",
        "random_key",
    ):
        raft[name] = selected[name]
    raft["raw_anchor_position"] = selected["position"]
    raft["raft_position"] = positions.astype(np.float32)
    if np.any(matched):
        raft["raft_velocity"][matched] = (
            (positions[matched] - previous[previous_index[matched]]["raft_position"])
            / dt
        ).astype(np.float32)
    if np.any(~matched):
        velocity = selected["velocity"][~matched].astype(np.float64)
        new_normals = normals[~matched]
        velocity -= np.sum(velocity * new_normals, axis=1)[:, None] * new_normals
        raft["raft_velocity"][~matched] = velocity.astype(np.float32)
    raft["neighbor_count"] = neighbors
    raft["contact_count"] = contact_count
    raft["maximum_contact_compression"] = maximum_compression_by_row.astype(
        np.float32
    )
    raft["packing_fraction"] = packing.astype(np.float32)
    raft["cluster_id"] = cluster_id
    raft["cluster_size"] = cluster_size
    raft["cluster_representative_count"] = cluster_representative.astype(np.float32)
    displacement = np.linalg.norm(positions - anchors, axis=1)
    raft["anchor_displacement"] = displacement.astype(np.float32)
    raft["support_value"] = support.astype(np.float32)
    raft["support_fallback"] = support_fallback.astype(np.uint8)
    raft["anchor_support_repair"] = anchor_repair.astype(np.uint8)

    gas_input = float(selected["phase_volume"].sum(dtype=np.float64))
    gas_output = float(raft["phase_volume"].sum(dtype=np.float64))
    unique_clusters = np.unique(cluster_id)
    return raft, {
        "surface_bubbles": len(raft),
        "inherited": int(np.count_nonzero(matched)),
        "new": int(np.count_nonzero(~matched)),
        "gas_volume_input_m3": gas_input,
        "gas_volume_output_m3": gas_output,
        "gas_volume_residual_m3": gas_input - gas_output,
        "support_fallbacks": int(np.count_nonzero(support_fallback)),
        "anchor_support_repairs": int(np.count_nonzero(anchor_repair)),
        "maximum_anchor_support_repair_m": float(
            anchor_repair_distance.max(initial=0.0)
        ),
        "support_bad": int(np.count_nonzero(support_bad)),
        "hard_tether_hits": hard_tether_hits,
        "recentered_clusters": recentered_clusters,
        "maximum_cluster_recentering_m": maximum_recentering,
        "maximum_anchor_displacement_m": float(displacement.max(initial=0.0)),
        "mean_anchor_displacement_m": float(displacement.mean()),
        "pair_attraction_constraints": total_attraction_constraints,
        "pair_overlap_constraints": total_overlap_constraints,
        "nonpenetration_corrections": nonpenetration_corrections,
        "recovery_components": recovery_components,
        "maximum_recovery_recentering_m": maximum_recovery_recentering,
        "recovery_corrections": recovery_corrections,
        "post_repair_recovery_components": post_repair_components,
        "maximum_post_repair_recentering_m": post_repair_recentering,
        "post_repair_recovery_corrections": post_repair_corrections,
        "remaining_overlap_pairs": int(np.count_nonzero(deformed_contact)),
        "maximum_overlap_m": float(physical_overlap.max(initial=0.0)),
        "deformed_contact_pairs": int(np.count_nonzero(deformed_contact)),
        "excess_compression_pairs": int(np.count_nonzero(excess_compression)),
        "maximum_contact_compression_fraction": float(
            compression.max(initial=0.0)
        ),
        "mean_neighbor_count": float(neighbors.mean()),
        "maximum_neighbor_count": int(neighbors.max(initial=0)),
        "packing_fraction_quantiles": np.quantile(
            packing, (0.0, 0.5, 0.9, 0.99, 1.0)
        ).tolist(),
        "clusters": len(unique_clusters),
        "largest_cluster": int(cluster_size.max(initial=0)),
        "compatible_final_pairs": len(compatible),
    }
