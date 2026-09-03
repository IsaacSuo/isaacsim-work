"""Volume-conservative Plateau-cell topology and render geometry.

Surface-raft marker packets are expanded into physical-scale cells instead of
being rendered as enlarged particles.  Contacting cells are partitioned by a
local Laguerre (power) diagram.  Every pair owns at most one shared membrane;
three or more incident membranes create an explicit Plateau node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .marker_birth import sample_scalar_trilinear
from .state_machine import splitmix64
from .surface_raft import RAFT_DTYPE


CELL_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("parent_marker_id", "<u8"),
        ("child_index", "<u2"),
        ("physical_radius", "<f4"),
        ("gas_volume", "<f8"),
        ("position", "<f4", (3,)),
        ("packet_anchor_position", "<f4", (3,)),
        ("packing_displacement", "<f4"),
        ("normal", "<f4", (3,)),
        ("footprint_area", "<f8"),
        ("base_height", "<f4"),
        ("peak_height", "<f4"),
        ("wetness", "<f4"),
        ("cluster_id", "<u8"),
        ("coalesced_group_id", "<u8"),
        ("lod_class", "u1"),
    ]
)

FILM_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("cell_a", "<u8"),
        ("cell_b", "<u8"),
        ("age", "<f4"),
        ("endpoint_a", "<f4", (3,)),
        ("endpoint_b", "<f4", (3,)),
        ("surface_normal", "<f4", (3,)),
        ("height", "<f4"),
        ("area", "<f8"),
        ("thickness", "<f8"),
        ("liquid_volume", "<f8"),
        ("curvature", "<f4"),
        ("wetness", "<f4"),
    ]
)

BORDER_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("kind", "u1"),
        ("owner_id", "<u8"),
        ("start", "<f4", (3,)),
        ("end", "<f4", (3,)),
        ("radius", "<f4"),
        ("liquid_volume", "<f8"),
        ("wetness", "<f4"),
    ]
)

NODE_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("incident_cells", "<u2"),
        ("incident_films", "<u2"),
        ("base_position", "<f4", (3,)),
        ("top_position", "<f4", (3,)),
        ("radius", "<f4"),
        ("liquid_volume", "<f8"),
        ("wetness", "<f4"),
    ]
)

EVENT_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("kind", "u1"),
        ("topology_id", "<u8"),
        ("event_time", "<f8"),
        ("gas_volume", "<f8"),
        ("film_area", "<f8"),
    ]
)


@dataclass(frozen=True)
class PlateauCellModel:
    polygon_sides: int = 20
    child_radial_spacing: float = 1.65
    footprint_radius_scale: float = 1.05
    base_height_fraction: float = 0.70
    initial_film_thickness_m: float = 5.0e-6
    minimum_film_thickness_m: float = 8.0e-8
    film_drainage_seconds: float = 2.5
    border_radius_fraction: float = 0.08
    minimum_border_radius_m: float = 2.5e-5
    maximum_border_radius_m: float = 4.0e-4
    node_radius_scale: float = 1.35
    endpoint_merge_tolerance_m: float = 2.0e-5
    maximum_cell_aspect: float = 12.0

    def __post_init__(self):
        values = (
            self.child_radial_spacing,
            self.footprint_radius_scale,
            self.base_height_fraction,
            self.initial_film_thickness_m,
            self.minimum_film_thickness_m,
            self.film_drainage_seconds,
            self.border_radius_fraction,
            self.minimum_border_radius_m,
            self.maximum_border_radius_m,
            self.node_radius_scale,
            self.endpoint_merge_tolerance_m,
            self.maximum_cell_aspect,
        )
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("Plateau model parameters must be finite and positive")
        if not 12 <= int(self.polygon_sides) <= 64:
            raise ValueError("polygon_sides must lie within 12..64")
        if not 0.0 < self.base_height_fraction < 1.0:
            raise ValueError("base_height_fraction must lie within (0,1)")
        if self.minimum_film_thickness_m >= self.initial_film_thickness_m:
            raise ValueError("minimum film thickness must be below initial thickness")

    def metadata(self):
        return {
            **self.__dict__,
            "cell_partition": "local surface Laguerre/power cells clipped by physical footprints",
            "gas_volume_policy": (
                "parent representative count is expanded into physical-radius children; "
                "cell prism-plus-pyramid volume equals the parent phase volume exactly"
            ),
            "film_policy": "one stable shared membrane per power-cell contact edge",
            "plateau_policy": "three or more incident shared-film endpoints form a node",
            "rupture_policy": (
                "film thickness drains exponentially; reaching the minimum creates an "
                "auditable rupture/coalesced-group event without deleting gas volume"
            ),
        }


def _normalized(vectors, fallback=(0.0, 1.0, 0.0)):
    vectors = np.asarray(vectors, dtype=np.float64).copy()
    length = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(length) & (length > 1.0e-12)
    vectors[valid] /= length[valid, None]
    vectors[~valid] = fallback
    return vectors


def _sample_phi(fields, spec, positions):
    """Sample signed surface distance from either a grid or surface sampler."""

    if hasattr(fields, "sample_phi"):
        return np.asarray(fields.sample_phi(positions), dtype=np.float64)
    return sample_scalar_trilinear(fields["phi"], positions, spec)


def _sample_normal(fields, spec, positions):
    """Sample smooth surface normals from either a grid or surface sampler."""

    if hasattr(fields, "sample_normal"):
        return _normalized(fields.sample_normal(positions))
    return _normalized(
        np.column_stack(
            [
                sample_scalar_trilinear(
                    fields["normal"][..., axis], positions, spec, outside=0.0
                )
                for axis in range(3)
            ]
        )
    )


def _stable_id(values, salt):
    value = np.uint64(int(salt) & ((1 << 64) - 1))
    for item in sorted(int(v) for v in values):
        value = splitmix64(
            np.asarray([value ^ np.uint64(item)], dtype=np.uint64)
        )[0]
    return value


def _tangent_basis(normal, key):
    normal = _normalized(np.asarray(normal, dtype=np.float64)[None, :])[0]
    reference = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(normal[0])) > 0.85:
        reference[:] = (0.0, 0.0, 1.0)
    first = _normalized(
        (reference - np.dot(reference, normal) * normal)[None, :]
    )[0]
    second = np.cross(normal, first)
    angle = ((int(key) >> 11) / float(1 << 53)) * 2.0 * np.pi
    rotated = np.cos(angle) * first + np.sin(angle) * second
    return rotated, np.cross(normal, rotated)


def expand_marker_packets(raft, fields, spec, model: PlateauCellModel):
    """Expand representative packets without increasing any physical radius."""

    raft = np.asarray(raft)
    if raft.dtype != RAFT_DTYPE:
        raise ValueError("raft has the wrong dtype")
    if not len(raft):
        return np.empty(0, dtype=CELL_DTYPE), {
            "parent_volume_residual_m3": 0.0,
            "expanded_cells": 0,
            "source_marker_packets": 0,
            "maximum_children_per_packet": 0,
        }
    positions = raft["raft_position"].astype(np.float64)
    normals = _sample_normal(fields, spec, positions)
    rows = []
    maximum_parent_residual = 0.0
    for row_index, parent in enumerate(raft):
        representative = float(parent["representative_count"])
        full = int(np.floor(representative + 1.0e-7))
        residual = representative - full
        weights = [1.0] * full
        if residual > 1.0e-6:
            weights.append(residual)
        if not weights:
            weights = [representative]
        parent_volume = float(parent["phase_volume"])
        weight_total = float(sum(weights))
        base_radius = float(parent["physical_radius"])
        first, second = _tangent_basis(normals[row_index], parent["random_key"])
        child_volumes = []
        for child_index, weight in enumerate(weights):
            fraction = weight / weight_total
            gas_volume = parent_volume * fraction
            physical_radius = base_radius * weight ** (1.0 / 3.0)
            if child_index == 0:
                offset = np.zeros(3, dtype=np.float64)
            else:
                golden = 2.399963229728653
                angle = child_index * golden
                radial = (
                    model.child_radial_spacing
                    * base_radius
                    * np.sqrt(float(child_index))
                )
                offset = radial * (
                    np.cos(angle) * first + np.sin(angle) * second
                )
            position = positions[row_index] + offset
            phi = _sample_phi(fields, spec, position[None, :])[0]
            local_normal = _sample_normal(fields, spec, position[None, :])[0]
            if np.isfinite(phi):
                position = position - phi * local_normal + 0.20 * physical_radius * local_normal
            cell_id = _stable_id(
                (int(parent["marker_id"]), child_index), 0xCE11A11
            )
            rows.append(
                (
                    cell_id,
                    parent["marker_id"],
                    child_index,
                    physical_radius,
                    gas_volume,
                    position,
                    position,
                    0.0,
                    local_normal,
                )
            )
            child_volumes.append(gas_volume)
        maximum_parent_residual = max(
            maximum_parent_residual,
            abs(sum(child_volumes) - parent_volume),
        )
    cells = np.zeros(len(rows), dtype=CELL_DTYPE)
    for index, row in enumerate(rows):
        (
            cells["id"][index],
            cells["parent_marker_id"][index],
            cells["child_index"][index],
            cells["physical_radius"][index],
            cells["gas_volume"][index],
            cells["position"][index],
            cells["packet_anchor_position"][index],
            cells["packing_displacement"][index],
            cells["normal"][index],
        ) = row
    order = np.argsort(cells["id"])
    cells = cells[order]
    return cells, {
        "parent_volume_residual_m3": maximum_parent_residual,
        "expanded_cells": len(cells),
        "source_marker_packets": len(raft),
        "maximum_children_per_packet": int(
            np.max(np.bincount(np.searchsorted(raft["marker_id"], cells["parent_marker_id"])))
        )
        if len(cells)
        else 0,
    }


def resolve_power_containment(cells, fields, spec, maximum_displacement=0.012, iterations=128):
    """Move hidden weighted sites tangentially until every gas cell is visible.

    A highly unequal compressed pair can place the small site's radical plane
    outside its own footprint. That is not a valid two-cell Plateau topology.
    The derived render cells are therefore separated on the liquid tangent
    sheet, without changing radius or gas volume and within an audited tether.
    """

    if len(cells) < 2:
        return cells, {"packing_corrections": 0, "unresolved_containments": 0, "maximum_packing_displacement_m": 0.0}
    cells = cells.copy()
    anchor = cells["packet_anchor_position"].astype(np.float64)
    position = cells["position"].astype(np.float64)
    radius = cells["physical_radius"].astype(np.float64)
    correction_count = 0
    for _ in range(iterations):
        candidate = cKDTree(position).query_pairs(
            2.0 * float(radius.max()), output_type="ndarray"
        )
        if not len(candidate):
            break
        delta = position[candidate[:, 1]] - position[candidate[:, 0]]
        distance = np.linalg.norm(delta, axis=1)
        first_radius = radius[candidate[:, 0]]
        second_radius = radius[candidate[:, 1]]
        visibility = np.sqrt(
            np.maximum(np.abs(first_radius**2 - second_radius**2), 0.0)
        ) + 0.10 * np.minimum(first_radius, second_radius)
        hidden = distance < visibility
        if not np.any(hidden):
            break
        pairs = candidate[hidden]
        delta = delta[hidden]
        distance = distance[hidden]
        visibility = visibility[hidden]
        corrections = np.zeros_like(position)
        counts = np.zeros(len(position), dtype=np.int32)
        for pair_index, (first, second) in enumerate(pairs):
            average_normal = _normalized(
                (cells["normal"][first] + cells["normal"][second])[None, :]
            )[0]
            tangent = delta[pair_index] - np.dot(delta[pair_index], average_normal) * average_normal
            tangent_length = float(np.linalg.norm(tangent))
            if tangent_length <= 1.0e-12:
                direction, _ = _tangent_basis(
                    average_normal, int(cells["id"][first] ^ cells["id"][second])
                )
            else:
                direction = tangent / tangent_length
            needed = float(visibility[pair_index] - distance[pair_index])
            mobility_first = 1.0 / max(radius[first], 1.0e-8)
            mobility_second = 1.0 / max(radius[second], 1.0e-8)
            total = mobility_first + mobility_second
            corrections[first] -= direction * needed * mobility_first / total
            corrections[second] += direction * needed * mobility_second / total
            counts[first] += 1
            counts[second] += 1
        active = counts > 0
        position[active] += 0.85 * corrections[active] / counts[active, None]
        offset = position - anchor
        offset_length = np.linalg.norm(offset, axis=1)
        tethered = offset_length > maximum_displacement
        position[tethered] = anchor[tethered] + offset[tethered] * (
            maximum_displacement / offset_length[tethered]
        )[:, None]
        phi = _sample_phi(fields, spec, position[active])
        sampled_normal = _sample_normal(fields, spec, position[active])
        finite = np.isfinite(phi)
        active_ids = np.flatnonzero(active)
        position[active_ids[finite]] -= phi[finite, None] * sampled_normal[finite]
        cells["normal"][active_ids] = sampled_normal.astype(np.float32)
        correction_count += len(pairs)
    candidate = cKDTree(position).query_pairs(
        2.0 * float(radius.max()), output_type="ndarray"
    )
    unresolved = 0
    if len(candidate):
        distance = np.linalg.norm(position[candidate[:, 1]] - position[candidate[:, 0]], axis=1)
        visibility = np.sqrt(
            np.maximum(
                np.abs(radius[candidate[:, 0]] ** 2 - radius[candidate[:, 1]] ** 2),
                0.0,
            )
        ) + 0.10 * np.minimum(radius[candidate[:, 0]], radius[candidate[:, 1]])
        unresolved = int(np.count_nonzero(distance < visibility - 1.0e-7))
    displacement = np.linalg.norm(position - anchor, axis=1)
    cells["position"] = position.astype(np.float32)
    cells["packing_displacement"] = displacement.astype(np.float32)
    return cells, {
        "packing_corrections": correction_count,
        "unresolved_containments": unresolved,
        "maximum_packing_displacement_m": float(displacement.max(initial=0.0)),
    }


def _contact_pairs(cells):
    if len(cells) < 2:
        return np.empty((0, 2), dtype=np.int64)
    position = cells["position"].astype(np.float64)
    radius = cells["physical_radius"].astype(np.float64)
    candidate = cKDTree(position).query_pairs(
        2.0 * float(radius.max()), output_type="ndarray"
    )
    if not len(candidate):
        return candidate.astype(np.int64)
    distance = np.linalg.norm(
        position[candidate[:, 1]] - position[candidate[:, 0]], axis=1
    )
    contact = radius[candidate[:, 0]] + radius[candidate[:, 1]]
    normal_alignment = np.sum(
        cells["normal"][candidate[:, 0]] * cells["normal"][candidate[:, 1]],
        axis=1,
    )
    return candidate[(distance < contact) & (normal_alignment >= 0.50)].astype(
        np.int64, copy=False
    )


def _components(count, pairs, ids):
    parent = np.arange(count, dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        a, b = find(int(a)), find(int(b))
        if a != b:
            if ids[a] <= ids[b]:
                parent[b] = a
            else:
                parent[a] = b

    for a, b in pairs:
        union(a, b)
    root = np.asarray([find(i) for i in range(count)], dtype=np.int64)
    return [np.flatnonzero(root == value) for value in np.unique(root)]


def _clip_polygon(polygon, a, b, tolerance=1.0e-12):
    if not len(polygon):
        return polygon
    output = []
    previous = polygon[-1]
    previous_value = float(np.dot(a, previous) - b)
    previous_inside = previous_value <= tolerance
    for current in polygon:
        current_value = float(np.dot(a, current) - b)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            t = previous_value / denominator if abs(denominator) > 1.0e-20 else 0.0
            output.append(previous + t * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return np.asarray(output, dtype=np.float64)


def _polygon_area_centroid(polygon):
    if len(polygon) < 3:
        return 0.0, np.zeros(2, dtype=np.float64)
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - polygon[:, 1] * following[:, 0]
    area2 = float(cross.sum())
    area = 0.5 * abs(area2)
    if abs(area2) <= 1.0e-20:
        return 0.0, polygon.mean(axis=0)
    centroid = np.sum(
        (polygon + np.roll(polygon, -1, axis=0)) * cross[:, None], axis=0
    ) / (3.0 * area2)
    return area, centroid


def _shared_power_edge(local_centers, radii, first, second, members):
    ci, cj = local_centers[first], local_centers[second]
    ri, rj = radii[first], radii[second]
    a = 2.0 * (cj - ci)
    length2 = float(np.dot(a, a))
    if length2 <= 1.0e-20:
        return None
    b = float(np.dot(cj, cj) - rj * rj - np.dot(ci, ci) + ri * ri)
    point = a * (b / length2)
    direction = np.array((-a[1], a[0]), dtype=np.float64) / np.sqrt(length2)
    lower, upper = -np.inf, np.inf
    for center, radius in ((ci, ri), (cj, rj)):
        delta = point - center
        along = float(np.dot(delta, direction))
        perpendicular2 = float(np.dot(delta, delta) - along * along)
        half = np.sqrt(max(radius * radius - perpendicular2, 0.0))
        lower = max(lower, -along - half)
        upper = min(upper, -along + half)
    for other in members:
        if other in (first, second):
            continue
        ck, rk = local_centers[other], radii[other]
        constraint_a = 2.0 * (ck - ci)
        constraint_b = float(
            np.dot(ck, ck) - rk * rk - np.dot(ci, ci) + ri * ri
        )
        coefficient = float(np.dot(constraint_a, direction))
        constant = float(constraint_b - np.dot(constraint_a, point))
        if abs(coefficient) <= 1.0e-14:
            if constant < 0.0:
                return None
        elif coefficient > 0.0:
            upper = min(upper, constant / coefficient)
        else:
            lower = max(lower, constant / coefficient)
    if not np.isfinite(lower + upper) or upper - lower <= 1.0e-7:
        return None
    return point + lower * direction, point + upper * direction


def _component_frame(cells, members):
    position = cells["position"][members].astype(np.float64)
    normal = _normalized(
        cells["normal"][members].astype(np.float64).mean(axis=0)[None, :]
    )[0]
    origin = position.mean(axis=0)
    first, second = _tangent_basis(normal, int(cells["id"][members].min()))
    local = np.column_stack(
        (np.dot(position - origin, first), np.dot(position - origin, second))
    )
    return origin, first, second, normal, local


def resolve_power_slivers(
    cells,
    fields,
    spec,
    model,
    minimum_area_fraction=None,
    maximum_displacement=0.012,
    iterations=64,
):
    """Enforce a volume-derived minimum visible area for every power cell.

    A centroidal displacement is not a valid solver for a symmetrically
    squeezed cell: the visible polygon and the weighted site can have the same
    centroid even when the polygon is much too small.  Instead, this pass treats
    every restricted power half-plane as an active inequality.  Contacting
    sites are separated by the minimum tangent-sheet distance that leaves the
    required clearance on *both* sides of their radical axis.  Consequently the
    intersection contains a disk large enough to satisfy the cell-height bound.
    Gas volume and physical radius never change.
    """

    if len(cells) < 2:
        return cells, {
            "power_sliver_corrections": 0,
            "unresolved_power_slivers": 0,
            "minimum_power_cell_area_fraction": 1.0,
            "maximum_required_power_cell_area_fraction": 0.0,
            "minimum_power_halfplane_clearance_fraction": 1.0,
            "unresolved_power_cell_diagnostics": [],
        }
    cells = cells.copy()
    anchor = cells["packet_anchor_position"].astype(np.float64)
    position = cells["position"].astype(np.float64)
    correction_count = 0
    footprint_radius = (
        cells["physical_radius"].astype(np.float64)
        * model.footprint_radius_scale
    )
    if minimum_area_fraction is None:
        maximum_height = (
            model.maximum_cell_aspect
            * cells["physical_radius"].astype(np.float64)
        )
        required_area = 3.0 * cells["gas_volume"].astype(np.float64) / np.maximum(
            maximum_height, 1.0e-30
        )
        required_fraction = required_area / np.maximum(
            np.pi * footprint_radius**2, 1.0e-30
        )
    else:
        required_fraction = np.full(len(cells), float(minimum_area_fraction))

    # A small, dimensionless safety allowance absorbs polygon and float32
    # round-off.  It is tied to the derived area constraint, not to world scale.
    clearance_fraction = np.sqrt(np.clip(required_fraction, 0.0, 0.95))
    clearance_fraction = np.minimum(clearance_fraction + 1.0e-3, 0.98)

    def evaluate(current_cells):
        pairs = _contact_pairs(current_cells)
        components = _components(len(current_cells), pairs, current_cells["id"])
        adjacency = [set() for _ in range(len(current_cells))]
        for first, second in pairs:
            adjacency[int(first)].add(int(second))
            adjacency[int(second)].add(int(first))
        fractions = np.ones(len(current_cells), dtype=np.float64)
        minimum_clearance = np.ones(len(current_cells), dtype=np.float64)
        worst_neighbour = np.full(len(current_cells), -1, dtype=np.int64)
        angles = np.arange(model.polygon_sides) * (
            2.0 * np.pi / model.polygon_sides
        )
        for members in components:
            if len(members) < 2:
                continue
            _, _, _, _, local = _component_frame(current_cells, members)
            local_by_global = {
                int(global_id): local[i] for i, global_id in enumerate(members)
            }
            for global_id in members:
                global_id = int(global_id)
                center = local_by_global[global_id]
                radius = footprint_radius[global_id]
                polygon = center + radius * np.column_stack(
                    (np.cos(angles), np.sin(angles))
                )
                for other in adjacency[global_id]:
                    other_center = local_by_global[other]
                    other_radius = footprint_radius[other]
                    delta = other_center - center
                    distance = float(np.linalg.norm(delta))
                    if distance <= 1.0e-14:
                        clearance = -np.inf
                    else:
                        clearance = (
                            distance * distance
                            + radius * radius
                            - other_radius * other_radius
                        ) / (2.0 * distance * radius)
                    if clearance < minimum_clearance[global_id]:
                        minimum_clearance[global_id] = clearance
                        worst_neighbour[global_id] = int(other)
                    a = 2.0 * delta
                    b = float(
                        np.dot(other_center, other_center)
                        - other_radius**2
                        - np.dot(center, center)
                        + radius**2
                    )
                    polygon = _clip_polygon(polygon, a, b)
                    if len(polygon) < 3:
                        break
                area, _ = _polygon_area_centroid(polygon)
                fractions[global_id] = area / max(
                    np.pi * radius * radius, 1.0e-30
                )
        bad = fractions + 2.0e-6 < required_fraction
        diagnostics = []
        for cell_index in np.flatnonzero(bad):
            other = int(worst_neighbour[cell_index])
            diagnostics.append(
                {
                    "cell_id": int(current_cells["id"][cell_index]),
                    "parent_marker_id": int(
                        current_cells["parent_marker_id"][cell_index]
                    ),
                    "physical_radius_m": float(
                        current_cells["physical_radius"][cell_index]
                    ),
                    "gas_volume_m3": float(current_cells["gas_volume"][cell_index]),
                    "actual_area_fraction": float(fractions[cell_index]),
                    "required_area_fraction": float(required_fraction[cell_index]),
                    "minimum_halfplane_clearance_fraction": float(
                        minimum_clearance[cell_index]
                    ),
                    "worst_neighbour_cell_id": (
                        int(current_cells["id"][other]) if other >= 0 else None
                    ),
                    "contact_neighbour_cell_ids": [
                        int(current_cells["id"][value])
                        for value in sorted(adjacency[cell_index])
                    ],
                    "packing_displacement_m": float(
                        np.linalg.norm(position[cell_index] - anchor[cell_index])
                    ),
                }
            )
        return pairs, components, fractions, minimum_clearance, diagnostics

    def minimum_pair_distance(first_radius, second_radius, first_alpha):
        # Solve d^2 - 2*alpha*r_i*d + r_i^2-r_j^2 >= 0 on the
        # post-containment (large-distance) branch.
        discriminant = (
            second_radius * second_radius
            - (1.0 - first_alpha * first_alpha) * first_radius * first_radius
        )
        if discriminant <= 0.0:
            return 0.0
        return first_alpha * first_radius + np.sqrt(discriminant)

    for _ in range(iterations):
        cells["position"] = position.astype(np.float32)
        pairs, components, fractions, _, diagnostics = evaluate(cells)
        if not diagnostics:
            break
        corrections = np.zeros_like(position)
        counts = np.zeros(len(cells), dtype=np.int32)
        for members in components:
            if len(members) < 2:
                continue
            _, axis_first, axis_second, _, local = _component_frame(
                cells, members
            )
            member_local_index = {
                int(global_id): local_index
                for local_index, global_id in enumerate(members)
            }
            member_set = set(int(value) for value in members)
            for first, second in pairs:
                first, second = int(first), int(second)
                if first not in member_set or second not in member_set:
                    continue
                first_local = local[member_local_index[first]]
                second_local = local[member_local_index[second]]
                delta = second_local - first_local
                distance = float(np.linalg.norm(delta))
                first_radius = footprint_radius[first]
                second_radius = footprint_radius[second]
                target = max(
                    minimum_pair_distance(
                        first_radius,
                        second_radius,
                        clearance_fraction[first],
                    ),
                    minimum_pair_distance(
                        second_radius,
                        first_radius,
                        clearance_fraction[second],
                    ),
                )
                if distance + 1.0e-9 >= target:
                    continue
                if distance <= 1.0e-12:
                    angle = (
                        int(cells["id"][first] ^ cells["id"][second]) % 65521
                    ) * (2.0 * np.pi / 65521.0)
                    direction2 = np.array((np.cos(angle), np.sin(angle)))
                else:
                    direction2 = delta / distance
                direction = direction2[0] * axis_first + direction2[1] * axis_second
                needed = target - distance
                mobility_first = 1.0 / max(first_radius, 1.0e-12)
                mobility_second = 1.0 / max(second_radius, 1.0e-12)
                mobility_total = mobility_first + mobility_second
                corrections[first] -= (
                    direction * needed * mobility_first / mobility_total
                )
                corrections[second] += (
                    direction * needed * mobility_second / mobility_total
                )
                counts[first] += 1
                counts[second] += 1
        active = counts > 0
        if not np.any(active):
            break
        step = 0.85 * corrections
        step_length = np.linalg.norm(step, axis=1)
        maximum_step = 0.45 * cells["physical_radius"].astype(np.float64)
        limited = step_length > maximum_step
        step[limited] *= (maximum_step[limited] / step_length[limited])[:, None]
        position[active] += step[active]
        offset = position - anchor
        length = np.linalg.norm(offset, axis=1)
        tethered = length > maximum_displacement
        position[tethered] = anchor[tethered] + offset[tethered] * (
            maximum_displacement / length[tethered]
        )[:, None]
        active_ids = np.flatnonzero(active)
        phi = _sample_phi(fields, spec, position[active_ids])
        sampled_normal = _sample_normal(fields, spec, position[active_ids])
        finite = np.isfinite(phi)
        position[active_ids[finite]] -= phi[finite, None] * sampled_normal[finite]
        cells["normal"][active_ids] = sampled_normal.astype(np.float32)
        correction_count += int(np.count_nonzero(active))

    cells["position"] = position.astype(np.float32)
    _, _, fractions, minimum_clearance, diagnostics = evaluate(cells)
    unresolved = len(diagnostics)
    displacement = np.linalg.norm(position - anchor, axis=1)
    cells["position"] = position.astype(np.float32)
    cells["packing_displacement"] = displacement.astype(np.float32)
    return cells, {
        "power_sliver_corrections": correction_count,
        "unresolved_power_slivers": unresolved,
        "minimum_power_cell_area_fraction": float(fractions.min(initial=1.0)),
        "maximum_required_power_cell_area_fraction": float(
            required_fraction.max(initial=0.0)
        ),
        "minimum_power_halfplane_clearance_fraction": float(
            minimum_clearance.min(initial=1.0)
        ),
        "unresolved_power_cell_diagnostics": diagnostics,
    }


def _event(kind, topology_id, event_time, gas_volume=0.0, film_area=0.0):
    record = np.zeros(1, dtype=EVENT_DTYPE)
    record["id"] = _stable_id((topology_id, int(round(event_time * 1.0e9))), kind)
    record["kind"] = np.uint8(kind)
    record["topology_id"] = np.uint64(topology_id)
    record["event_time"] = event_time
    record["gas_volume"] = gas_volume
    record["film_area"] = film_area
    return record[0]


def build_plateau_cells(
    raft,
    fields,
    spec,
    active_surface_film_area_m2,
    event_time,
    dt,
    previous_films=None,
    previous_nodes=None,
    previous_ruptured_pairs=None,
    model=PlateauCellModel(),
):
    """Build one persistent Plateau topology snapshot and real geometry."""

    cells, expansion_metrics = expand_marker_packets(raft, fields, spec, model)
    cells, packing_metrics = resolve_power_containment(cells, fields, spec)
    initial_unresolved_containments = packing_metrics["unresolved_containments"]
    cells, sliver_metrics = resolve_power_slivers(cells, fields, spec, model)
    if sliver_metrics["unresolved_power_slivers"]:
        raise RuntimeError(
            "Power-cell active-set constraints remained unresolved: "
            f"{sliver_metrics['unresolved_power_cell_diagnostics'][:4]}"
        )
    # The area solver is stronger than the old pairwise visibility pre-pass and
    # can recover symmetric hidden sites that leave the latter with no descent
    # direction.  Audit containment again without moving the converged cells.
    cells, final_packing_audit = resolve_power_containment(
        cells, fields, spec, iterations=0
    )
    packing_metrics = {
        **packing_metrics,
        "initial_unresolved_containments": int(initial_unresolved_containments),
        "unresolved_containments": int(
            final_packing_audit["unresolved_containments"]
        ),
        "maximum_packing_displacement_m": float(
            final_packing_audit["maximum_packing_displacement_m"]
        ),
    }
    if packing_metrics["unresolved_containments"]:
        raise RuntimeError(
            "Power-cell containment remained after active-set area solving: "
            f"{packing_metrics['unresolved_containments']}"
        )
    pairs = _contact_pairs(cells)
    components = _components(len(cells), pairs, cells["id"])
    adjacency = [set() for _ in range(len(cells))]
    for first, second in pairs:
        adjacency[int(first)].add(int(second))
        adjacency[int(second)].add(int(first))
    previous_films = (
        np.empty(0, dtype=FILM_DTYPE)
        if previous_films is None
        else np.asarray(previous_films)
    )
    previous_nodes = (
        np.empty(0, dtype=NODE_DTYPE)
        if previous_nodes is None
        else np.asarray(previous_nodes)
    )
    ruptured = set(
        int(value)
        for value in (
            [] if previous_ruptured_pairs is None else previous_ruptured_pairs
        )
    )
    previous_film_by_id = {int(row["id"]): row for row in previous_films}

    polygons = {}
    frames = {}
    shared_edges = []
    radii = cells["physical_radius"].astype(np.float64) * model.footprint_radius_scale
    for members in components:
        origin, first_axis, second_axis, normal, local = _component_frame(cells, members)
        frames[int(cells["id"][members].min())] = (
            origin,
            first_axis,
            second_axis,
            normal,
        )
        local_by_global = {int(global_id): local[i] for i, global_id in enumerate(members)}
        local_centers_global = np.zeros((len(cells), 2), dtype=np.float64)
        for global_id in members:
            local_centers_global[int(global_id)] = local_by_global[int(global_id)]
        angles = np.arange(model.polygon_sides) * (2.0 * np.pi / model.polygon_sides)
        for global_id in members:
            center = local_by_global[int(global_id)]
            polygon = center + radii[global_id] * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
            for other in adjacency[int(global_id)]:
                ci, cj = center, local_by_global[int(other)]
                a = 2.0 * (cj - ci)
                b = float(
                    np.dot(cj, cj)
                    - radii[other] ** 2
                    - np.dot(ci, ci)
                    + radii[global_id] ** 2
                )
                polygon = _clip_polygon(polygon, a, b)
                if len(polygon) < 3:
                    break
            polygons[int(global_id)] = polygon
        for a, b in pairs:
            if a not in members or b not in members:
                continue
            relevant = np.asarray(
                sorted(
                    {int(a), int(b)}
                    | adjacency[int(a)]
                    | adjacency[int(b)]
                ),
                dtype=np.int64,
            )
            edge = _shared_power_edge(
                local_centers_global,
                radii,
                int(a),
                int(b),
                relevant,
            )
            if edge is not None:
                shared_edges.append((int(a), int(b), edge, int(cells["id"][members].min())))

    # Assign footprint areas and exact gas-volume heights component by component.
    for members in components:
        ratios = []
        for index in members:
            area, _ = _polygon_area_centroid(polygons[int(index)])
            cells["footprint_area"][index] = area
            ratios.append(float(cells["gas_volume"][index]) / max(area, 1.0e-20))
        shared_height = model.base_height_fraction * min(ratios)
        for index, ratio in zip(members, ratios):
            peak_height = shared_height + 3.0 * (ratio - shared_height)
            maximum = model.maximum_cell_aspect * float(cells["physical_radius"][index])
            if peak_height > maximum:
                raise RuntimeError(
                    "A Plateau cell exceeds the configured aspect gate; refusing "
                    "to alter gas volume or inflate its physical footprint"
                )
            cells["base_height"][index] = shared_height
            cells["peak_height"][index] = peak_height
            cluster_id = int(cells["id"][members].min())
            cells["cluster_id"][index] = cluster_id
            cells["coalesced_group_id"][index] = cells["id"][index]
            packing = len(members) / max(
                sum(np.pi * radii[members] ** 2), 1.0e-20
            ) * np.pi * float(cells["physical_radius"][index]) ** 2
            cells["wetness"][index] = np.clip(0.25 + 0.5 * packing, 0.0, 1.0)
            radius = float(cells["physical_radius"][index])
            cells["lod_class"][index] = 0 if radius < 0.00045 else 1 if radius < 0.0015 else 2

    films = []
    events = []
    endpoint_records = []
    for first, second, edge, cluster_id in shared_edges:
        cell_a, cell_b = int(cells["id"][first]), int(cells["id"][second])
        film_id = int(_stable_id((cell_a, cell_b), 0xF11A5))
        previous = previous_film_by_id.get(film_id)
        age = 0.0 if previous is None else float(previous["age"]) + dt
        thickness = model.initial_film_thickness_m * np.exp(
            -age / model.film_drainage_seconds
        )
        if thickness <= model.minimum_film_thickness_m:
            ruptured.add(film_id)
            events.append(
                _event(
                    3,
                    film_id,
                    event_time,
                    gas_volume=float(cells["gas_volume"][first] + cells["gas_volume"][second]),
                )
            )
            continue
        if film_id in ruptured:
            continue
        origin, axis_first, axis_second, normal = frames[cluster_id]
        p0 = origin + edge[0][0] * axis_first + edge[0][1] * axis_second
        p1 = origin + edge[1][0] * axis_first + edge[1][1] * axis_second
        height = min(
            float(cells["base_height"][first]),
            float(cells["base_height"][second]),
        )
        length = float(np.linalg.norm(p1 - p0))
        area = length * height
        wetness = float(
            np.clip(
                0.5
                * (cells["wetness"][first] + cells["wetness"][second])
                * thickness
                / model.initial_film_thickness_m,
                0.0,
                1.0,
            )
        )
        radius_a = float(cells["physical_radius"][first])
        radius_b = float(cells["physical_radius"][second])
        curvature = 2.0 * (1.0 / radius_a - 1.0 / radius_b)
        films.append(
            (
                film_id,
                min(cell_a, cell_b),
                max(cell_a, cell_b),
                age,
                p0,
                p1,
                normal,
                height,
                area,
                thickness,
                area * thickness,
                curvature,
                wetness,
            )
        )
        endpoint_records.extend(
            [
                (p0, film_id, {cell_a, cell_b}, height, normal, wetness),
                (p1, film_id, {cell_a, cell_b}, height, normal, wetness),
            ]
        )
        if previous is None:
            events.append(_event(1, film_id, event_time, film_area=area))

    active_film_ids = {row[0] for row in films}
    for previous in previous_films:
        if int(previous["id"]) not in active_film_ids and int(previous["id"]) not in ruptured:
            events.append(
                _event(2, int(previous["id"]), event_time, film_area=float(previous["area"]))
            )
    film_array = np.zeros(len(films), dtype=FILM_DTYPE)
    for i, row in enumerate(films):
        for name, value in zip(FILM_DTYPE.names, row):
            film_array[name][i] = value
    if len(film_array):
        film_array = film_array[np.argsort(film_array["id"])]

    total_shared_area = float(film_array["area"].sum(dtype=np.float64))
    if total_shared_area > active_surface_film_area_m2 + 1.0e-12:
        raise RuntimeError(
            "Plateau shared membranes exceed the audited active surface-film area"
        )

    # Merge coincident power-edge endpoints into true multi-film Plateau nodes.
    groups = {}
    tolerance = model.endpoint_merge_tolerance_m
    for record in endpoint_records:
        key = tuple(np.round(record[0] / tolerance).astype(np.int64).tolist())
        groups.setdefault(key, []).append(record)
    nodes = []
    previous_node_ids = {int(row["id"]) for row in previous_nodes}
    for records in groups.values():
        incident_films = {record[1] for record in records}
        incident_cells = set().union(*(record[2] for record in records))
        if len(incident_films) < 3 or len(incident_cells) < 3:
            continue
        node_id = int(_stable_id(incident_cells, 0xA0DE))
        base = np.mean([record[0] for record in records], axis=0)
        height = min(record[3] for record in records)
        normal = _normalized(np.mean([record[4] for record in records], axis=0)[None, :])[0]
        wetness = float(np.mean([record[5] for record in records]))
        adjacent_radii = [
            float(cells["physical_radius"][np.searchsorted(cells["id"], cell_id)])
            for cell_id in incident_cells
        ]
        radius = float(
            np.clip(
                model.node_radius_scale
                * model.border_radius_fraction
                * min(adjacent_radii)
                * np.sqrt(max(wetness, 1.0e-4)),
                model.minimum_border_radius_m,
                model.maximum_border_radius_m,
            )
        )
        liquid_volume = 4.0 * np.pi * radius**3 / 3.0
        nodes.append(
            (
                node_id,
                len(incident_cells),
                len(incident_films),
                base,
                base + height * normal,
                radius,
                liquid_volume,
                wetness,
            )
        )
        if node_id not in previous_node_ids:
            events.append(_event(4, node_id, event_time))
    node_array = np.zeros(len(nodes), dtype=NODE_DTYPE)
    for i, row in enumerate(nodes):
        for name, value in zip(NODE_DTYPE.names, row):
            node_array[name][i] = value
    if len(node_array):
        node_array = node_array[np.argsort(node_array["id"])]
    current_node_ids = set(int(value) for value in node_array["id"])
    for previous_id in previous_node_ids - current_node_ids:
        events.append(_event(5, previous_id, event_time))

    borders = []
    for film in film_array:
        length = float(np.linalg.norm(film["endpoint_b"] - film["endpoint_a"]))
        cell_a = cells[np.searchsorted(cells["id"], film["cell_a"])]
        cell_b = cells[np.searchsorted(cells["id"], film["cell_b"])]
        radius = float(
            np.clip(
                model.border_radius_fraction
                * min(float(cell_a["physical_radius"]), float(cell_b["physical_radius"]))
                * np.sqrt(max(float(film["wetness"]), 1.0e-4)),
                model.minimum_border_radius_m,
                model.maximum_border_radius_m,
            )
        )
        start = film["endpoint_a"] + film["height"] * film["surface_normal"]
        end = film["endpoint_b"] + film["height"] * film["surface_normal"]
        borders.append(
            (
                _stable_id((int(film["id"]),), 0xB04DE2),
                1,
                film["id"],
                start,
                end,
                radius,
                np.pi * radius**2 * length,
                film["wetness"],
            )
        )
    for node in node_array:
        length = float(np.linalg.norm(node["top_position"] - node["base_position"]))
        borders.append(
            (
                _stable_id((int(node["id"]),), 0xB04DE3),
                2,
                node["id"],
                node["base_position"],
                node["top_position"],
                node["radius"],
                np.pi * float(node["radius"]) ** 2 * length,
                node["wetness"],
            )
        )
    border_array = np.zeros(len(borders), dtype=BORDER_DTYPE)
    for i, row in enumerate(borders):
        for name, value in zip(BORDER_DTYPE.names, row):
            border_array[name][i] = value
    if len(border_array):
        border_array = border_array[np.argsort(border_array["id"])]

    # Ruptured films create conservative coalesced rendering groups.
    cell_index = {int(value): index for index, value in enumerate(cells["id"])}
    parent = np.arange(len(cells), dtype=np.int64)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for film_id in ruptured:
        previous = previous_film_by_id.get(film_id)
        if previous is None:
            continue
        a = cell_index.get(int(previous["cell_a"]))
        b = cell_index.get(int(previous["cell_b"]))
        if a is None or b is None:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.asarray([find(i) for i in range(len(cells))], dtype=np.int64)
    for root in np.unique(roots):
        members = np.flatnonzero(roots == root)
        group_id = cells["id"][members].min()
        cells["coalesced_group_id"][members] = group_id

    geometry = build_geometry(cells, film_array, border_array, node_array, polygons, frames, model)
    event_array = np.asarray(events, dtype=EVENT_DTYPE)
    if len(event_array):
        event_array = event_array[np.argsort(event_array["id"])]
    analytic_volume = float(
        np.sum(
            cells["footprint_area"]
            * (
                cells["base_height"]
                + (cells["peak_height"] - cells["base_height"]) / 3.0
            ),
            dtype=np.float64,
        )
    )
    return {
        "cells": cells,
        "films": film_array,
        "borders": border_array,
        "nodes": node_array,
        "events": event_array,
        "ruptured_pair_ids": np.asarray(sorted(ruptured), dtype=np.uint64),
        **geometry,
    }, {
        **expansion_metrics,
        **packing_metrics,
        **sliver_metrics,
        "contact_pairs": len(pairs),
        "shared_films": len(film_array),
        "plateau_borders": len(border_array),
        "plateau_nodes": len(node_array),
        "topology_events": len(event_array),
        "ruptured_pairs": len(ruptured),
        "source_gas_volume_m3": float(raft["phase_volume"].sum(dtype=np.float64)),
        "cell_gas_volume_m3": float(cells["gas_volume"].sum(dtype=np.float64)),
        "analytic_geometry_volume_m3": analytic_volume,
        "shared_film_area_m2": total_shared_area,
        "available_surface_film_area_m2": float(active_surface_film_area_m2),
        "shared_film_liquid_volume_m3": float(
            film_array["liquid_volume"].sum(dtype=np.float64)
        ),
        "border_liquid_volume_m3": float(
            border_array["liquid_volume"].sum(dtype=np.float64)
        ),
        "node_liquid_volume_m3": float(
            node_array["liquid_volume"].sum(dtype=np.float64)
        ),
        "maximum_cell_aspect": float(
            np.max(
                cells["peak_height"]
                / np.maximum(2.0 * cells["physical_radius"], 1.0e-12)
            )
        )
        if len(cells)
        else 0.0,
    }


def _append_cylinder(vertices, triangles, materials, owners, start, end, radius, owner, sides=8):
    start = np.asarray(start, np.float64)
    end = np.asarray(end, np.float64)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-12 or radius <= 0.0:
        return
    axis /= length
    first, second = _tangent_basis(axis, int(owner))
    base = len(vertices)
    for endpoint in (start, end):
        for index in range(sides):
            angle = 2.0 * np.pi * index / sides
            vertices.append(endpoint + radius * (np.cos(angle) * first + np.sin(angle) * second))
    for index in range(sides):
        nxt = (index + 1) % sides
        triangles.extend(
            [
                (base + index, base + nxt, base + sides + nxt),
                (base + index, base + sides + nxt, base + sides + index),
            ]
        )
        materials.extend((3, 3))
        owners.extend((owner, owner))


def build_geometry(cells, films, borders, nodes, polygons, frames, model):
    vertices = []
    triangles = []
    materials = []
    owners = []
    film_by_cell = {}
    for film in films:
        for cell_id in (int(film["cell_a"]), int(film["cell_b"])):
            film_by_cell.setdefault(cell_id, []).append(film)
    for cell in cells:
        polygon = polygons[int(np.searchsorted(cells["id"], cell["id"]))]
        if len(polygon) < 3:
            continue
        origin, axis_first, axis_second, normal = frames[int(cell["cluster_id"])]
        area, centroid = _polygon_area_centroid(polygon)
        del area
        base_ring = [origin + p[0] * axis_first + p[1] * axis_second for p in polygon]
        top_ring = [p + float(cell["base_height"]) * normal for p in base_ring]
        base_center = origin + centroid[0] * axis_first + centroid[1] * axis_second
        top_center = base_center + float(cell["peak_height"]) * normal
        offset = len(vertices)
        vertices.extend(base_ring)
        vertices.extend(top_ring)
        vertices.extend((base_center, top_center))
        count = len(polygon)
        for index in range(count):
            nxt = (index + 1) % count
            triangles.append((offset + 2 * count, offset + nxt, offset + index))
            materials.append(1)
            owners.append(int(cell["id"]))
            triangles.append((offset + 2 * count + 1, offset + count + index, offset + count + nxt))
            materials.append(0)
            owners.append(int(cell["id"]))
            midpoint = 0.5 * (base_ring[index] + base_ring[nxt])
            shared = False
            for film in film_by_cell.get(int(cell["id"]), []):
                a, b = film["endpoint_a"].astype(np.float64), film["endpoint_b"].astype(np.float64)
                segment = b - a
                t = np.clip(np.dot(midpoint - a, segment) / max(np.dot(segment, segment), 1.0e-20), 0.0, 1.0)
                if np.linalg.norm(midpoint - (a + t * segment)) <= model.endpoint_merge_tolerance_m:
                    shared = True
                    break
            if not shared:
                triangles.extend(
                    [
                        (offset + index, offset + nxt, offset + count + nxt),
                        (offset + index, offset + count + nxt, offset + count + index),
                    ]
                )
                materials.extend((1, 1))
                owners.extend((int(cell["id"]), int(cell["id"])))
    for film in films:
        normal = film["surface_normal"].astype(np.float64)
        p0 = film["endpoint_a"].astype(np.float64)
        p1 = film["endpoint_b"].astype(np.float64)
        p2 = p1 + float(film["height"]) * normal
        p3 = p0 + float(film["height"]) * normal
        base = len(vertices)
        vertices.extend((p0, p1, p2, p3))
        triangles.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
        materials.extend((2, 2))
        owners.extend((int(film["id"]), int(film["id"])))
    for border in borders:
        _append_cylinder(
            vertices,
            triangles,
            materials,
            owners,
            border["start"],
            border["end"],
            float(border["radius"]),
            int(border["id"]),
        )
    for node in nodes:
        center = node["top_position"].astype(np.float64)
        radius = float(node["radius"])
        base = len(vertices)
        vertices.extend(
            (
                center + (radius, 0.0, 0.0),
                center + (-radius, 0.0, 0.0),
                center + (0.0, radius, 0.0),
                center + (0.0, -radius, 0.0),
                center + (0.0, 0.0, radius),
                center + (0.0, 0.0, -radius),
            )
        )
        faces = (
            (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
            (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
        )
        triangles.extend(tuple(base + value for value in face) for face in faces)
        materials.extend((4,) * len(faces))
        owners.extend((int(node["id"]),) * len(faces))
    return {
        "vertices": np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        "triangles": np.asarray(triangles, dtype=np.int32).reshape((-1, 3)),
        "triangle_material": np.asarray(materials, dtype=np.uint8),
        "triangle_owner_id": np.asarray(owners, dtype=np.uint64),
    }
