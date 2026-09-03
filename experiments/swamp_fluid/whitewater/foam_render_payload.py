"""Conservative multi-scale render primitives for v6 surface foam."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.spatial import cKDTree

from .liquid_fields import GridSpec
from .marker_solver import _normalized, sample_vector_trilinear
from .state_machine import splitmix64
from .surface_foam import FOAM_DTYPE, REPELLENT_DTYPE


class FoamRenderClass(IntEnum):
    MICRO_DENSITY = 0
    BUBBLE_CLUSTER = 1
    MACRO_PATCH = 2


def _sample_normals(normal_field, positions, spec):
    if hasattr(normal_field, "sample_normal"):
        return _normalized(normal_field.sample_normal(positions))
    return sample_vector_trilinear(normal_field, positions, spec)


PATCH_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("source_foam_id", "<u8"),
        ("render_class", "u1"),
        ("position", "<f4", (3,)),
        ("normal", "<f4", (3,)),
        ("principal_direction", "<f4", (3,)),
        ("major_radius", "<f4"),
        ("minor_radius", "<f4"),
        ("film_area", "<f4"),
        ("support_area", "<f4"),
        ("coverage", "<f4"),
        ("hole_fraction", "<f4"),
        ("cluster_id", "<u8"),
        ("cluster_size", "<u4"),
        ("random_key", "<u8"),
    ]
)


RING_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("source_repellent_id", "<u8"),
        ("position", "<f4", (3,)),
        ("normal", "<f4", (3,)),
        ("inner_radius", "<f4"),
        ("outer_radius", "<f4"),
        ("film_area", "<f4"),
        ("coverage", "<f4"),
        ("strength", "<f4"),
        ("host_cluster_id", "<u8"),
        ("host_patch_count", "<u4"),
        ("enclosure_fraction", "<f4"),
        ("random_key", "<u8"),
    ]
)


@dataclass(frozen=True)
class FoamRenderModel:
    micro_radius_m: float = 0.00050
    macro_radius_m: float = 0.00250
    ring_width_fraction: float = 0.35
    minimum_ring_width_m: float = 0.00020
    maximum_ring_coverage: float = 0.95
    maximum_ring_width_m: float = 0.00120
    patch_connection_scale: float = 1.20
    minimum_host_patch_count: int = 2
    minimum_host_overlap: float = 0.05
    minimum_enclosure_fraction: float = 0.50
    enclosure_samples: int = 16
    macro_single_host_scale: float = 1.25
    maximum_rim_transfer_fraction: float = 0.25

    def __post_init__(self):
        values = (
            self.micro_radius_m,
            self.macro_radius_m,
            self.ring_width_fraction,
            self.minimum_ring_width_m,
            self.maximum_ring_coverage,
            self.maximum_ring_width_m,
            self.patch_connection_scale,
            self.minimum_host_overlap,
            self.minimum_enclosure_fraction,
            self.macro_single_host_scale,
            self.maximum_rim_transfer_fraction,
        )
        if not np.isfinite(values).all() or min(values) <= 0.0:
            raise ValueError("Foam render model values must be finite and positive")
        if self.macro_radius_m <= self.micro_radius_m:
            raise ValueError("Macro radius must exceed micro radius")
        if self.maximum_ring_coverage > 1.0:
            raise ValueError("Maximum ring coverage cannot exceed one")
        if self.maximum_ring_width_m < self.minimum_ring_width_m:
            raise ValueError("Maximum ring width must exceed the minimum width")
        if self.minimum_host_patch_count < 2 or self.enclosure_samples < 8:
            raise ValueError("Hosted holes require at least two patches and eight sectors")
        if not 0.0 < self.minimum_enclosure_fraction <= 1.0:
            raise ValueError("Enclosure fraction must lie within (0, 1]")
        if not 0.0 < self.maximum_rim_transfer_fraction <= 1.0:
            raise ValueError("Rim transfer fraction must lie within (0, 1]")

    def metadata(self):
        return {
            "micro_radius_m": self.micro_radius_m,
            "macro_radius_m": self.macro_radius_m,
            "ring_width_fraction": self.ring_width_fraction,
            "minimum_ring_width_m": self.minimum_ring_width_m,
            "maximum_ring_coverage": self.maximum_ring_coverage,
            "maximum_ring_width_m": self.maximum_ring_width_m,
            "patch_connection_scale": self.patch_connection_scale,
            "minimum_host_patch_count": self.minimum_host_patch_count,
            "minimum_host_overlap": self.minimum_host_overlap,
            "minimum_enclosure_fraction": self.minimum_enclosure_fraction,
            "enclosure_samples": self.enclosure_samples,
            "macro_single_host_scale": self.macro_single_host_scale,
            "maximum_rim_transfer_fraction": self.maximum_rim_transfer_fraction,
            "classes": {
                "0": "sub-pixel micro density",
                "1": "instanced bubble cluster",
                "2": "macro anisotropic foam patch",
            },
            "hole_model": (
                "repellent holes require one connected host-film cluster and sampled "
                "perimeter enclosure; only the finite Plateau-border annulus receives "
                "film transferred from its host patches"
            ),
        }


def _stable_ids(values, salt):
    return splitmix64(
        np.asarray(values, dtype=np.uint64) ^ np.uint64(salt & ((1 << 64) - 1))
    )


def _tangent_frames(normals, directions):
    normals = _normalized(normals)
    tangent = np.asarray(directions, dtype=np.float64).copy()
    tangent -= np.sum(tangent * normals, axis=1)[:, None] * normals
    length = np.linalg.norm(tangent, axis=1)
    degenerate = length < 1.0e-7
    if np.any(degenerate):
        reference = np.tile((1.0, 0.0, 0.0), (np.count_nonzero(degenerate), 1))
        parallel = np.abs(normals[degenerate, 0]) > 0.90
        reference[parallel] = (0.0, 0.0, 1.0)
        tangent[degenerate] = reference - np.sum(
            reference * normals[degenerate], axis=1
        )[:, None] * normals[degenerate]
    tangent = _normalized(tangent)
    bitangent = _normalized(np.cross(normals, tangent))
    return normals, tangent, bitangent


def _connected_patch_clusters(positions, normals, equivalent_radius, source_ids, scale):
    """Return deterministic overlap-component IDs and sizes for foam parcels."""
    count = len(positions)
    if not count:
        return np.empty(0, np.uint64), np.empty(0, np.uint32)
    parent = np.arange(count, dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        root_first = find(int(first))
        root_second = find(int(second))
        if root_first != root_second:
            if root_first < root_second:
                parent[root_second] = root_first
            else:
                parent[root_first] = root_second

    maximum_distance = float(2.0 * scale * np.max(equivalent_radius))
    pairs = cKDTree(positions).query_pairs(maximum_distance, output_type="ndarray")
    if len(pairs):
        pairs = np.asarray(pairs, dtype=np.int64)
        distance = np.linalg.norm(
            positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1
        )
        threshold = scale * (
            equivalent_radius[pairs[:, 0]] + equivalent_radius[pairs[:, 1]]
        )
        normal_dot = np.sum(normals[pairs[:, 0]] * normals[pairs[:, 1]], axis=1)
        connected = pairs[(distance <= threshold) & (normal_dot >= 0.65)]
        for first, second in connected:
            union(first, second)
    roots = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    unique_roots, inverse, sizes = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    cluster_ids = np.empty(len(unique_roots), dtype=np.uint64)
    for index, root in enumerate(unique_roots):
        cluster_ids[index] = np.min(source_ids[roots == root])
    return cluster_ids[inverse], sizes[inverse].astype(np.uint32)


def _repellent_frame(normal):
    normal = _normalized(np.asarray(normal, dtype=np.float64)[None])[0]
    reference = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(reference @ normal)) > 0.90:
        reference = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    tangent = reference - float(reference @ normal) * normal
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)
    return normal, tangent, bitangent


def _perimeter_enclosure(
    centre,
    radius,
    ring_normal,
    host_indices,
    positions,
    normals,
    tangent,
    bitangent,
    major,
    minor,
    sample_count,
):
    """Measure how much of a proposed hole perimeter is embedded in host film."""
    _, ring_tangent, ring_bitangent = _repellent_frame(ring_normal)
    angles = 2.0 * np.pi * np.arange(sample_count, dtype=np.float64) / sample_count
    perimeter = (
        centre[None]
        + radius
        * (
            np.cos(angles)[:, None] * ring_tangent[None]
            + np.sin(angles)[:, None] * ring_bitangent[None]
        )
    )
    enclosed = np.zeros(sample_count, dtype=bool)
    for host in host_indices:
        delta = perimeter - positions[host]
        normal_distance = np.abs(delta @ normals[host])
        local_u = delta @ tangent[host]
        local_v = delta @ bitangent[host]
        inside = (
            (local_u / max(major[host], 1.0e-9)) ** 2
            + (local_v / max(minor[host], 1.0e-9)) ** 2
            <= 1.0
        )
        enclosed |= inside & (normal_distance <= max(0.001, 0.5 * radius))
    return float(np.count_nonzero(enclosed) / sample_count)


def _bounded_proportional_allocation(total, weights, capacity):
    """Allocate ``total`` without exceeding any host's remaining capacity."""
    weights = np.asarray(weights, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    target = min(float(total), float(capacity.sum(dtype=np.float64)))
    allocation = np.zeros(len(weights), dtype=np.float64)
    active = (weights > 0.0) & (capacity > 0.0)
    for _ in range(len(weights) + 1):
        remaining = target - float(allocation.sum(dtype=np.float64))
        if remaining <= 1.0e-18 or not np.any(active):
            break
        active_weights = weights[active]
        proposal = remaining * active_weights / active_weights.sum()
        ids = np.flatnonzero(active)
        room = capacity[ids] - allocation[ids]
        accepted = np.minimum(proposal, room)
        allocation[ids] += accepted
        active[ids[room <= proposal + 1.0e-18]] = False
    return allocation


def compose_foam_render_payload(
    foam,
    repellents,
    normal_field,
    spec: GridSpec,
    model=FoamRenderModel(),
):
    """Create patch/ring primitives while conserving visible film area."""

    foam = np.asarray(foam)
    repellents = np.asarray(repellents)
    if foam.dtype != FOAM_DTYPE or repellents.dtype != REPELLENT_DTYPE:
        raise ValueError("Surface foam or repellent array has the wrong dtype")
    if not len(foam):
        return (
            np.empty(0, dtype=PATCH_DTYPE),
            np.empty(0, dtype=RING_DTYPE),
            {
                "source_film_area_m2": 0.0,
                "patch_film_area_m2": 0.0,
                "ring_film_area_m2": 0.0,
                "area_balance_residual_m2": 0.0,
                "affected_patches": 0,
                "active_rings": 0,
            },
        )
    positions = foam["position"].astype(np.float64)
    normals = _sample_normals(normal_field, positions, spec)
    normals, tangent, bitangent = _tangent_frames(
        normals, foam["principal_direction"]
    )
    support = foam["support_area"].astype(np.float64)
    film = foam["film_area"].astype(np.float64)
    anisotropy = foam["anisotropy"].astype(np.float64)
    major = np.sqrt(support * anisotropy / np.pi)
    minor = np.sqrt(support / (np.pi * anisotropy))
    film_remaining = film.copy()
    hole_fraction = np.zeros(len(foam), dtype=np.float64)
    ring_area = np.zeros(len(repellents), dtype=np.float64)
    equivalent_radius = np.sqrt(support / np.pi)
    cluster_id, cluster_size = _connected_patch_clusters(
        positions,
        normals,
        equivalent_radius,
        foam["id"],
        model.patch_connection_scale,
    )
    ring_outer = np.zeros(len(repellents), dtype=np.float64)
    ring_host_cluster = np.zeros(len(repellents), dtype=np.uint64)
    ring_host_count = np.zeros(len(repellents), dtype=np.uint32)
    ring_enclosure = np.zeros(len(repellents), dtype=np.float64)
    ring_hosted_patch_rows = 0

    if len(repellents):
        rep_position = repellents["position"].astype(np.float64)
        rep_radius = repellents["radius"].astype(np.float64)
        rep_fade = np.clip(
            1.0
            - repellents["age"].astype(np.float64)
            / repellents["lifetime"].astype(np.float64),
            0.0,
            1.0,
        )
        rep_strength = repellents["strength"].astype(np.float64) * rep_fade
        rep_normal = _sample_normals(normal_field, rep_position, spec)
        patch_tree = cKDTree(positions)
        maximum_major = float(np.max(major))
        for rep_index in np.argsort(repellents["id"]):
            if rep_strength[rep_index] <= 0.0:
                continue
            candidates = np.asarray(
                patch_tree.query_ball_point(
                    rep_position[rep_index],
                    maximum_major + rep_radius[rep_index],
                ),
                dtype=np.int64,
            )
            if not len(candidates):
                continue
            delta = rep_position[rep_index] - positions[candidates]
            normal_distance = np.abs(
                np.sum(delta * normals[candidates], axis=1)
            )
            local_u = np.sum(delta * tangent[candidates], axis=1)
            local_v = np.sum(delta * bitangent[candidates], axis=1)
            scale_u = major[candidates] + rep_radius[rep_index]
            scale_v = minor[candidates] + rep_radius[rep_index]
            q = np.sqrt((local_u / scale_u) ** 2 + (local_v / scale_v) ** 2)
            surface_proximity = np.exp(
                -0.5
                * (
                    normal_distance / max(rep_radius[rep_index], 1.0e-6)
                )
                ** 2
            )
            compact = np.clip(1.0 - q, 0.0, 1.0)
            compact = compact * compact * (3.0 - 2.0 * compact)
            geometric_overlap = surface_proximity * compact
            active = geometric_overlap >= model.minimum_host_overlap
            candidates = candidates[active]
            geometric_overlap = geometric_overlap[active]
            q = q[active]
            if not len(candidates):
                continue
            candidate_clusters = np.unique(cluster_id[candidates])
            cluster_scores = np.asarray(
                [
                    np.sum(
                        geometric_overlap[cluster_id[candidates] == value]
                        * film_remaining[candidates[cluster_id[candidates] == value]],
                        dtype=np.float64,
                    )
                    for value in candidate_clusters
                ]
            )
            chosen_cluster = candidate_clusters[int(np.argmax(cluster_scores))]
            chosen = cluster_id[candidates] == chosen_cluster
            hosts = candidates[chosen]
            host_overlap = geometric_overlap[chosen]
            host_q = q[chosen]
            macro_single_host = bool(
                len(hosts) == 1
                and major[hosts[0]]
                >= model.macro_single_host_scale * rep_radius[rep_index]
                and host_q[0] <= 0.5
            )
            if len(hosts) < model.minimum_host_patch_count and not macro_single_host:
                continue
            enclosure = _perimeter_enclosure(
                rep_position[rep_index],
                rep_radius[rep_index],
                rep_normal[rep_index],
                hosts,
                positions,
                normals,
                tangent,
                bitangent,
                major,
                minor,
                model.enclosure_samples,
            )
            if enclosure < model.minimum_enclosure_fraction:
                continue
            width = np.clip(
                max(
                    model.minimum_ring_width_m,
                    model.ring_width_fraction * rep_radius[rep_index],
                ),
                model.minimum_ring_width_m,
                model.maximum_ring_width_m,
            )
            outer = rep_radius[rep_index] + width
            annulus_area = np.pi * (outer * outer - rep_radius[rep_index] ** 2)
            local_coverage = np.average(
                np.clip(film_remaining[hosts] / support[hosts], 0.0, 1.0),
                weights=np.maximum(host_overlap, 1.0e-12),
            )
            requested = (
                annulus_area
                * min(local_coverage, model.maximum_ring_coverage)
                * rep_strength[rep_index]
                * enclosure
            )
            capacity = (
                film_remaining[hosts] * model.maximum_rim_transfer_fraction
            )
            allocation = _bounded_proportional_allocation(
                requested, host_overlap, capacity
            )
            transferred = float(allocation.sum(dtype=np.float64))
            if transferred <= 0.0:
                continue
            film_remaining[hosts] -= allocation
            local_hole = np.clip(
                rep_strength[rep_index] * enclosure * host_overlap,
                0.0,
                0.95,
            )
            hole_fraction[hosts] = 1.0 - (
                1.0 - hole_fraction[hosts]
            ) * (1.0 - local_hole)
            ring_area[rep_index] = transferred
            ring_outer[rep_index] = outer
            ring_host_cluster[rep_index] = chosen_cluster
            ring_host_count[rep_index] = len(hosts)
            ring_enclosure[rep_index] = enclosure
            ring_hosted_patch_rows += len(hosts)

    patches = np.zeros(len(foam), dtype=PATCH_DTYPE)
    patches["id"] = _stable_ids(foam["id"], 0xFA7C1100)
    patches["source_foam_id"] = foam["id"]
    render_class = np.full(len(foam), FoamRenderClass.BUBBLE_CLUSTER, dtype=np.uint8)
    render_class[equivalent_radius < model.micro_radius_m] = FoamRenderClass.MICRO_DENSITY
    render_class[equivalent_radius >= model.macro_radius_m] = FoamRenderClass.MACRO_PATCH
    patches["render_class"] = render_class
    patches["position"] = positions.astype(np.float32)
    patches["normal"] = normals.astype(np.float32)
    patches["principal_direction"] = tangent.astype(np.float32)
    patches["major_radius"] = major.astype(np.float32)
    patches["minor_radius"] = minor.astype(np.float32)
    patches["film_area"] = film_remaining.astype(np.float32)
    patches["support_area"] = support.astype(np.float32)
    effective_support = support * np.maximum(1.0 - hole_fraction, 1.0e-6)
    patches["coverage"] = np.clip(
        film_remaining / effective_support, 0.0, 1.0
    ).astype(np.float32)
    patches["hole_fraction"] = hole_fraction.astype(np.float32)
    patches["cluster_id"] = cluster_id
    patches["cluster_size"] = cluster_size
    patches["random_key"] = foam["random_key"]
    if len(patches):
        patches = patches[np.argsort(patches["id"])]

    active_ring = ring_area > 0.0
    selected_repellents = repellents[active_ring]
    selected_area = ring_area[active_ring]
    rings = np.zeros(len(selected_repellents), dtype=RING_DTYPE)
    if len(rings):
        ring_position = selected_repellents["position"].astype(np.float64)
        ring_normal = _sample_normals(normal_field, ring_position, spec)
        inner = selected_repellents["radius"].astype(np.float64)
        outer = ring_outer[active_ring]
        ring_support = np.pi * (outer**2 - inner**2)
        fade = np.clip(
            1.0
            - selected_repellents["age"].astype(np.float64)
            / selected_repellents["lifetime"].astype(np.float64),
            0.0,
            1.0,
        )
        rings["id"] = _stable_ids(selected_repellents["id"], 0xA11CE551)
        rings["source_repellent_id"] = selected_repellents["id"]
        rings["position"] = ring_position.astype(np.float32)
        rings["normal"] = ring_normal.astype(np.float32)
        rings["inner_radius"] = inner.astype(np.float32)
        rings["outer_radius"] = outer.astype(np.float32)
        rings["film_area"] = selected_area.astype(np.float32)
        rings["coverage"] = np.clip(
            selected_area / ring_support, 0.0, model.maximum_ring_coverage
        ).astype(np.float32)
        rings["strength"] = (
            selected_repellents["strength"].astype(np.float64) * fade
        ).astype(np.float32)
        rings["host_cluster_id"] = ring_host_cluster[active_ring]
        rings["host_patch_count"] = ring_host_count[active_ring]
        rings["enclosure_fraction"] = ring_enclosure[active_ring].astype(
            np.float32
        )
        rings["random_key"] = selected_repellents["random_key"]
        rings = rings[np.argsort(rings["id"])]

    source_area = float(film.sum(dtype=np.float64))
    patch_area = float(patches["film_area"].sum(dtype=np.float64))
    ring_film_area = float(rings["film_area"].sum(dtype=np.float64))
    return patches, rings, {
        "source_film_area_m2": source_area,
        "patch_film_area_m2": patch_area,
        "ring_film_area_m2": ring_film_area,
        "area_balance_residual_m2": source_area - patch_area - ring_film_area,
        "affected_patches": int(np.count_nonzero(hole_fraction > 0.0)),
        "active_rings": len(rings),
        "embedded_rings": len(rings),
        "orphan_repellents": int(len(repellents) - len(rings)),
        "ring_hosted_patch_rows": int(ring_hosted_patch_rows),
        "connected_patch_clusters": int(len(np.unique(cluster_id))),
        "multi_patch_clusters": int(
            len(np.unique(cluster_id[cluster_size >= model.minimum_host_patch_count]))
        ),
        "render_class_counts": {
            "micro_density": int(
                np.count_nonzero(render_class == FoamRenderClass.MICRO_DENSITY)
            ),
            "bubble_cluster": int(
                np.count_nonzero(render_class == FoamRenderClass.BUBBLE_CLUSTER)
            ),
            "macro_patch": int(
                np.count_nonzero(render_class == FoamRenderClass.MACRO_PATCH)
            ),
        },
    }
