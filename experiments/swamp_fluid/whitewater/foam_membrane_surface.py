"""Conservative mesoscopic foam membranes on a retained Splashsurf surface.

The surface-foam ledger remains authoritative.  This module uses an
anisotropic density field only to choose *where* ledger area is represented;
it then copies real Splashsurf triangles until the exact eligible film area is
exhausted.  No opacity, emission or screen-space radius is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FoamMembraneModel:
    minimum_abs_up_normal: float = 0.50
    maximum_anchor_distance_m: float = 0.016
    minimum_sigma_major_m: float = 0.0040
    minimum_sigma_minor_m: float = 0.0030
    sigma_radius_scale: float = 1.75
    maximum_neighbors: int = 24
    minimum_face_alignment: float = 0.20
    target_island_area_m2: float = 0.000150
    minimum_seed_separation_m: float = 0.024
    temporal_hysteresis_radius_m: float = 0.012
    temporal_preference_strength: float = 1.0

    def __post_init__(self):
        positive = (
            self.minimum_abs_up_normal,
            self.maximum_anchor_distance_m,
            self.minimum_sigma_major_m,
            self.minimum_sigma_minor_m,
            self.sigma_radius_scale,
            self.maximum_neighbors,
            self.minimum_face_alignment,
            self.target_island_area_m2,
            self.minimum_seed_separation_m,
            self.temporal_hysteresis_radius_m,
            self.temporal_preference_strength,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0.0:
            raise ValueError("Foam membrane model values must be finite and positive")
        if self.minimum_abs_up_normal > 1.0 or self.minimum_face_alignment > 1.0:
            raise ValueError("Normal gates cannot exceed one")

    def metadata(self):
        return {
            "minimum_abs_up_normal": self.minimum_abs_up_normal,
            "maximum_anchor_distance_m": self.maximum_anchor_distance_m,
            "minimum_sigma_major_m": self.minimum_sigma_major_m,
            "minimum_sigma_minor_m": self.minimum_sigma_minor_m,
            "sigma_radius_scale": self.sigma_radius_scale,
            "maximum_neighbors": self.maximum_neighbors,
            "minimum_face_alignment": self.minimum_face_alignment,
            "target_island_area_m2": self.target_island_area_m2,
            "minimum_seed_separation_m": self.minimum_seed_separation_m,
            "temporal_hysteresis_radius_m": self.temporal_hysteresis_radius_m,
            "temporal_preference_strength": self.temporal_preference_strength,
            "selection_policy": (
                "anisotropic density ranks retained Splashsurf triangles; "
                "real triangle area is consumed exactly and the final triangle "
                "is similarity-scaled for the residual area"
            ),
        }


def load_triangle_obj(path):
    """Read the vertex/triangle subset of an OBJ without an optional dependency."""

    vertices = []
    triangles = []
    with open(path, "r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith("f "):
                values = line.split()[1:]
                face = [int(value.split("/", 1)[0]) - 1 for value in values]
                if len(face) != 3:
                    raise ValueError("Foam membrane input OBJ must be triangulated")
                triangles.append(tuple(face))
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("OBJ has no valid vertices")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,) or not len(triangles):
        raise ValueError("OBJ has no valid triangles")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("OBJ triangle index is outside the vertex array")
    return vertices, triangles


def _unit(values):
    values = np.asarray(values, dtype=np.float64)
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    fallback = np.zeros_like(values)
    fallback[..., 1] = 1.0
    return np.divide(values, length, out=fallback, where=length > 1.0e-12)


def _surface_geometry(vertices, triangles):
    points = vertices[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1.0e-12
    if not np.all(valid):
        triangles = triangles[valid]
        points = points[valid]
        cross = cross[valid]
        double_area = double_area[valid]
    return (
        triangles,
        points.mean(axis=1),
        0.5 * double_area,
        cross / double_area[:, None],
    )


def _triangle_neighbors(triangles):
    """Build a deterministic edge-neighbor table for a triangle surface."""

    triangle_count = len(triangles)
    edge_vertices = np.sort(
        triangles[:, ((0, 1), (1, 2), (2, 0))], axis=2
    ).reshape(-1, 2)
    owners = np.repeat(np.arange(triangle_count, dtype=np.int64), 3)
    slots = np.tile(np.arange(3, dtype=np.int8), triangle_count)
    order = np.lexsort((edge_vertices[:, 1], edge_vertices[:, 0]))
    edge_vertices = edge_vertices[order]
    owners = owners[order]
    slots = slots[order]
    same = np.all(edge_vertices[1:] == edge_vertices[:-1], axis=1)
    neighbors = np.full((triangle_count, 3), -1, dtype=np.int64)
    paired = np.flatnonzero(same)
    # A clean Splashsurf manifold has groups of exactly two.  Do not create an
    # arbitrary bridge when three or more faces share a malformed edge.
    unique_pair = np.ones(len(paired), dtype=bool)
    if len(paired) > 1:
        unique_pair[1:] &= paired[1:] != paired[:-1] + 1
        unique_pair[:-1] &= paired[1:] != paired[:-1] + 1
    paired = paired[unique_pair]
    first_owner = owners[paired]
    second_owner = owners[paired + 1]
    neighbors[first_owner, slots[paired]] = second_owner
    neighbors[second_owner, slots[paired + 1]] = first_owner
    return neighbors


def _density_peak_seeds(
    candidate_indices,
    density,
    centres,
    target_area,
    model,
):
    desired = max(1, int(np.ceil(target_area / model.target_island_area_m2)))
    order = np.lexsort((candidate_indices, -density))
    seeds = []
    seed_centres = []
    minimum_distance2 = model.minimum_seed_separation_m**2
    for row in order:
        centre = centres[candidate_indices[row]]
        if seed_centres:
            delta = np.asarray(seed_centres) - centre
            if float(np.min(np.sum(delta * delta, axis=1))) < minimum_distance2:
                continue
        seeds.append(int(candidate_indices[row]))
        seed_centres.append(centre)
        if len(seeds) >= desired:
            break
    return np.asarray(seeds, dtype=np.int64)


def _connected_density_order(
    candidate_indices,
    density,
    centres,
    neighbors,
    target_area,
    face_areas,
    model,
):
    candidate_mask = np.zeros(len(neighbors), dtype=bool)
    candidate_mask[candidate_indices] = True
    density_by_face = np.zeros(len(neighbors), dtype=np.float64)
    density_by_face[candidate_indices] = density
    seeds = _density_peak_seeds(
        candidate_indices, density, centres, target_area, model
    )
    selected = np.zeros(len(neighbors), dtype=bool)
    queued = np.zeros(len(neighbors), dtype=bool)
    heap = []

    def queue(face):
        face = int(face)
        if face < 0 or queued[face] or selected[face] or not candidate_mask[face]:
            return
        heapq.heappush(heap, (-float(density_by_face[face]), face))
        queued[face] = True

    for seed in seeds:
        queue(seed)
    output = []
    accumulated = 0.0
    global_order = np.lexsort((candidate_indices, -density))
    next_global = 0
    while accumulated < target_area:
        if not heap:
            while next_global < len(global_order):
                face = int(candidate_indices[global_order[next_global]])
                next_global += 1
                if not selected[face]:
                    queue(face)
                    break
            if not heap:
                break
        _, face = heapq.heappop(heap)
        if selected[face]:
            continue
        selected[face] = True
        output.append(face)
        accumulated += float(face_areas[face])
        for adjacent in neighbors[face]:
            queue(adjacent)
    if accumulated < target_area:
        raise RuntimeError("Connected membrane growth exhausted the candidate surface")
    return np.asarray(output, dtype=np.int64), seeds


def _component_metrics(source_faces, neighbors):
    selected = set(map(int, np.unique(source_faces)))
    sizes = []
    while selected:
        seed = selected.pop()
        stack = [seed]
        count = 0
        while stack:
            face = stack.pop()
            count += 1
            for adjacent in neighbors[face]:
                adjacent = int(adjacent)
                if adjacent in selected:
                    selected.remove(adjacent)
                    stack.append(adjacent)
        sizes.append(count)
    sizes = np.asarray(sizes, dtype=np.int64)
    return {
        "connected_components": int(len(sizes)),
        "single_triangle_components": int(np.count_nonzero(sizes == 1)),
        "largest_component_triangles": int(sizes.max()) if len(sizes) else 0,
        "median_component_triangles": float(np.median(sizes)) if len(sizes) else 0.0,
    }


def eligible_patch_mask(patches, macro_render_class=2, model=FoamMembraneModel()):
    """Keep top-facing non-macro parcels that are not explicit hole hosts."""

    return (
        (patches["film_area"] > 0.0)
        & (patches["hole_fraction"] <= 0.0)
        & (patches["render_class"] != macro_render_class)
        & (np.abs(patches["normal"][:, 1]) >= model.minimum_abs_up_normal)
    )


def _candidate_faces(face_tree, positions, radius, face_count):
    selected = np.zeros(face_count, dtype=bool)
    for position in positions:
        selected[face_tree.query_ball_point(position, radius)] = True
    return np.flatnonzero(selected)


def _density_and_binding(centres, face_normals, patches, model):
    positions = patches["position"].astype(np.float64)
    normals = _unit(patches["normal"])
    tangent = patches["principal_direction"].astype(np.float64)
    tangent -= np.sum(tangent * normals, axis=1)[:, None] * normals
    tangent = _unit(tangent)
    bitangent = _unit(np.cross(normals, tangent))
    sigma_major = np.maximum(
        model.minimum_sigma_major_m,
        patches["major_radius"].astype(np.float64) * model.sigma_radius_scale,
    )
    sigma_minor = np.maximum(
        model.minimum_sigma_minor_m,
        patches["minor_radius"].astype(np.float64) * model.sigma_radius_scale,
    )
    film = patches["film_area"].astype(np.float64)
    tree = cKDTree(positions)
    neighbor_count = min(model.maximum_neighbors, len(positions))
    distances, indices = tree.query(
        centres,
        k=neighbor_count,
        distance_upper_bound=model.maximum_anchor_distance_m,
    )
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    valid = indices < len(positions)
    safe = np.minimum(indices, len(positions) - 1)
    delta = centres[:, None, :] - positions[safe]
    alignment = np.abs(np.sum(face_normals[:, None, :] * normals[safe], axis=2))
    compatible = valid & (alignment >= model.minimum_face_alignment)
    du = np.sum(delta * tangent[safe], axis=2)
    dv = np.sum(delta * bitangent[safe], axis=2)
    dn = np.sum(delta * normals[safe], axis=2)
    normal_sigma = np.maximum(0.5 * sigma_minor[safe], 0.0015)
    q2 = (
        (du / sigma_major[safe]) ** 2
        + (dv / sigma_minor[safe]) ** 2
        + (dn / normal_sigma) ** 2
    )
    contribution = (
        film[safe]
        / (2.0 * np.pi * sigma_major[safe] * sigma_minor[safe])
        * np.exp(-0.5 * q2)
        * alignment**2
    )
    contribution[~compatible] = 0.0
    density = contribution.sum(axis=1, dtype=np.float64)
    compatible_distance = np.where(compatible, distances, np.inf)
    nearest_slot = np.argmin(compatible_distance, axis=1)
    row = np.arange(len(centres))
    nearest_distance = compatible_distance[row, nearest_slot]
    nearest_index = safe[row, nearest_slot]
    return density, nearest_distance, nearest_index


def _exploded_selected_geometry(
    vertices,
    triangles,
    face_normals,
    face_areas,
    ordered_faces,
    target_area,
    nearest_patch,
):
    cumulative = np.cumsum(face_areas[ordered_faces], dtype=np.float64)
    full_count = int(np.searchsorted(cumulative, target_area, side="right"))
    full_faces = ordered_faces[:full_count]
    represented = float(face_areas[full_faces].sum(dtype=np.float64))
    remaining = max(0.0, float(target_area - represented))
    output_vertices = []
    output_triangles = []
    output_normals = []
    output_source_faces = []
    output_nearest_patch = []

    def append_triangle(points, normal, source_face, patch_index):
        start = len(output_vertices)
        output_vertices.extend(points)
        output_triangles.append((start, start + 1, start + 2))
        output_normals.append(normal)
        output_source_faces.append(source_face)
        output_nearest_patch.append(patch_index)

    for face in full_faces:
        append_triangle(
            vertices[triangles[face]],
            face_normals[face],
            int(face),
            int(nearest_patch[face]),
        )
    partial_area = 0.0
    if remaining > 1.0e-14:
        if full_count >= len(ordered_faces):
            raise RuntimeError("Candidate surface area is smaller than foam film area")
        face = int(ordered_faces[full_count])
        points = vertices[triangles[face]]
        scale = np.sqrt(remaining / face_areas[face])
        centre = points.mean(axis=0)
        partial = centre + scale * (points - centre)
        append_triangle(
            partial,
            face_normals[face],
            face,
            int(nearest_patch[face]),
        )
        partial_area = remaining
    return {
        "vertices": np.asarray(output_vertices, dtype=np.float64),
        "triangles": np.asarray(output_triangles, dtype=np.int32),
        "triangle_normals": np.asarray(output_normals, dtype=np.float64),
        "source_face_indices": np.asarray(output_source_faces, dtype=np.int64),
        "nearest_eligible_patch_rows": np.asarray(output_nearest_patch, dtype=np.int32),
        "full_face_count": len(full_faces),
        "partial_face_area_m2": partial_area,
    }


def build_conservative_membrane(
    vertices,
    triangles,
    patches,
    model=FoamMembraneModel(),
    temporal_reference_centres=None,
    eligibility_mask=None,
):
    """Return a real-surface membrane and a fully explicit area/binding audit."""

    patches = np.asarray(patches)
    mask = (
        eligible_patch_mask(patches, model=model)
        if eligibility_mask is None
        else np.asarray(eligibility_mask, dtype=bool)
    )
    if mask.shape != (len(patches),):
        raise ValueError("eligibility_mask must contain one boolean per patch")
    eligible = patches[mask]
    target_area = float(eligible["film_area"].sum(dtype=np.float64))
    if not len(eligible) or target_area <= 0.0:
        raise ValueError("No eligible foam film is available for membrane reconstruction")
    triangles, centres, face_areas, face_normals = _surface_geometry(
        np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)
    )
    face_tree = cKDTree(centres)
    candidate_indices = _candidate_faces(
        face_tree,
        eligible["position"].astype(np.float64),
        model.maximum_anchor_distance_m,
        len(centres),
    )
    candidate_density, candidate_distance, candidate_nearest = _density_and_binding(
        centres[candidate_indices], face_normals[candidate_indices], eligible, model
    )
    usable = np.isfinite(candidate_distance) & (candidate_density > 0.0)
    candidate_indices = candidate_indices[usable]
    candidate_density = candidate_density[usable]
    candidate_distance = candidate_distance[usable]
    candidate_nearest = candidate_nearest[usable]
    if not len(candidate_indices):
        raise RuntimeError("No retained surface faces pass the membrane binding gate")
    selection_density = candidate_density.copy()
    temporal_metrics = {
        "enabled": temporal_reference_centres is not None,
        "reference_points": 0,
        "candidate_within_hysteresis_radius_fraction": 0.0,
        "policy": (
            "native-velocity-advected previous membrane biases current physical-density "
            "ranking only; it creates neither candidate support nor film area"
        ),
    }
    if temporal_reference_centres is not None:
        temporal_reference_centres = np.asarray(
            temporal_reference_centres, dtype=np.float64
        ).reshape((-1, 3))
        if len(temporal_reference_centres):
            temporal_distance = cKDTree(temporal_reference_centres).query(
                centres[candidate_indices], k=1
            )[0]
            temporal_weight = np.exp(
                -0.5
                * (temporal_distance / model.temporal_hysteresis_radius_m) ** 2
            )
            selection_density *= (
                1.0 + model.temporal_preference_strength * temporal_weight
            )
            temporal_metrics.update(
                {
                    "reference_points": len(temporal_reference_centres),
                    "candidate_within_hysteresis_radius_fraction": float(
                        np.mean(
                            temporal_distance <= model.temporal_hysteresis_radius_m
                        )
                    ),
                    "maximum_density_rank_multiplier": float(
                        np.max(1.0 + model.temporal_preference_strength * temporal_weight)
                    ),
                }
            )
    available_area = float(face_areas[candidate_indices].sum(dtype=np.float64))
    if available_area < target_area:
        raise RuntimeError(
            f"Candidate surface area {available_area} is smaller than target {target_area}"
        )
    neighbors = _triangle_neighbors(triangles)
    ordered_faces, seeds = _connected_density_order(
        candidate_indices,
        selection_density,
        centres,
        neighbors,
        target_area,
        face_areas,
        model,
    )
    nearest_for_all_faces = np.full(len(triangles), -1, dtype=np.int32)
    nearest_for_all_faces[candidate_indices] = candidate_nearest
    geometry = _exploded_selected_geometry(
        vertices,
        triangles,
        face_normals,
        face_areas,
        ordered_faces,
        target_area,
        nearest_for_all_faces,
    )
    # OBJ winding is not an authoritative liquid-outward convention.  Orient
    # every offset membrane triangle toward its nearest audited foam normal so
    # the film cannot be displaced underneath the water by arbitrary winding.
    nearest_rows = geometry["nearest_eligible_patch_rows"]
    source_normals = _unit(eligible["normal"][nearest_rows])
    orientation = np.sum(geometry["triangle_normals"] * source_normals, axis=1)
    geometry["triangle_normals"][orientation < 0.0] *= -1.0
    selected_source_faces = geometry["source_face_indices"]
    selected_candidate_row = {
        int(face): row for row, face in enumerate(candidate_indices)
    }
    relocation = np.asarray(
        [candidate_distance[selected_candidate_row[int(face)]] for face in selected_source_faces],
        dtype=np.float64,
    )
    represented_points = geometry["vertices"][geometry["triangles"]]
    represented_area = float(
        0.5
        * np.linalg.norm(
            np.cross(
                represented_points[:, 1] - represented_points[:, 0],
                represented_points[:, 2] - represented_points[:, 0],
            ),
            axis=1,
        ).sum(dtype=np.float64)
    )
    geometry["aggregated_source_foam_ids"] = eligible["source_foam_id"].copy()
    density_by_source_face = {
        int(face): float(density)
        for face, density in zip(candidate_indices, candidate_density)
    }
    triangle_optical_depth = np.asarray(
        [density_by_source_face[int(face)] for face in selected_source_faces],
        dtype=np.float32,
    )
    triangle_coverage = (
        1.0 - np.exp(-triangle_optical_depth.astype(np.float64))
    ).astype(np.float32)
    geometry["triangle_optical_depth"] = triangle_optical_depth
    geometry["triangle_coverage"] = triangle_coverage
    connectivity = _component_metrics(selected_source_faces, neighbors)
    metrics = {
        "source_patch_rows": len(patches),
        "eligible_patch_rows": len(eligible),
        "residual_explicit_patch_rows": int(len(patches) - len(eligible)),
        "target_membrane_area_m2": target_area,
        "represented_membrane_area_m2": represented_area,
        "area_residual_m2": target_area - represented_area,
        "candidate_face_count": len(candidate_indices),
        "candidate_area_m2": available_area,
        "selected_triangle_count": len(geometry["triangles"]),
        "selected_full_face_count": geometry.pop("full_face_count"),
        "partial_face_area_m2": geometry.pop("partial_face_area_m2"),
        "density_seed_count": len(seeds),
        "connectivity": connectivity,
        "relocation_distance_m": {
            "maximum": float(relocation.max()),
            "p50": float(np.percentile(relocation, 50)),
            "p95": float(np.percentile(relocation, 95)),
            "gate": model.maximum_anchor_distance_m,
        },
        "optical_depth": {
            "minimum": float(triangle_optical_depth.min()),
            "median": float(np.median(triangle_optical_depth)),
            "maximum": float(triangle_optical_depth.max()),
            "coverage_minimum": float(triangle_coverage.min()),
            "coverage_median": float(np.median(triangle_coverage)),
            "coverage_maximum": float(triangle_coverage.max()),
            "policy": "coverage=1-exp(-local anisotropic parcel optical depth)",
        },
        "temporal_hysteresis": temporal_metrics,
    }
    return geometry, metrics
