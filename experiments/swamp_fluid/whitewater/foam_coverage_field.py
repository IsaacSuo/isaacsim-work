"""Area-normalized, surface-advected FoamGenerator render mask prototype.

The field is an optical render proxy, not physical soap-film area or gas
volume. Current surface parcels deposit dimensionless optical measure through
kernels normalized by actual triangle area. Previous face optical measure
(``tau * face_area``) is moved with native tangential carrier velocity,
projected onto the current exact surface, and conservatively remapped before
temporal assimilation.

The output is a scalar attribute on the complete liquid mesh. It never selects
triangles to create a solid membrane and never derives velocity from position
finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


COVERAGE_SOURCE_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("source_row", "<i4"),
        ("position", "<f4", (3,)),
        ("normal", "<f4", (3,)),
        ("native_velocity", "<f4", (3,)),
        ("velocity", "<f4", (3,)),
        ("opacity", "<f4"),
        ("remaining_lifetime", "<f4"),
        ("kernel_support_radius", "<f4"),
        ("instantaneous_target_optical_depth", "<f4"),
        ("target_optical_depth", "<f4"),
        ("anchor_face_index", "<i8"),
        ("anchor_authority", "u1"),
    ]
)


@dataclass(frozen=True)
class FoamCoverageFieldModel:
    kernel_support_radius_m: float = 0.008
    minimum_face_alignment: float = 0.20
    minimum_relative_kernel_weight: float = 1.0e-4
    maximum_faces_per_source: int = 384
    history_weight: float = 0.68
    history_decay_seconds: float = 2.5
    history_remap_radius_m: float = 0.006
    history_max_projection_distance_m: float = 0.016
    history_minimum_normal_alignment: float = 0.10

    def __post_init__(self):
        positive = (
            self.kernel_support_radius_m,
            self.minimum_face_alignment,
            self.minimum_relative_kernel_weight,
            self.maximum_faces_per_source,
            self.history_decay_seconds,
            self.history_remap_radius_m,
            self.history_max_projection_distance_m,
            self.history_minimum_normal_alignment,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0.0:
            raise ValueError("Coverage-field parameters must be finite and positive")
        if max(
            self.minimum_face_alignment, self.history_minimum_normal_alignment
        ) > 1.0:
            raise ValueError("Normal-alignment thresholds cannot exceed one")
        if not 0.0 <= self.history_weight <= 1.0:
            raise ValueError("history_weight must be within [0, 1]")

    def metadata(self):
        return {
            "maturity": "water-surface foam mask prototype",
            "quantity": "dimensionless render optical depth and coverage",
            "physical_area_or_gas_volume_claimed": False,
            "kernel_support_radius_m": self.kernel_support_radius_m,
            "kernel_support_authority": "render-only calibration",
            "source_normalization": (
                "kernel times actual face area; exact optical-measure closure per source"
            ),
            "history_quantity": "face optical depth times actual face area",
            "history_transport": (
                "native tangential carrier velocity, exact surface projection, "
                "area-normalized conservative remap"
            ),
            "history_weight": self.history_weight,
            "history_decay_seconds": self.history_decay_seconds,
            "history_remap_radius_m": self.history_remap_radius_m,
            "history_max_projection_distance_m": (
                self.history_max_projection_distance_m
            ),
            "history_minimum_normal_alignment": (
                self.history_minimum_normal_alignment
            ),
            "minimum_face_alignment": self.minimum_face_alignment,
            "minimum_relative_kernel_weight": self.minimum_relative_kernel_weight,
            "maximum_faces_per_source": self.maximum_faces_per_source,
            "finite_difference_velocity_used": False,
            "face_identity_transport_used": False,
            "geometry_policy": (
                "full liquid surface plus scalar attributes; no selected solid membrane"
            ),
        }


def coverage_to_optical_depth(coverage):
    coverage = np.asarray(coverage, dtype=np.float64)
    if np.any(~np.isfinite(coverage)) or np.any(
        (coverage < 0.0) | (coverage >= 1.0)
    ):
        raise ValueError("Coverage must be finite and within [0, 1)")
    return -np.log1p(-coverage)


def kernel_integral_area(radius):
    """Analytic area of exp(-4.5 (r/R)^2) over a radius-R disk."""
    radius = np.asarray(radius, dtype=np.float64)
    return np.pi * radius**2 * (1.0 - np.exp(-4.5)) / 4.5


def _unit(values):
    values = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    fallback = np.zeros_like(values)
    fallback[..., 1] = 1.0
    return np.divide(values, lengths, out=fallback, where=lengths > 1.0e-12)


def _surface_geometry(vertices, triangles):
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3)")
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("triangles contain invalid vertex indices")
    points = vertices[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    if np.any(double_area <= 1.0e-14):
        raise ValueError("Coverage field does not accept degenerate triangles")
    return points.mean(axis=1), 0.5 * double_area, cross / double_area[:, None]


def _kernel_candidates(
    position,
    normal,
    anchor_face,
    radius,
    centres,
    face_area,
    face_normals,
    face_tree,
    minimum_alignment,
    minimum_relative_weight,
    maximum_faces,
):
    if anchor_face < 0 or anchor_face >= len(centres):
        raise ValueError("anchor face is outside the current surface")
    candidates = np.asarray(face_tree.query_ball_point(position, radius), dtype=np.int64)
    candidates = np.unique(np.append(candidates, anchor_face))
    alignment = np.abs(face_normals[candidates] @ _unit(normal[None])[0])
    distance = np.linalg.norm(centres[candidates] - position, axis=1)
    kernel = np.exp(-4.5 * (distance / radius) ** 2) * alignment**2
    kernel[alignment < minimum_alignment] = 0.0
    maximum = float(kernel.max(initial=0.0))
    if maximum <= 0.0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    keep = kernel >= maximum * minimum_relative_weight
    candidates = candidates[keep]
    kernel = kernel[keep]
    if len(kernel) > maximum_faces:
        order = np.lexsort((candidates, -kernel))[:maximum_faces]
        candidates = candidates[order]
        kernel = kernel[order]
    # This is the key tessellation-invariance rule: quadrature is weighted by
    # physical triangle area, never by triangle count or peak value.
    measure_weight = kernel * face_area[candidates]
    total = float(measure_weight.sum(dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return candidates, measure_weight / total


def _deposit_current_sources(
    centres, face_area, face_normals, sources, model, face_tree
):
    face_measure = np.zeros(len(centres), dtype=np.float64)
    velocity_numerator = np.zeros((len(centres), 3), dtype=np.float64)
    source_hits = np.zeros(len(sources), dtype=np.int32)
    target_measure = np.zeros(len(sources), dtype=np.float64)
    allocated_measure = np.zeros(len(sources), dtype=np.float64)
    if len(np.unique(sources["id"])) != len(sources):
        raise ValueError("Coverage source IDs must be unique")
    for source_row, source in enumerate(sources):
        target_tau = float(source["target_optical_depth"])
        radius = float(source["kernel_support_radius"])
        if not np.isfinite(target_tau) or target_tau < 0.0:
            raise ValueError("target_optical_depth must be finite and non-negative")
        if not np.isfinite(radius) or radius <= 0.0:
            radius = model.kernel_support_radius_m
        measure = target_tau * float(kernel_integral_area(radius))
        target_measure[source_row] = measure
        if measure == 0.0:
            continue
        candidates, normalized = _kernel_candidates(
            np.asarray(source["position"], dtype=np.float64),
            np.asarray(source["normal"], dtype=np.float64),
            int(source["anchor_face_index"]),
            radius,
            centres,
            face_area,
            face_normals,
            face_tree,
            model.minimum_face_alignment,
            model.minimum_relative_kernel_weight,
            model.maximum_faces_per_source,
        )
        if not len(candidates):
            continue
        allocation = measure * normalized
        allocation[-1] += measure - float(allocation.sum(dtype=np.float64))
        np.add.at(face_measure, candidates, allocation)
        for axis in range(3):
            np.add.at(
                velocity_numerator[:, axis],
                candidates,
                allocation * float(source["velocity"][axis]),
            )
        source_hits[source_row] = len(candidates)
        allocated_measure[source_row] = float(allocation.sum(dtype=np.float64))
    return (
        face_measure,
        velocity_numerator,
        source_hits,
        target_measure,
        allocated_measure,
    )


def _empty_history_metrics():
    return {
        "input_optical_measure_m2": 0.0,
        "accepted_optical_measure_m2": 0.0,
        "rejected_optical_measure_m2": 0.0,
        "allocated_optical_measure_m2": 0.0,
        "maximum_allocation_residual_m2": 0.0,
        "input_samples": 0,
        "accepted_samples": 0,
        "rejected_distance_samples": 0,
        "rejected_normal_samples": 0,
    }


def _remap_history(
    centres,
    face_area,
    face_normals,
    previous,
    projection,
    dt,
    model,
    face_tree,
):
    face_measure = np.zeros(len(centres), dtype=np.float64)
    velocity_numerator = np.zeros((len(centres), 3), dtype=np.float64)
    if previous is None:
        return face_measure, velocity_numerator, _empty_history_metrics()
    if projection is None:
        raise ValueError("History projection is required when previous field exists")
    previous_tau = np.asarray(previous["active_face_optical_depth"], dtype=np.float64)
    previous_area = np.asarray(previous["active_face_area"], dtype=np.float64)
    previous_velocity = np.asarray(previous["active_face_velocity"], dtype=np.float64)
    previous_normal = _unit(previous["active_face_normal"])
    count = len(previous_tau)
    position = np.asarray(projection["position"], dtype=np.float64)
    normal = _unit(projection["normal"])
    face_index = np.asarray(projection["face_index"], dtype=np.int64)
    distance = np.asarray(projection["distance"], dtype=np.float64)
    if (
        previous_area.shape != (count,)
        or previous_velocity.shape != (count, 3)
        or previous_normal.shape != (count, 3)
        or position.shape != (count, 3)
        or normal.shape != (count, 3)
        or face_index.shape != (count,)
        or distance.shape != (count,)
    ):
        raise ValueError("History projection arrays have inconsistent shapes")
    alignment = np.abs(np.sum(previous_normal * normal, axis=1))
    distance_valid = distance <= model.history_max_projection_distance_m
    normal_valid = alignment >= model.history_minimum_normal_alignment
    accepted = distance_valid & normal_valid
    decayed_input = (
        previous_tau
        * previous_area
        * np.exp(-float(dt) / model.history_decay_seconds)
    )
    maximum_residual = 0.0
    for row in np.flatnonzero(accepted):
        measure = float(decayed_input[row])
        candidates, normalized = _kernel_candidates(
            position[row],
            normal[row],
            int(face_index[row]),
            model.history_remap_radius_m,
            centres,
            face_area,
            face_normals,
            face_tree,
            model.history_minimum_normal_alignment,
            model.minimum_relative_kernel_weight,
            model.maximum_faces_per_source,
        )
        if not len(candidates):
            accepted[row] = False
            continue
        allocation = measure * normalized
        allocation[-1] += measure - float(allocation.sum(dtype=np.float64))
        np.add.at(face_measure, candidates, allocation)
        for axis in range(3):
            np.add.at(
                velocity_numerator[:, axis],
                candidates,
                allocation * previous_velocity[row, axis],
            )
        maximum_residual = max(
            maximum_residual,
            abs(measure - float(allocation.sum(dtype=np.float64))),
        )
    total_input = float(decayed_input.sum(dtype=np.float64))
    allocated = float(face_measure.sum(dtype=np.float64))
    return face_measure, velocity_numerator, {
        "input_optical_measure_m2": total_input,
        "accepted_optical_measure_m2": allocated,
        "rejected_optical_measure_m2": total_input - allocated,
        "allocated_optical_measure_m2": allocated,
        "maximum_allocation_residual_m2": maximum_residual,
        "input_samples": int(count),
        "accepted_samples": int(np.count_nonzero(accepted)),
        "rejected_distance_samples": int(np.count_nonzero(~distance_valid)),
        "rejected_normal_samples": int(
            np.count_nonzero(distance_valid & ~normal_valid)
        ),
    }


def rasterize_advected_coverage_field(
    vertices,
    triangles,
    sources,
    previous=None,
    history_projection=None,
    dt=1.0 / 30.0,
    model=FoamCoverageFieldModel(),
):
    """Transport history and assimilate current sources on the full surface."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    sources = np.asarray(sources, dtype=COVERAGE_SOURCE_DTYPE)
    centres, face_area, face_normals = _surface_geometry(vertices, triangles)
    face_tree = cKDTree(centres)
    (
        current_measure,
        current_velocity_numerator,
        source_hits,
        source_target_measure,
        source_allocated_measure,
    ) = _deposit_current_sources(
        centres, face_area, face_normals, sources, model, face_tree
    )
    history_measure, history_velocity_numerator, history_metrics = _remap_history(
        centres,
        face_area,
        face_normals,
        previous,
        history_projection,
        dt,
        model,
        face_tree,
    )
    current_tau = np.divide(
        current_measure,
        face_area,
        out=np.zeros_like(current_measure),
        where=face_area > 0.0,
    )
    history_tau = np.divide(
        history_measure,
        face_area,
        out=np.zeros_like(history_measure),
        where=face_area > 0.0,
    )
    current_velocity = np.divide(
        current_velocity_numerator,
        current_measure[:, None],
        out=np.zeros_like(current_velocity_numerator),
        where=current_measure[:, None] > 0.0,
    )
    history_velocity = np.divide(
        history_velocity_numerator,
        history_measure[:, None],
        out=np.zeros_like(history_velocity_numerator),
        where=history_measure[:, None] > 0.0,
    )
    has_current = current_measure > 0.0
    has_history = history_measure > 0.0
    both = has_current & has_history
    only_current = has_current & ~has_history
    only_history = has_history & ~has_current
    optical_depth = np.zeros(len(centres), dtype=np.float64)
    velocity = np.zeros((len(centres), 3), dtype=np.float64)
    optical_depth[both] = (
        model.history_weight * history_tau[both]
        + (1.0 - model.history_weight) * current_tau[both]
    )
    velocity[both] = (
        model.history_weight * history_velocity[both]
        + (1.0 - model.history_weight) * current_velocity[both]
    )
    optical_depth[only_current] = current_tau[only_current]
    velocity[only_current] = current_velocity[only_current]
    # Applying the temporal weight to unsupported history prevents a long
    # residual trail even though the render calibration decay is 2.5 seconds.
    optical_depth[only_history] = model.history_weight * history_tau[only_history]
    velocity[only_history] = history_velocity[only_history]

    # Sparse support is already bounded by the finite kernels.  Do not apply a
    # second, implicit tau cutoff here: it silently discards remapped optical
    # measure and makes transport depend on tessellation and accumulation order.
    active = np.flatnonzero(has_current | has_history)
    coverage = -np.expm1(-optical_depth[active])
    field = {
        "active_face_indices": active.astype(np.int32),
        "active_face_centres": centres[active].astype(np.float32),
        "active_face_area": face_area[active],
        # Keep tau in float64 because active_face_optical_measure is the exact
        # serialized product tau * actual triangle area.  Down-casting tau here
        # breaks that cache invariant even though the visual error is tiny.
        "active_face_optical_depth": optical_depth[active],
        "active_face_optical_measure": optical_depth[active] * face_area[active],
        "active_face_coverage": coverage.astype(np.float32),
        "active_face_velocity": velocity[active].astype(np.float32),
        "active_face_normal": face_normals[active].astype(np.float32),
        "active_face_current_optical_measure": current_measure[active],
        "active_face_history_optical_measure": history_measure[active],
        "source_id": np.asarray(sources["id"], dtype=np.uint64),
        "source_face_hit_count": source_hits,
        "source_target_optical_measure": source_target_measure,
        "source_allocated_optical_measure": source_allocated_measure,
    }
    source_residual = np.abs(source_target_measure - source_allocated_measure)
    metrics = {
        "source_count": int(len(sources)),
        "source_with_surface_support_count": int(np.count_nonzero(source_hits)),
        "source_without_surface_support_count": int(np.count_nonzero(source_hits == 0)),
        "source_target_optical_measure_m2": float(
            source_target_measure.sum(dtype=np.float64)
        ),
        "source_allocated_optical_measure_m2": float(
            source_allocated_measure.sum(dtype=np.float64)
        ),
        "source_maximum_allocation_residual_m2": float(
            source_residual.max(initial=0.0)
        ),
        "history": history_metrics,
        "active_face_count": int(len(active)),
        "active_surface_area_m2": float(face_area[active].sum(dtype=np.float64)),
        "coverage_integral_m2": float(
            np.sum(face_area[active] * coverage, dtype=np.float64)
        ),
        "coverage_is_render_proxy_not_physical_area": True,
        "actual_face_area_normalization_used": True,
        "surface_field_history_advected": previous is not None,
        "surface_transport_velocity_is_native_tangent": True,
        "finite_difference_velocity_used": False,
        "optical_depth": {
            "median": float(np.median(optical_depth[active])) if len(active) else 0.0,
            "p90": float(np.quantile(optical_depth[active], 0.90)) if len(active) else 0.0,
            "p99": float(np.quantile(optical_depth[active], 0.99)) if len(active) else 0.0,
            "maximum": float(optical_depth[active].max(initial=0.0)),
        },
    }
    return field, metrics
