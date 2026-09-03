"""Synthetic gates for area normalization and true surface-field transport."""

import json

import numpy as np
from scipy.spatial import cKDTree

from whitewater.foam_coverage_field import (
    COVERAGE_SOURCE_DTYPE,
    FoamCoverageFieldModel,
    coverage_to_optical_depth,
    rasterize_advected_coverage_field,
)


def plane_grid(cells):
    axis = np.linspace(-0.03, 0.03, cells + 1)
    vertices = np.asarray(
        [(x, 0.0, z) for z in axis for x in axis], dtype=np.float64
    )
    triangles = []
    width = cells + 1
    for z in range(cells):
        for x in range(cells):
            lower = z * width + x
            triangles.extend(
                (
                    (lower, lower + width + 1, lower + 1),
                    (lower, lower + width, lower + width + 1),
                )
            )
    return vertices, np.asarray(triangles, dtype=np.int64)


def face_centres(vertices, triangles):
    return vertices[triangles].mean(axis=1)


def source_for_surface(vertices, triangles):
    sources = np.zeros(1, dtype=COVERAGE_SOURCE_DTYPE)
    sources["id"] = 11
    sources["source_row"] = 0
    sources["position"] = (0.0, 0.0, 0.0)
    sources["normal"] = (0.0, 1.0, 0.0)
    sources["native_velocity"] = (0.06, 0.02, 0.0)
    sources["velocity"] = (0.06, 0.0, 0.0)
    sources["opacity"] = 1.0
    sources["remaining_lifetime"] = 2.0
    sources["kernel_support_radius"] = 0.018
    sources["instantaneous_target_optical_depth"] = coverage_to_optical_depth(0.72)
    sources["target_optical_depth"] = sources[
        "instantaneous_target_optical_depth"
    ]
    sources["anchor_face_index"] = int(
        cKDTree(face_centres(vertices, triangles)).query((0.0, 0.0, 0.0))[1]
    )
    sources["anchor_authority"] = 0
    return sources


model = FoamCoverageFieldModel(
    kernel_support_radius_m=0.018,
    history_remap_radius_m=0.009,
    history_max_projection_distance_m=0.016,
)
coarse_vertices, coarse_triangles = plane_grid(8)
fine_vertices, fine_triangles = plane_grid(16)
coarse_sources = source_for_surface(coarse_vertices, coarse_triangles)
fine_sources = source_for_surface(fine_vertices, fine_triangles)
coarse, coarse_metrics = rasterize_advected_coverage_field(
    coarse_vertices, coarse_triangles, coarse_sources, model=model
)
fine, fine_metrics = rasterize_advected_coverage_field(
    fine_vertices, fine_triangles, fine_sources, model=model
)

# The optical measure must close exactly and remain invariant when the same
# physical patch is tessellated four times more densely.
for metrics in (coarse_metrics, fine_metrics):
    assert metrics["source_maximum_allocation_residual_m2"] <= 1.0e-18
    assert metrics["actual_face_area_normalization_used"] is True
assert np.isclose(
    coarse_metrics["source_allocated_optical_measure_m2"],
    fine_metrics["source_allocated_optical_measure_m2"],
    rtol=0.0,
    atol=1.0e-18,
)
coverage_integral_relative_change = abs(
    fine_metrics["coverage_integral_m2"] - coarse_metrics["coverage_integral_m2"]
) / coarse_metrics["coverage_integral_m2"]
assert coverage_integral_relative_change <= 0.08

# Transport the coarse field to a differently tessellated current surface.
predicted = (
    coarse["active_face_centres"].astype(np.float64)
    + coarse["active_face_velocity"].astype(np.float64) / 30.0
)
fine_centres = face_centres(fine_vertices, fine_triangles)
projected_face = cKDTree(fine_centres).query(predicted)[1]
projection = {
    "position": predicted,
    "normal": np.repeat(((0.0, 1.0, 0.0),), len(predicted), axis=0),
    "face_index": projected_face,
    "distance": np.zeros(len(predicted), dtype=np.float64),
}
transported, transported_metrics = rasterize_advected_coverage_field(
    fine_vertices,
    fine_triangles,
    np.empty(0, dtype=COVERAGE_SOURCE_DTYPE),
    previous=coarse,
    history_projection=projection,
    dt=1.0 / 30.0,
    model=model,
)
history = transported_metrics["history"]
assert transported_metrics["surface_field_history_advected"] is True
assert history["accepted_samples"] == history["input_samples"]
assert history["maximum_allocation_residual_m2"] <= 1.0e-18
assert np.isclose(
    history["accepted_optical_measure_m2"],
    history["allocated_optical_measure_m2"],
    rtol=0.0,
    atol=1.0e-18,
)
assert len(transported["active_face_indices"]) > 0

# A history sample beyond the hard projection distance must not leave a ghost.
far_projection = dict(projection)
far_projection["distance"] = np.full(
    len(predicted), model.history_max_projection_distance_m + 0.001
)
rejected, rejected_metrics = rasterize_advected_coverage_field(
    fine_vertices,
    fine_triangles,
    np.empty(0, dtype=COVERAGE_SOURCE_DTYPE),
    previous=coarse,
    history_projection=far_projection,
    dt=1.0 / 30.0,
    model=model,
)
assert rejected_metrics["history"]["accepted_samples"] == 0
assert rejected_metrics["history"]["rejected_distance_samples"] == len(predicted)
assert len(rejected["active_face_indices"]) == 0

print(
    json.dumps(
        {
            "valid": True,
            "coverage_integral_relative_change": coverage_integral_relative_change,
            "coarse": coarse_metrics,
            "fine": fine_metrics,
            "transported": transported_metrics,
            "rejected": rejected_metrics,
        },
        indent=2,
    )
)
