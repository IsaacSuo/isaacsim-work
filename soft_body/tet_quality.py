"""Numerical quality metrics for tetrahedral simulation and collision meshes."""

from __future__ import annotations

import numpy as np


EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
FACE_TRIPLES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


def signed_tetrahedron_volumes(points, indices) -> np.ndarray:
    """Return one signed volume for every tetrahedron."""
    vertices = np.asarray(points, dtype=np.float64)
    tetrahedra_indices = np.asarray(indices, dtype=np.int64).reshape(-1, 4)
    if not len(vertices) or not len(tetrahedra_indices):
        return np.empty(0, dtype=np.float64)
    tetrahedra = vertices[tetrahedra_indices]
    matrices = np.stack(
        (
            tetrahedra[:, 1] - tetrahedra[:, 0],
            tetrahedra[:, 2] - tetrahedra[:, 0],
            tetrahedra[:, 3] - tetrahedra[:, 0],
        ),
        axis=1,
    )
    return np.linalg.det(matrices) / 6.0


def tetrahedron_minimum_altitudes(points, indices) -> np.ndarray:
    """Return the smallest vertex-to-opposite-face altitude of every tet."""
    vertices = np.asarray(points, dtype=np.float64)
    tetrahedra_indices = np.asarray(indices, dtype=np.int64).reshape(-1, 4)
    if not len(vertices) or not len(tetrahedra_indices):
        return np.empty(0, dtype=np.float64)
    tetrahedra = vertices[tetrahedra_indices]
    volumes = np.abs(signed_tetrahedron_volumes(vertices, tetrahedra_indices))
    face_areas = np.stack(
        [
            0.5
            * np.linalg.norm(
                np.cross(
                    tetrahedra[:, second] - tetrahedra[:, first],
                    tetrahedra[:, third] - tetrahedra[:, first],
                ),
                axis=1,
            )
            for first, second, third in FACE_TRIPLES
        ],
        axis=1,
    )
    altitudes = 3.0 * volumes[:, None] / np.maximum(face_areas, 1.0e-18)
    return np.min(altitudes, axis=1)


def compute_tet_deformation(points, indices, bind_points) -> dict:
    """Report per-frame deformation ratios ``J = current_volume / rest_volume``."""
    vertices = np.asarray(points, dtype=np.float64)
    bind_vertices = np.asarray(bind_points, dtype=np.float64)
    tetrahedra_indices = np.asarray(indices, dtype=np.int64).reshape(-1, 4)
    if len(vertices) != len(bind_vertices):
        raise ValueError("Bind pose and current point arrays must have equal length")
    if not len(vertices) or not len(tetrahedra_indices):
        raise ValueError("Tet deformation requires non-empty points and indices")
    if tetrahedra_indices.min() < 0 or tetrahedra_indices.max() >= len(vertices):
        raise ValueError("Tetrahedron indices reference vertices outside the point array")

    current_volumes = signed_tetrahedron_volumes(vertices, tetrahedra_indices)
    rest_volumes = signed_tetrahedron_volumes(bind_vertices, tetrahedra_indices)
    valid_rest = np.abs(rest_volumes) > 1.0e-18
    ratios = np.full(len(rest_volumes), np.nan, dtype=np.float64)
    ratios[valid_rest] = current_volumes[valid_rest] / rest_volumes[valid_rest]
    if not np.any(valid_rest):
        raise ValueError("Every bind-pose tetrahedron has near-zero volume")
    minimum_tet = int(np.nanargmin(ratios))
    current_tet = vertices[tetrahedra_indices[minimum_tet]]
    rest_tet = bind_vertices[tetrahedra_indices[minimum_tet]]

    def edge_lengths(tetrahedron):
        return [
            float(np.linalg.norm(tetrahedron[first] - tetrahedron[second]))
            for first, second in EDGE_PAIRS
        ]

    rest_altitudes = tetrahedron_minimum_altitudes(
        bind_vertices, tetrahedra_indices
    )
    return {
        "minimum_j": float(ratios[minimum_tet]),
        "minimum_j_tet": minimum_tet,
        "j_below_0_3": int(np.count_nonzero(ratios < 0.3)),
        "j_below_0_2": int(np.count_nonzero(ratios < 0.2)),
        "j_below_0_1": int(np.count_nonzero(ratios < 0.1)),
        "inverted_tets": int(np.count_nonzero(ratios < 0.0)),
        "near_zero_rest_tets": int(np.count_nonzero(~valid_rest)),
        "minimum_rest_altitude_m": float(np.min(rest_altitudes[valid_rest])),
        "critical_tet_rest_altitude_m": float(rest_altitudes[minimum_tet]),
        "critical_tet_rest_volume_m3": float(abs(rest_volumes[minimum_tet])),
        "critical_tet_current_volume_m3": float(current_volumes[minimum_tet]),
        "critical_tet_rest_edge_lengths_m": edge_lengths(rest_tet),
        "critical_tet_current_edge_lengths_m": edge_lengths(current_tet),
    }


def compute_tet_quality(points, indices, *, bind_points=None) -> dict | None:
    """Summarize shape quality and optional inversion relative to a bind pose."""
    vertices = np.asarray(points, dtype=np.float64)
    tetrahedra_indices = np.asarray(indices, dtype=np.int64).reshape(-1, 4)
    if not len(vertices) or not len(tetrahedra_indices):
        return None
    if tetrahedra_indices.min() < 0 or tetrahedra_indices.max() >= len(vertices):
        raise ValueError("Tetrahedron indices reference vertices outside the point array")

    tetrahedra = vertices[tetrahedra_indices]
    signed_volumes = signed_tetrahedron_volumes(vertices, tetrahedra_indices)
    edge_lengths = np.stack(
        [
            np.linalg.norm(tetrahedra[:, first] - tetrahedra[:, second], axis=1)
            for first, second in EDGE_PAIRS
        ],
        axis=1,
    )
    maximum_edges = edge_lengths.max(axis=1)
    minimum_edges = edge_lengths.min(axis=1)
    normalized_volumes = np.abs(signed_volumes) / np.maximum(
        maximum_edges**3, 1.0e-18
    )
    aspect_ratios = maximum_edges / np.maximum(minimum_edges, 1.0e-12)
    result = {
        "tetrahedron_count": int(len(tetrahedra_indices)),
        "negative_signed_volume_count": int(np.count_nonzero(signed_volumes < 0.0)),
        "near_zero_volume_count": int(
            np.count_nonzero(np.abs(signed_volumes) < 1.0e-12)
        ),
        "absolute_volume_min_m3": float(np.min(np.abs(signed_volumes))),
        "absolute_volume_p05_m3": float(
            np.percentile(np.abs(signed_volumes), 5)
        ),
        "normalized_volume_min": float(np.min(normalized_volumes)),
        "normalized_volume_p05": float(np.percentile(normalized_volumes, 5)),
        "edge_aspect_ratio_max": float(np.max(aspect_ratios)),
        "edge_aspect_ratio_p95": float(np.percentile(aspect_ratios, 95)),
    }

    if bind_points is not None:
        bind_vertices = np.asarray(bind_points, dtype=np.float64)
        if len(bind_vertices) != len(vertices):
            raise ValueError("Bind pose and current point arrays must have equal length")
        bind_volumes = signed_tetrahedron_volumes(bind_vertices, tetrahedra_indices)
        valid_bind = np.abs(bind_volumes) > 1.0e-18
        ratios = np.full(len(bind_volumes), np.nan, dtype=np.float64)
        ratios[valid_bind] = signed_volumes[valid_bind] / bind_volumes[valid_bind]
        result["inverted_from_bind_pose_count"] = int(
            np.count_nonzero(signed_volumes * bind_volumes < 0.0)
        )
        result["minimum_signed_volume_ratio_to_bind"] = float(
            np.nanmin(ratios)
        )
    return result
