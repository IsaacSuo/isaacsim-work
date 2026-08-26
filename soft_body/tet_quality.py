"""Numerical quality metrics for tetrahedral simulation and collision meshes."""

from __future__ import annotations

import numpy as np


EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


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
