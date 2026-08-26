"""Pure triangle-surface penetration helpers used by offline auditors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


TRIANGLE_EDGES = ((0, 1), (1, 2), (2, 0))


def segment_triangle_intersection(
    start,
    end,
    triangle,
    *,
    epsilon: float = 1.0e-9,
) -> np.ndarray | None:
    """Return a segment/triangle hit using the Möller–Trumbore test.

    Coplanar segments deliberately return ``None``. Coplanar contact is not
    evidence of volume penetration and is handled by the separate depth test.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    triangle = np.asarray(triangle, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,) or triangle.shape != (3, 3):
        raise ValueError("Expected 3D segment endpoints and one (3, 3) triangle")

    direction = end - start
    edge_one = triangle[1] - triangle[0]
    edge_two = triangle[2] - triangle[0]
    cross = np.cross(direction, edge_two)
    determinant = float(np.dot(edge_one, cross))
    if abs(determinant) <= epsilon:
        return None
    inverse = 1.0 / determinant
    offset = start - triangle[0]
    barycentric_u = inverse * float(np.dot(offset, cross))
    if barycentric_u < -epsilon or barycentric_u > 1.0 + epsilon:
        return None
    offset_cross = np.cross(offset, edge_one)
    barycentric_v = inverse * float(np.dot(direction, offset_cross))
    if barycentric_v < -epsilon or barycentric_u + barycentric_v > 1.0 + epsilon:
        return None
    segment_amount = inverse * float(np.dot(edge_two, offset_cross))
    if segment_amount < -epsilon or segment_amount > 1.0 + epsilon:
        return None
    return start + np.clip(segment_amount, 0.0, 1.0) * direction


def exact_triangle_crossings(
    first_points,
    first_triangles,
    second_points,
    second_triangles,
    candidate_pairs: Iterable[tuple[int, int]],
    *,
    epsilon: float = 1.0e-9,
    point_digits: int = 7,
    maximum_reported_points: int = 100,
) -> dict:
    """Verify candidate triangle pairs by testing all six triangle edges."""
    first_points = np.asarray(first_points, dtype=np.float64)
    second_points = np.asarray(second_points, dtype=np.float64)
    first_triangles = np.asarray(first_triangles, dtype=np.int64).reshape(-1, 3)
    second_triangles = np.asarray(second_triangles, dtype=np.int64).reshape(-1, 3)
    candidates = [(int(first), int(second)) for first, second in candidate_pairs]
    exact_pairs: set[tuple[int, int]] = set()
    hit_points: set[tuple[float, float, float]] = set()

    for first_index, second_index in candidates:
        first_triangle = first_points[first_triangles[first_index]]
        second_triangle = second_points[second_triangles[second_index]]
        pair_hit = False
        for source, target in (
            (first_triangle, second_triangle),
            (second_triangle, first_triangle),
        ):
            for start_index, end_index in TRIANGLE_EDGES:
                hit = segment_triangle_intersection(
                    source[start_index],
                    source[end_index],
                    target,
                    epsilon=epsilon,
                )
                if hit is not None:
                    pair_hit = True
                    hit_points.add(
                        tuple(round(float(value), point_digits) for value in hit)
                    )
        if pair_hit:
            exact_pairs.add((first_index, second_index))

    return {
        "candidate_triangle_pairs": len(candidates),
        "exact_crossing_triangle_pairs": len(exact_pairs),
        "unique_crossing_points": len(hit_points),
        "crossing_points": [
            list(point)
            for point in sorted(hit_points)[:maximum_reported_points]
        ],
    }


def penetration_exceeds_tolerance(
    maximum_depth_m: float,
    tolerance_m: float,
) -> bool:
    """Return whether measured depth is a material penetration, not contact noise."""
    if tolerance_m < 0.0:
        raise ValueError("Penetration tolerance must be non-negative")
    return float(maximum_depth_m) > float(tolerance_m)
