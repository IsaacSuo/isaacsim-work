"""Pure geometry helpers used by the soft-body collision pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class TriangleBoxCropper:
    """Crop triangles against an axis-aligned box without adding cap faces."""

    def __init__(
        self,
        bounds: Sequence[float],
        *,
        vertex_digits: int = 7,
        area_epsilon: float = 1.0e-10,
    ) -> None:
        if len(bounds) != 6:
            raise ValueError(f"Expected six collision bounds, got: {bounds}")
        self.minimum = np.asarray(bounds[:3], dtype=np.float64)
        self.maximum = np.asarray(bounds[3:], dtype=np.float64)
        if np.any(self.maximum <= self.minimum):
            raise ValueError(f"Invalid local collision bounds: {bounds}")

        self.vertex_digits = vertex_digits
        self.area_epsilon = area_epsilon
        self.points: list[np.ndarray] = []
        self.triangles: list[tuple[int, int, int]] = []
        self.clipped_source_triangles = 0
        self.rejected_degenerate = 0
        self._vertex_lookup: dict[tuple[float, float, float], int] = {}

    @staticmethod
    def _clip_polygon_to_plane(polygon, axis, limit, keep_greater):
        if not polygon:
            return []
        result = []
        previous = polygon[-1]
        previous_inside = (
            previous[axis] >= limit - 1.0e-9
            if keep_greater
            else previous[axis] <= limit + 1.0e-9
        )
        for current in polygon:
            current_inside = (
                current[axis] >= limit - 1.0e-9
                if keep_greater
                else current[axis] <= limit + 1.0e-9
            )
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                if abs(denominator) > 1.0e-15:
                    amount = (limit - previous[axis]) / denominator
                    intersection = previous + amount * (current - previous)
                    intersection[axis] = limit
                    result.append(intersection)
            if current_inside:
                result.append(current)
            previous = current
            previous_inside = current_inside
        return result

    def _crop_triangle(self, triangle):
        polygon = [point.copy() for point in triangle]
        for axis in range(3):
            polygon = self._clip_polygon_to_plane(
                polygon, axis, self.minimum[axis], True
            )
            polygon = self._clip_polygon_to_plane(
                polygon, axis, self.maximum[axis], False
            )
            if len(polygon) < 3:
                return []
        return polygon

    def _output_index(self, point) -> int:
        key = tuple(round(float(value), self.vertex_digits) for value in point)
        index = self._vertex_lookup.get(key)
        if index is None:
            index = len(self.points)
            self._vertex_lookup[key] = index
            self.points.append(np.asarray(point, dtype=np.float64))
        return index

    def add_triangle(self, triangle) -> bool:
        """Add one source triangle and return whether it contributed output."""
        triangle = np.asarray(triangle, dtype=np.float64)
        if triangle.shape != (3, 3):
            raise ValueError(f"Expected triangle shape (3, 3), got {triangle.shape}")
        if np.any(np.max(triangle, axis=0) < self.minimum) or np.any(
            np.min(triangle, axis=0) > self.maximum
        ):
            return False

        polygon = self._crop_triangle(triangle)
        if len(polygon) < 3:
            return False
        if np.any(triangle < self.minimum) or np.any(triangle > self.maximum):
            self.clipped_source_triangles += 1

        contributed = False
        for offset in range(1, len(polygon) - 1):
            cropped = (polygon[0], polygon[offset], polygon[offset + 1])
            area_twice = np.linalg.norm(
                np.cross(cropped[1] - cropped[0], cropped[2] - cropped[0])
            )
            if not np.isfinite(area_twice) or area_twice <= self.area_epsilon:
                self.rejected_degenerate += 1
                continue
            self.triangles.append(tuple(self._output_index(point) for point in cropped))
            contributed = True
        return contributed
