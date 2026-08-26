from __future__ import annotations

import unittest

import numpy as np

from soft_body.penetration import (
    exact_triangle_crossings,
    penetration_exceeds_tolerance,
    segment_triangle_intersection,
)


class PenetrationGeometryTests(unittest.TestCase):
    def test_segment_crosses_triangle(self):
        triangle = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        hit = segment_triangle_intersection((0.2, 0.2, -1), (0.2, 0.2, 1), triangle)
        np.testing.assert_allclose(hit, (0.2, 0.2, 0.0))

    def test_coplanar_segment_is_not_penetration_crossing(self):
        triangle = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        self.assertIsNone(
            segment_triangle_intersection((0.1, 0.1, 0), (0.8, 0.1, 0), triangle)
        )

    def test_exact_crossing_filters_broadphase_false_positive(self):
        first_points = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        second_points = ((0.2, 0.2, -1), (0.2, 0.2, 1), (0.8, 0.2, 0))
        result = exact_triangle_crossings(
            first_points,
            ((0, 1, 2),),
            second_points,
            ((0, 1, 2),),
            ((0, 0),),
        )
        self.assertEqual(result["candidate_triangle_pairs"], 1)
        self.assertEqual(result["exact_crossing_triangle_pairs"], 1)

    def test_depth_threshold_is_strict(self):
        self.assertFalse(penetration_exceeds_tolerance(0.001, 0.001))
        self.assertTrue(penetration_exceeds_tolerance(0.0011, 0.001))


if __name__ == "__main__":
    unittest.main()
