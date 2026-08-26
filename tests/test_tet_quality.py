from __future__ import annotations

import unittest

import numpy as np

from soft_body.tet_quality import compute_tet_quality


class TetQualityTests(unittest.TestCase):
    def setUp(self):
        self.points = np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            dtype=np.float64,
        )
        self.indices = np.asarray(((0, 1, 2, 3),), dtype=np.int64)

    def test_reports_regular_tetrahedron(self):
        report = compute_tet_quality(
            self.points, self.indices, bind_points=self.points
        )

        self.assertEqual(report["tetrahedron_count"], 1)
        self.assertEqual(report["inverted_from_bind_pose_count"], 0)
        self.assertAlmostEqual(report["minimum_signed_volume_ratio_to_bind"], 1.0)

    def test_detects_inversion_relative_to_bind_pose(self):
        current = self.points.copy()
        current[3, 2] = -1.0

        report = compute_tet_quality(
            current, self.indices, bind_points=self.points
        )

        self.assertEqual(report["inverted_from_bind_pose_count"], 1)
        self.assertLess(report["minimum_signed_volume_ratio_to_bind"], 0.0)

    def test_rejects_invalid_indices(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            compute_tet_quality(self.points, ((0, 1, 2, 9),))


if __name__ == "__main__":
    unittest.main()
