from __future__ import annotations

import unittest

import numpy as np

from soft_body.geometry import TriangleBoxCropper


class TriangleBoxCropperTests(unittest.TestCase):
    def test_preserves_triangle_inside_bounds(self):
        cropper = TriangleBoxCropper((0, 0, 0, 1, 1, 1))

        contributed = cropper.add_triangle(((0.1, 0.2, 0.3), (0.8, 0.2, 0.3), (0.1, 0.9, 0.3)))

        self.assertTrue(contributed)
        self.assertEqual(len(cropper.points), 3)
        self.assertEqual(len(cropper.triangles), 1)
        self.assertEqual(cropper.clipped_source_triangles, 0)

    def test_clips_crossing_triangle_without_caps(self):
        cropper = TriangleBoxCropper((0, -1, -1, 1, 1, 1))

        contributed = cropper.add_triangle(((-1, 0, 0), (2, 0, 0), (0.5, 2, 0)))

        self.assertTrue(contributed)
        self.assertEqual(cropper.clipped_source_triangles, 1)
        self.assertGreaterEqual(len(cropper.triangles), 1)
        points = np.asarray(cropper.points)
        self.assertTrue(np.all(points >= cropper.minimum - 1.0e-9))
        self.assertTrue(np.all(points <= cropper.maximum + 1.0e-9))

    def test_rejects_triangle_outside_bounds(self):
        cropper = TriangleBoxCropper((0, 0, 0, 1, 1, 1))

        contributed = cropper.add_triangle(((2, 2, 2), (3, 2, 2), (2, 3, 2)))

        self.assertFalse(contributed)
        self.assertEqual(cropper.points, [])
        self.assertEqual(cropper.triangles, [])

    def test_rejects_degenerate_triangle(self):
        cropper = TriangleBoxCropper((0, 0, 0, 1, 1, 1))

        contributed = cropper.add_triangle(((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)))

        self.assertFalse(contributed)
        self.assertEqual(cropper.rejected_degenerate, 1)

    def test_rejects_invalid_bounds(self):
        with self.assertRaises(ValueError):
            TriangleBoxCropper((0, 0, 0, 0, 1, 1))


if __name__ == "__main__":
    unittest.main()
