from __future__ import annotations

import unittest

import numpy as np

from soft_body.tet_quality import (
    compute_tet_deformation,
    compute_tet_quality,
    compute_tet_surface_topology,
    compute_tet_volume_topology,
    tetrahedron_minimum_altitudes,
)


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

        deformation = compute_tet_deformation(
            current, self.indices, self.points
        )
        self.assertEqual(deformation["minimum_j_tet"], 0)
        self.assertEqual(deformation["inverted_tets"], 1)
        self.assertAlmostEqual(deformation["minimum_j"], -1.0)

    def test_computes_minimum_altitude(self):
        altitudes = tetrahedron_minimum_altitudes(self.points, self.indices)
        self.assertEqual(len(altitudes), 1)
        self.assertGreater(altitudes[0], 0.0)

    def test_internal_shared_tet_face_is_not_non_manifold_boundary(self):
        faces = (
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (0, 1, 2),
            (0, 1, 4),
            (0, 2, 4),
            (1, 2, 4),
        )
        topology = compute_tet_surface_topology(faces)
        self.assertEqual(topology["internal_shared_face_count"], 1)
        self.assertEqual(topology["boundary_triangle_count"], 6)
        self.assertEqual(topology["non_manifold_boundary_edges"], 0)
        self.assertEqual(topology["faces_with_more_than_two_incident_tets"], 0)

        volume_topology = compute_tet_volume_topology(
            ((0, 1, 2, 3), (0, 2, 1, 4))
        )
        self.assertEqual(volume_topology["internal_shared_face_count"], 1)
        self.assertEqual(volume_topology["boundary_triangle_count"], 6)
        self.assertEqual(volume_topology["non_manifold_boundary_edges"], 0)

    def test_internal_face_with_duplicate_vertex_ids_is_welded(self):
        points = np.asarray(
            (
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (0, 0, -1),
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
            ),
            dtype=np.float64,
        )
        faces = (
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (5, 6, 7),
            (5, 6, 4),
            (5, 7, 4),
            (6, 7, 4),
        )
        topology = compute_tet_surface_topology(faces, points=points)
        self.assertEqual(topology["input_vertex_count"], 8)
        self.assertEqual(topology["welded_vertex_count"], 5)
        self.assertEqual(topology["internal_shared_face_count"], 1)
        self.assertEqual(topology["non_manifold_boundary_edges"], 0)

    def test_rejects_invalid_indices(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            compute_tet_quality(self.points, ((0, 1, 2, 9),))


if __name__ == "__main__":
    unittest.main()
