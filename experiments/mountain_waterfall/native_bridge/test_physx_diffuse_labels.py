from __future__ import annotations

import unittest

import numpy as np

from physx_diffuse_labels import label_histogram, recover_labels


class DiffuseLabelRecoveryTests(unittest.TestCase):
    def test_physx_threshold_boundaries_and_strict_radius(self) -> None:
        clusters = []
        queries = []
        expected_counts = [3, 4, 7, 8, 16]
        for cluster_index, count in enumerate(expected_counts):
            centre = np.asarray([cluster_index * 10.0, 0.0, 0.0], dtype=np.float32)
            queries.append(centre)
            angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float32)
            clusters.extend(
                centre + np.asarray([0.25 * np.cos(angle), 0.25 * np.sin(angle), 0.0])
                for angle in angles
            )
            # Exactly on the radius must not count: PhysX uses dSq < radiusSq.
            clusters.append(centre + np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

        labels, counts = recover_labels(
            np.asarray(clusters, dtype=np.float32),
            np.asarray(queries, dtype=np.float32),
            1.0,
        )
        np.testing.assert_array_equal(counts, np.asarray(expected_counts, dtype=np.uint8))
        np.testing.assert_array_equal(labels, np.asarray([0, 1, 1, 2, 2], dtype=np.uint8))
        self.assertEqual(label_histogram(labels), {"spray": 1, "foam": 2, "bubble": 2})

    def test_neighbor_count_is_capped_at_physx_limit(self) -> None:
        primary = np.zeros((32, 3), dtype=np.float32)
        labels, counts = recover_labels(primary, np.zeros((1, 3), dtype=np.float32), 0.5)
        self.assertEqual(int(counts[0]), 16)
        self.assertEqual(int(labels[0]), 2)

    def test_rejects_nonfinite_input(self) -> None:
        with self.assertRaises(ValueError):
            recover_labels(
                np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
