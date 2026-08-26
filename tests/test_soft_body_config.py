from __future__ import annotations

import unittest

from soft_body.config import normalize_body_specs


DEFAULTS = {
    "density": 1050.0,
    "youngs_modulus": 500000.0,
    "poissons_ratio": 0.45,
    "linear_damping": 0.45,
    "settling_damping": 2.0,
    "restitution": 0.25,
}


class BodyConfigTests(unittest.TestCase):
    def test_normalizes_body_without_mutating_source(self):
        source = {
            "model": "carrot.stl",
            "model_height": "0.735",
            "spawn_x": "1",
            "drop_height": "2",
            "spawn_z": "3",
        }

        result = normalize_body_specs([source], DEFAULTS)

        self.assertEqual(result[0]["spawn"], (1.0, 2.0, 3.0))
        self.assertEqual(result[0]["physics_kind"], "deformable")
        self.assertEqual(result[0]["density"], 1050.0)
        self.assertNotIn("collision_contact_offset", result[0])
        self.assertIn("spawn_x", source)

    def test_rejects_unsupported_physics_kind(self):
        body = {
            "model": "carrot.stl",
            "model_height": 1,
            "spawn_x": 0,
            "drop_height": 1,
            "spawn_z": 0,
            "physics_kind": "cloth",
        }
        with self.assertRaisesRegex(ValueError, "Unsupported physics_kind"):
            normalize_body_specs([body], DEFAULTS)

    def test_rejects_inverted_contact_offsets(self):
        body = {
            "model": "carrot.stl",
            "model_height": 1,
            "spawn_x": 0,
            "drop_height": 1,
            "spawn_z": 0,
            "collision_contact_offset": 0.01,
            "collision_rest_offset": 0.02,
        }
        with self.assertRaisesRegex(ValueError, "contact_offset"):
            normalize_body_specs([body], DEFAULTS)


if __name__ == "__main__":
    unittest.main()
