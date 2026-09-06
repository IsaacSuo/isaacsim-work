from __future__ import annotations

import unittest

from experiments.coupled_scenes.build_glass_cabinet_run import deep_pour_spec


class GlassCabinetBuilderTests(unittest.TestCase):
    def test_deep_pour_meets_original_cabinet_three_centimetre_target(self):
        inner_size = (2.0, 1.1272727272727272, 1.4909090909090907)

        result = deep_pour_spec(inner_size, 0.004)

        self.assertEqual(result["prewarm_frames"], 120)
        self.assertEqual(result["inlet_frames"], 300)
        self.assertEqual(result["post_inlet_frames"], 90)
        self.assertEqual(result["source_start_frame"], 121)
        self.assertEqual(result["source_stop_frame"], 420)
        self.assertEqual(result["total_frames"], 510)
        self.assertAlmostEqual(result["target_water_volume_liters"], 89.45454545)
        self.assertEqual(result["nozzle_columns"], [41, 29])
        self.assertEqual(result["nozzle_size_m"], [0.164, 0.116])
        self.assertAlmostEqual(result["inlet_speed_m_s"], 0.96)
        self.assertAlmostEqual(
            result["inlet_speed_m_s"] / (60.0 * 0.004), 4.0
        )
        self.assertGreaterEqual(
            result["projected_water_depth_m"], result["target_water_depth_m"]
        )
        self.assertAlmostEqual(result["projected_water_volume_liters"], 91.3152)
        self.assertEqual(result["projected_particle_count"], 1_426_800)

    def test_deep_pour_rejects_non_positive_inputs(self):
        for spacing in (0.0, -0.004):
            with self.subTest(spacing=spacing):
                with self.assertRaises(ValueError):
                    deep_pour_spec((2.0, 1.0, 1.5), spacing)


if __name__ == "__main__":
    unittest.main()
