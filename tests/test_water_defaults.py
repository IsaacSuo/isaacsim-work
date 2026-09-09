import json
from pathlib import Path
import tempfile
import unittest

from coupled_scene.water_defaults import (
    DEFAULT_VORTICITY_CONFINEMENT, surface_vorticity, water_vorticity,
)
from experiments.coupled_scenes.prepare_conditioned_inlet_preview import prepare


class WaterDefaultsTests(unittest.TestCase):
    def test_default_and_explicit_legacy_values(self):
        self.assertEqual(DEFAULT_VORTICITY_CONFINEMENT, 10.)
        self.assertEqual(water_vorticity({}), 10.)
        self.assertEqual(water_vorticity({'vorticity_confinement': .02}), .02)
        self.assertEqual(water_vorticity({'vorticity_confinement': 0}), 0.)

    def test_surface_production_and_historical_diagnostics(self):
        self.assertEqual(surface_vorticity(), 10.)
        self.assertEqual(surface_vorticity(decay_test=True), .02)
        self.assertEqual(surface_vorticity(decay_test=True, decay_override=10), 10.)
        self.assertEqual(surface_vorticity(decay_test=True, decay_override=.02), .02)
        with self.assertRaises(ValueError):
            surface_vorticity(decay_override=10)

    def test_invalid_vorticity(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                water_vorticity({'vorticity_confinement': value})

    def test_server_generator_writes_new_value_without_changing_inlet(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'job'
            run = prepare(output, Path(tmp) / 'scenes', Path(tmp) / 'python.sh')
            event = json.loads((output / 'glass_cabinet_pour.json').read_text())
            self.assertEqual(event['source']['vorticity_confinement'], 10.)
            self.assertEqual(event['source']['spacing'], .004)
            self.assertEqual(event['source']['velocity'], [0, -.96, 0])
            self.assertEqual(event['source']['solver_position_iterations'], 64)
            self.assertEqual(event['emission_chunk_particles'], 8192)
            self.assertEqual(run['frames'], 198)
            self.assertEqual(run['notes']['vorticity_confinement'], 10.)
            self.assertAlmostEqual(run['notes']['nominal_birth_litres'], 4.8384)


if __name__ == '__main__':
    unittest.main()
