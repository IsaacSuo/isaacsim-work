"""CPU-only tests for the settling-to-recording readiness decision."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coupled_scene.surface_settling import SettlingGate, prewarm_damping


def sample(t, **overrides):
    return dict(simulated_seconds=t, rms_speed_m_s=.001, speed_p99_m_s=.005,
                max_speed_m_s=.02, regional_level_p99_m=[.08]*4,
                outside_side_or_floor_count=0, over_rim_particle_count=0,
                **overrides)


class GateTests(unittest.TestCase):
    def test_relaxed_speed_accepts_v3_like_state(self):
        gate = SettlingGate(minimum_seconds=0)
        for i in range(6):
            row = sample(i/10)
            row.update(rms_speed_m_s=.0266,speed_p99_m_s=.0716,max_speed_m_s=.166)
            ready = gate.observe(row)
        self.assertTrue(ready)

    def test_relaxed_limits_still_reject_fast_or_escaped_state(self):
        for field,value in [('rms_speed_m_s',.0351),('speed_p99_m_s',.0901),
                            ('max_speed_m_s',.5001),('outside_side_or_floor_count',1),
                            ('over_rim_particle_count',1)]:
            with self.subTest(field=field):
                gate = SettlingGate(minimum_seconds=0)
                for i in range(7):
                    row=sample(i/10)
                    row[field]=value
                    self.assertFalse(gate.observe(row))

    def test_assisted_damping_restores_original(self):
        self.assertEqual(prewarm_damping(0,None),(.01,'normal_validation'))
        self.assertEqual(prewarm_damping(0,5.),(5.,'assisted_relaxation'))
        self.assertEqual(prewarm_damping(1.5,5.),(5.,'damping_release'))
        middle, phase = prewarm_damping(1.75,5.)
        self.assertAlmostEqual(middle,2.505)
        self.assertEqual(phase,'damping_release')
        self.assertEqual(prewarm_damping(2.,5.),(.01,'normal_validation'))
        self.assertEqual(prewarm_damping(8.,5.),(.01,'normal_validation'))

    def test_no_readiness_from_assisted_window(self):
        gate = SettlingGate(minimum_seconds=3.)
        for i in range(20,30):
            self.assertFalse(gate.observe(sample(i/10)))
        self.assertTrue(gate.observe(sample(3.)))

    def test_minimum_and_window(self):
        gate = SettlingGate()
        for i in range(20):
            self.assertFalse(gate.observe(sample(i/10)))
        self.assertTrue(gate.observe(sample(2.)))

    def test_transient_restarts_window(self):
        gate = SettlingGate(minimum_seconds=0)
        for i in range(6):
            row = sample(i/10)
            if i == 3:
                row['over_rim_particle_count'] = 1
            self.assertFalse(gate.observe(row))
        for i in range(6,9):
            self.assertFalse(gate.observe(sample(i/10)))
        self.assertTrue(gate.observe(sample(.9)))

    def test_local_wave_does_not_cancel_globally(self):
        gate = SettlingGate(minimum_seconds=0)
        for i in range(7):
            row = sample(i/10)
            row['regional_level_p99_m'] = [.08+i*.001,.08-i*.001,.08,.08]
            self.assertFalse(gate.observe(row))

    def test_fast_particle_and_nan_reject(self):
        for field, value in [('max_speed_m_s',1.), ('rms_speed_m_s',float('nan')),
                             ('regional_level_p99_m',[None]*4)]:
            gate = SettlingGate(minimum_seconds=0)
            for i in range(7):
                row = sample(i/10)
                row[field] = value
                self.assertFalse(gate.observe(row))


if __name__ == '__main__':
    unittest.main()
