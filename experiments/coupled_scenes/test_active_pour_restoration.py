"""CPU checks for the bounded low-to-runtime vorticity experiment."""
import unittest

from run_active_pour_probe import restored_vorticity,restoration_gate,resolution_offsets
from coupled_scene.surface_settling import SettlingGate


class RestorationTests(unittest.TestCase):
    def test_resolution_offsets_preserve_default_and_scale_together(self):
        baseline=resolution_offsets(.004)
        self.assertEqual(baseline['fluid_rest'],.002)
        self.assertEqual(baseline['system_contact'],.002/.6+.001)
        self.assertEqual(baseline['wall_contact'],.004)
        for key,value in resolution_offsets(.003).items():
            self.assertAlmostEqual(value,baseline[key]*.75)
        with self.assertRaises(ValueError):resolution_offsets(.002)

    def test_relaxation_is_local_and_speed_only(self):
        ordinary=SettlingGate()
        strict=restoration_gate(6.)
        relaxed=restoration_gate(6.,True)
        self.assertEqual(strict.limits,ordinary.limits)
        self.assertEqual(relaxed.limits,{**ordinary.limits,'rms_speed_m_s':.08,'speed_p99_m_s':.20})
        self.assertEqual(SettlingGate().limits,ordinary.limits)
        self.assertEqual(relaxed.window_seconds,2.)

    def test_relaxed_gate_accepts_observed_motion_but_keeps_guards(self):
        for violation in [None,'rms_speed_m_s','speed_p99_m_s','max_speed_m_s',
                          'outside_side_or_floor_count','over_rim_particle_count','regional_level_p99_m']:
            gate=restoration_gate(6.,True)
            for i in range(21):
                row=dict(simulated_seconds=4.+i/10,rms_speed_m_s=.069,
                         speed_p99_m_s=.166,max_speed_m_s=.283,
                         outside_side_or_floor_count=0,over_rim_particle_count=0,
                         regional_level_p99_m=[.124]*4)
                if i==10 and violation:
                    row[violation]=[.127]*4 if violation=='regional_level_p99_m' else 1.
                ready=gate.observe(row)
            self.assertEqual(ready,violation is None,violation)

    def test_ramp_endpoints_and_monotonicity(self):
        self.assertEqual(restored_vorticity(-1), .02)
        self.assertEqual(restored_vorticity(0), .02)
        self.assertAlmostEqual(restored_vorticity(.5), 5.01)
        self.assertEqual(restored_vorticity(1), 10.)
        self.assertEqual(restored_vorticity(4), 10.)
        values=[restored_vorticity(i/720) for i in range(721)]
        self.assertTrue(all(a<=b for a,b in zip(values,values[1:])))
        self.assertLess(values[1]-values[0], .0001)
        self.assertLess(values[-1]-values[-2], .0001)

    def test_full_restored_window_required(self):
        gate=SettlingGate(minimum_seconds=6.,window_seconds=2.)
        for i in range(21):
            row=dict(simulated_seconds=4.+i/10, rms_speed_m_s=.026,
                     speed_p99_m_s=.069,max_speed_m_s=.15,
                     outside_side_or_floor_count=0,over_rim_particle_count=0,
                     regional_level_p99_m=[.124]*4)
            self.assertEqual(gate.observe(row),i==20)

    def test_reexcitation_rejects_window(self):
        gate=SettlingGate(minimum_seconds=6.,window_seconds=2.)
        for i in range(21):
            row=dict(simulated_seconds=4.+i/10,rms_speed_m_s=.068 if i==10 else .026,
                     speed_p99_m_s=.069,max_speed_m_s=.15,
                     outside_side_or_floor_count=0,over_rim_particle_count=0,
                     regional_level_p99_m=[.124]*4)
            self.assertFalse(gate.observe(row))


if __name__=='__main__':
    unittest.main()
