import unittest
from unittest.mock import patch
from run_active_pour_probe import official_water_settings,atomic_json,resolution_offsets
from coupled_scene.surface_settling import prewarm_damping


class OfficialWaterTests(unittest.TestCase):
    def test_three_mm_keeps_material_and_scales_offsets(self):
        old,old_material=official_water_settings(.004)
        new,new_material=official_water_settings(.003)
        self.assertEqual(old_material,new_material)
        for key in old:self.assertAlmostEqual(new[key],old[key]*.75)
        self.assertAlmostEqual(new['fluid_rest'],.0015)
        self.assertAlmostEqual(new['particle_contact'],.0025252525252525255)

    def test_official_prewarm_releases_to_zero(self):
        self.assertEqual(prewarm_damping(0,5.,runtime_damping=0.)[0],5.)
        self.assertEqual(prewarm_damping(1.5,5.,runtime_damping=0.)[0],5.)
        self.assertEqual(prewarm_damping(1.75,5.,runtime_damping=0.)[0],2.5)
        self.assertEqual(prewarm_damping(2.,5.,runtime_damping=0.)[0],0.)
        self.assertEqual(prewarm_damping(8.,5.,runtime_damping=0.)[0],0.)
        self.assertEqual(prewarm_damping(8.,5.)[0],.01)

    def test_meter_preset_and_offsets(self):
        offsets,material=official_water_settings(.004)
        self.assertEqual(material,dict(cohesion=.01,damping=0.,friction=.1,surface_tension=.0074,viscosity=1.7e-6,vorticity_confinement=0.))
        self.assertEqual(offsets['fluid_rest'],.002)
        self.assertAlmostEqual(offsets['particle_contact'],.002/(.99*.6))
        self.assertEqual(offsets['system_contact'],offsets['particle_contact'])
        self.assertAlmostEqual(offsets['solid_rest'],.002/.6)
        self.assertEqual(offsets['wall_contact'],.004)
        self.assertAlmostEqual(resolution_offsets(.004)['system_contact'],.002/.6+.001)

    def test_retry_only_transient_permission(self):
        with patch('run_active_pour_probe._atomic_json',side_effect=[PermissionError(),None]) as write,patch('run_active_pour_probe.time.sleep'):
            atomic_json('unused',{})
            self.assertEqual(write.call_count,2)
        with patch('run_active_pour_probe._atomic_json',side_effect=PermissionError()) as write,patch('run_active_pour_probe.time.sleep'):
            with self.assertRaises(PermissionError):atomic_json('unused',{})
            self.assertEqual(write.call_count,7)
        with patch('run_active_pour_probe._atomic_json',side_effect=OSError('disk full')) as write:
            with self.assertRaises(OSError):atomic_json('unused',{})
            self.assertEqual(write.call_count,1)


if __name__=='__main__':unittest.main()
