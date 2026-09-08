import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from coupled_scene.conditioned_inlet import guide_panels, upstream_sink_mask


class ConditionedInletTests(unittest.TestCase):
    def test_panels_are_open_ended_and_have_expected_clearance(self):
        cfg=dict(outlet_centre=[0,0,0],inner_size_xz=[.088,.064],wall_thickness=.01,upstream_height=.08,headroom=.04)
        panels=guide_panels(cfg)
        self.assertEqual(len(panels),4)
        for _,centre,size in panels:
            self.assertAlmostEqual(centre[1]-size[1]/2,0.)
            self.assertAlmostEqual(centre[1]+size[1]/2,.12)
        self.assertAlmostEqual(abs(panels[0][1][0])-panels[0][2][0]/2-.04,.004)

    def test_discharge_permanently_exempts_downstream_splash(self):
        discharged=np.zeros(4,dtype=bool)
        ids=np.arange(4)
        p=np.array([[0,.13,0],[0,-.01,0],[0,.11,0],[0,.12,0]])
        np.testing.assert_array_equal(upstream_sink_mask(p,ids,discharged,0,.12),[True,False,False,False])
        p[1,1]=.2
        self.assertFalse(upstream_sink_mask(p,ids,discharged,0,.12)[1])

    def test_removed_id_gaps_do_not_shift_tracking(self):
        discharged=np.zeros(10,dtype=bool)
        ids=np.array([1,5,9])
        p=np.array([[0,-1,0],[0,.13,0],[0,.05,0]])
        np.testing.assert_array_equal(upstream_sink_mask(p,ids,discharged,0,.12),[False,True,False])
        self.assertTrue(discharged[1])
        self.assertFalse(discharged[0])

    def test_portable_preset_without_old_local_caches(self):
        from experiments.coupled_scenes.prepare_conditioned_inlet_preview import prepare
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'preview'
            run = prepare(output, '/srv/scenes', '/srv/isaacsim/python.sh')
            event = json.loads((output / 'glass_cabinet_pour.json').read_text())
            self.assertEqual(run['frames'], 198)
            self.assertEqual(run['notes']['nominal_birth_particles'], 75600)
            self.assertEqual(event['source']['solver_position_iterations'], 64)
            self.assertEqual(run['command'][run['command'].index('--deformable-solver-position-iterations') + 1], '64')
            self.assertEqual(run['render_environment']['COUPLED_HIDE_UPSTREAM'], '1')
            self.assertNotIn('${', json.dumps(run))
            self.assertNotIn('Y:', json.dumps(run))
            self.assertAlmostEqual(event['source']['centre'][1] - event['conditioned_inlet']['outlet_centre'][1], .08)
            with self.assertRaises(FileExistsError):
                prepare(output, '/srv/scenes', '/srv/isaacsim/python.sh')

    def test_longer_preset_only_extends_emission_duration(self):
        from experiments.coupled_scenes.prepare_conditioned_inlet_preview import prepare
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'longer'
            run = prepare(output, '/srv/scenes', '/srv/isaacsim/python.sh', 372, 30)
            event = json.loads((output / 'glass_cabinet_pour.json').read_text())
            self.assertEqual(run['frames'], 522)
            self.assertEqual(event['source']['stop_frame'], 492)
            self.assertAlmostEqual(run['notes']['nominal_birth_litres'], 29.99808)
            self.assertEqual(event['source']['velocity'], [0, -.96, 0])

if __name__=='__main__':
    unittest.main()
