import json
import math
from pathlib import Path
import unittest
import numpy as np
from coupled_scene.active_drive import layout, motion_state, body_pose, rotation

CONFIG=Path(__file__).resolve().parents[1]/'configs/active_drive_scene.json'


def obb(body,shape,motion,t):
    pose=body_pose(body,motion,t);axes=np.array(rotation(pose['rotation_axis'],pose['angle_rad']))
    centre=np.array(pose['position_m'])+axes@np.array(shape['centre'])
    size=shape.get('size',[2*shape.get('radius',0),shape.get('height',0),2*shape.get('radius',0)])
    return centre,axes,np.array(size)/2


def separated(a,b):
    """SAT; cylinders use conservative enclosing boxes for layout clearance."""
    ca,ra,ha=a;cb,rb,hb=b
    axes=[*ra.T,*rb.T]+[np.cross(x,y) for x in ra.T for y in rb.T]
    for axis in axes:
        if np.linalg.norm(axis)<1e-8: continue
        axis=axis/np.linalg.norm(axis)
        if abs((cb-ca)@axis) > np.abs(ra.T@axis)@ha+np.abs(rb.T@axis)@hb+1e-8:
            return True
    return False


class ActiveDriveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cases=json.loads(CONFIG.read_text())['cases']

    def test_three_independent_actions(self):
        self.assertEqual(len(self.cases),3)
        for case in self.cases:
            recipe=layout(case)
            self.assertEqual([b['name'] for b in recipe['bodies'] if b['moving']],[case['motion']['target']])
            self.assertFalse(recipe['fill']['exact_particle_count_or_volume'])

    def test_analytic_velocities_match_motion(self):
        for case in self.cases:
            m=case['motion']
            for t in np.linspace(.01,case['duration_s']-.01,177):
                a=motion_state(m,t-1e-5);b=motion_state(m,t+1e-5);state=motion_state(m,t)
                np.testing.assert_allclose((np.array(b['displacement_m'])-a['displacement_m'])/2e-5,state['linear_velocity_m_s'],atol=1e-6)
                self.assertAlmostEqual((b['angle_rad']-a['angle_rad'])/2e-5,state['angular_velocity_rad_s'][m['axis']],places=6)

    def test_stopped_endpoints(self):
        for case in self.cases:
            for t in (-1,0,case['motion']['start_s'],case['motion']['stop_s'],case['duration_s']+1):
                state=motion_state(case['motion'],t)
                np.testing.assert_allclose(state['linear_velocity_m_s'],0,atol=1e-10)
                np.testing.assert_allclose(state['angular_velocity_rad_s'],0,atol=1e-10)

    def test_pour_fixed_lip_and_return(self):
        case=self.cases[0];donor=layout(case)['bodies'][1]
        for t in np.linspace(0,10,101): self.assertEqual(body_pose(donor,case['motion'],t)['position_m'],case['donor_lip_pivot_m'])
        self.assertAlmostEqual(motion_state(case['motion'],6)['angle_rad'],math.radians(-105))
        self.assertEqual(motion_state(case['motion'],10)['angle_rad'],0)

    def test_spin_continues_then_holds_angle(self):
        m=self.cases[1]['motion']
        self.assertAlmostEqual(motion_state(m,4)['angular_velocity_rad_s'][1],1.8)
        self.assertGreater(motion_state(m,8)['angle_rad'],2*math.pi)
        self.assertEqual(motion_state(m,8)['angle_rad'],motion_state(m,10)['angle_rad'])

    def test_piston_does_not_retract_and_has_headroom(self):
        case=self.cases[2];m=case['motion'];recipe=layout(case)
        self.assertEqual(motion_state(m,10)['displacement_m'],[.45,0,0])
        length,depth,width=recipe['fill']['size_m']
        self.assertLess(depth*length/(length-m['stroke_m']),case['tank_size_m'][1])
        self.assertGreater(recipe['fill']['centre_m'][0]-length/2,case['piston_start_x_m']+case['piston_thickness_m']/2)

    def test_moving_colliders_do_not_intersect_static_tanks(self):
        for case in self.cases:
            bodies=layout(case)['bodies'];moving=next(b for b in bodies if b['moving'])
            for t in np.linspace(0,case['duration_s'],301):
                for static in (b for b in bodies if not b['moving']):
                    for a in moving['collision']:
                        for b in static['collision']:
                            self.assertTrue(separated(obb(moving,a,case['motion'],t),obb(static,b,case['motion'],t)),
                                (case['id'],t,a['name'],b['name']))


if __name__=='__main__': unittest.main()
