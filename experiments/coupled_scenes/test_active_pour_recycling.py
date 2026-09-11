"""No GPU needed: recycling must not hide in-basin or airborne particles."""
import unittest
import numpy as np
from run_active_pour_probe import escaped_particle_mask,piston_prewarm_sink_mask


class RecyclingTests(unittest.TestCase):
    def test_piston_sink_does_not_hide_seal_or_floor_leaks(self):
        origin=np.array([1.,2.,3.])
        walls=np.array([[-.605,.039,-.172],[.6775,.283,.172]])
        plate=np.array([[-.0195,.07,-.1655],[.0195,.264,.1655]])
        p=np.array([[-.2,0.,-.20],[-.2,0.,.20],[-.55,0.,-.20],[-.48,0.,-.20],
                    [-.2,-1.,0.],[-.2,.05,-.20],[-.2,0.,-.20],[.8,0.,-.20]])
        v=np.zeros_like(p);v[:,1]=-1;v[6,1]=1
        mask=piston_prewarm_sink_mask(p+origin,v,origin,walls,origin+[-.5,0,0],plate)
        np.testing.assert_array_equal(mask,[True,True,False,False,False,False,False,False])

    def test_geometry_and_direction_not_speed(self):
        mesh=np.array([[-.525,0,-.3255],[.525,.15,.3255]])
        p=np.array([[.7,-.03,0],[.7,.1,0],[0,-.03,0],[.7,-.03,0],
                    [.53,-.03,0],[.7,-.01,0],[.7,-.03,0],[0,-.03,.4]])
        v=np.zeros_like(p);v[:,1]=-1;v[3,1]=1;v[6,1]=-.001;v[2,1]=-50
        origin=np.array([1.,2.,3.])
        actual=escaped_particle_mask(p+origin,v,origin,mesh)
        np.testing.assert_array_equal(actual,[True,False,False,False,False,False,True,True])

    def test_all_in_basin_kept(self):
        mesh=np.array([[-.525,0,-.3255],[.525,.15,.3255]])
        p=np.array([[0,.03,0],[.45,.11,0],[0,.4,0]])
        self.assertFalse(escaped_particle_mask(p,np.full_like(p,-50),[0,0,0],mesh).any())

    def test_unsupported_asymmetric_receiver_rejected(self):
        with self.assertRaises(ValueError):
            escaped_particle_mask(np.zeros((1,3)),np.zeros((1,3)),[0,0,0],
                                  np.array([[-.4,0,-.3],[.5,.15,.3]]))


if __name__=='__main__':
    unittest.main()
