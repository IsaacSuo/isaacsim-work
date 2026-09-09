"""Small CPU tests for wave-decay metrics; do not start Isaac/GPU."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from coupled_scene.surface_decay import wave_metrics, decay_threshold_times


class WaveMetricsTest(unittest.TestCase):
    def setUp(self):
        self.p=np.array([(x,z,y) for x in np.arange(-.775,.8,.05)
                         for y in np.arange(-.425,.45,.05) for z in (.02,.04,.06,.08)])
        self.v=np.zeros_like(self.p)

    def test_flat(self):
        m=wave_metrics(self.p,self.v,[0,0,0],[1.6,.28,.9])
        self.assertAlmostEqual(m['height_std_mm'],0)
        self.assertAlmostEqual(m['coarse_flow_rms_m_s'],0)

    def test_uniform_flow(self):
        self.v[:,0]=.1
        m=wave_metrics(self.p,self.v,[0,0,0],[1.6,.28,.9])
        self.assertAlmostEqual(m['coarse_flow_rms_m_s'],.1)
        self.assertLess(m['within_column_residual_rms_m_s'],1e-8)

    def test_opposing_flow_is_not_coherent(self):
        self.v[::2,0]=.1;self.v[1::2,0]=-.1
        m=wave_metrics(self.p,self.v,[0,0,0],[1.6,.28,.9])
        self.assertAlmostEqual(m['coarse_flow_rms_m_s'],0)
        self.assertAlmostEqual(m['within_column_residual_rms_m_s'],.1)

    def test_tilt(self):
        self.p[:,1]+=.01*self.p[:,0]
        m=wave_metrics(self.p,self.v,[0,0,0],[1.6,.28,.9])
        self.assertGreater(m['height_p95_p05_mm'],10)

    def test_threshold_rebound_and_short_followup(self):
        heights=[4.,7.,4.,3.,2.,1.,.4,.3,.2,.1]
        report=dict(hz=720,decay=dict(initial_metrics=dict(height_p95_p05_mm=33.)),rows=[
            dict(step=(i+1)*72,simulated_seconds=(i+1)/10,wave_metrics=dict(height_p95_p05_mm=h))
            for i,h in enumerate(heights)])
        result=decay_threshold_times(report)['thresholds_mm']
        self.assertEqual(result['5']['first_below_seconds'],.1)
        self.assertEqual(result['5']['confirmed_below_seconds'],.3)
        self.assertIsNone(result['0.5']['confirmed_below_seconds'])

    def test_threshold_sampling_matches_across_frequencies(self):
        results=[]
        for hz in (360,720,1440):
            report=dict(hz=hz,decay=dict(initial_metrics=dict(height_p95_p05_mm=33.)),rows=[
                dict(step=i*(hz//10),simulated_seconds=i/10,wave_metrics=dict(height_p95_p05_mm=4. if i<5 else .2))
                for i in range(1,11)])
            results.append(decay_threshold_times(report))
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[1],results[2])


if __name__=='__main__':unittest.main()
