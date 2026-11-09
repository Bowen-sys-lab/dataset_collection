"""Physical dimensions and continuous height of the ground refinement."""
import importlib.util
import unittest
import numpy as np


class GroundReliefTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.model02_ground_relief'),
                             'Ground relief authoring is missing')
        from scripts import model02_ground_relief
        return model02_ground_relief

    def test_brick_seam_is_two_mm_deep_and_meets_both_neighbours(self):
        m=self.module()
        for x in [.004,.608-.004,.608+.004]:
            self.assertAlmostEqual(m.height(x,.3,'paving'),-.002)
        self.assertAlmostEqual(m.height(.3,.3,'paving'),0.)
        self.assertAlmostEqual(m.height(.004+.0015,.3,'paving'),-.001)

    def test_thermal_quadrant_boundary_has_no_artificial_groove(self):
        m=self.module()
        self.assertAlmostEqual(m.height(.304,.3,'paving'),0.)
        self.assertAlmostEqual(m.height(.3,.304,'paving'),0.)

    def test_chamfer_tessellation_preserves_projected_footprint(self):
        m=self.module()
        from scripts.blender.model02_geometry import polygon_area
        poly=[np.array(p,dtype=float) for p in [[.004,.004,0],[.304,.004,0],[.304,.304,0],[.004,.304,0]]]
        pieces=m.relief_pieces(poly,'paving')
        self.assertGreater(len(pieces),1)
        flat=[np.asarray(p).copy() for p in pieces]
        for p in flat:p[:,2]=0.
        self.assertAlmostEqual(sum(polygon_area(list(p)) for p in flat),.09,places=10)
        self.assertTrue(all(-.002-1e-12<=v[2]<=0 for p in pieces for v in p))

    def test_concrete_control_joint_uses_shallow_recess(self):
        m=self.module()
        self.assertAlmostEqual(m.height(.005,1.,'road'),-.001)
        self.assertAlmostEqual(m.height(2.,2.,'road'),0.)


if __name__=='__main__':unittest.main()
