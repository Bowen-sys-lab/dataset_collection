import unittest
import numpy as np
from scripts.model02_collision import consolidate_collision


class CollisionTests(unittest.TestCase):
    def test_local_coordinates_weld_seams_without_moving_valid_faces(self):
        p=np.array([[700.,-800.,3.],[700.1,-800.,3.],[700.,-799.9,3.],
                    [700.,-800.,3.],[700.1,-800.,3.],[700.,-799.9,3.]])
        f=np.array([[0,1,2],[3,4,5],[0,0,1]])
        origin,points,faces,report=consolidate_collision(p,f)
        self.assertLess(np.abs(points).max(),1)
        np.testing.assert_allclose(points[faces]+origin,p[f[:2]],atol=1e-6)
        self.assertEqual(len(points),3)
        self.assertEqual(report['discarded_degenerate_triangles'],1)


if __name__=='__main__':unittest.main()
