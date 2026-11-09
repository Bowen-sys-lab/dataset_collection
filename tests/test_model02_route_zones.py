import unittest
import numpy as np

from scripts.blender.model02_route_zones import split_grid, corridor_distance, angular_bin,cell_in_corridor
from scripts.blender.model02_route_refinement import sector_split
from scripts.blender.model02_geometry import polygon_area


class RouteZonesTests(unittest.TestCase):
    def test_grid_preserves_area_and_interpolated_uv(self):
        p=np.array([[0.,0.,0.,0.,0.],[4.,0.,0.,1.,0.],[4.,3.,0.,1.,1.],[0.,3.,0.,0.,1.]])
        cells=split_grid(p,np.eye(3),[2.,2.,2.])
        area=0.
        for _,poly in cells:
            xyz=np.asarray(poly)[:,:3]
            self.assertLessEqual(np.ptp(xyz[:,0]),2.000001)
            self.assertLessEqual(np.ptp(xyz[:,1]),2.000001)
            for v in poly:
                np.testing.assert_allclose(v[3:5],[v[0]/4,v[1]/3],atol=1e-9)
            area+=sum(np.linalg.norm(np.cross(xyz[i]-xyz[0],xyz[i+1]-xyz[0]))/2 for i in range(1,len(xyz)-1))
        self.assertAlmostEqual(area,12.)

    def test_corridor_and_angular_boundaries(self):
        self.assertAlmostEqual(corridor_distance([5,3],[[0,0],[10,0]]),3.)
        self.assertEqual(angular_bin(1,0,8),4)
        self.assertEqual(angular_bin(0,1,8),6)
        self.assertEqual(angular_bin(-1,0,8),0)

    def test_radial_split_keeps_horizontal_surface_area(self):
        p=np.array([[-2.,-2.,3.,0.,0.],[2.,-2.,3.,1.,0.],[2.,2.,3.,1.,1.],[-2.,2.,3.,0.,1.]])
        parts=sector_split(p,[0.,0.],8)
        self.assertAlmostEqual(sum(polygon_area(q) for q in parts),16.,places=8)

    def test_boundary_cell_not_lost_by_skewed_fragment_mean(self):
        # x=20 touches the20m boundary, even if a fragment mean is near x=22.
        self.assertTrue(cell_in_corridor([10,0,0],[[0,-100],[0,100]],20.,2.))
        self.assertFalse(cell_in_corridor([20,0,0],[[0,-100],[0,100]],20.,2.))


if __name__=='__main__':unittest.main()
