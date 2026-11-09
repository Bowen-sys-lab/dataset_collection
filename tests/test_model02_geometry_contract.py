import unittest
import numpy as np

from scripts.blender.model02_geometry import clip_polygon, split_by_planes


class GeometryContractTests(unittest.TestCase):
    def polygon(self):
        # Columns: xyz, uv, corner normal. A hand-checked two-square-metre face.
        return [np.array([0.,0.,0.,0.,0.,0.,0.,1.]),
                np.array([2.,0.,0.,1.,0.,0.,0.,1.]),
                np.array([2.,1.,0.,1.,1.,0.,0.,1.]),
                np.array([0.,1.,0.,0.,1.,0.,0.,1.])]

    def area(self, polygon):
        p = np.asarray(polygon)[:,:3]
        return sum(np.linalg.norm(np.cross(p[i]-p[0],p[i+1]-p[0]))/2 for i in range(1,len(p)-1))

    def test_split_has_no_lost_area_or_texture_discontinuity(self):
        pieces = split_by_planes(self.polygon(),0,[.5,1.5])
        self.assertAlmostEqual(sum(self.area(p) for p in pieces),2.)
        self.assertEqual(len(pieces),3)
        for piece in pieces:
            for vertex in piece:
                self.assertAlmostEqual(vertex[3],vertex[0]/2.)
                self.assertAlmostEqual(vertex[4],vertex[1])
                np.testing.assert_allclose(vertex[5:],[0.,0.,1.])

    def test_tangent_clip_does_not_create_a_degenerate_piece(self):
        self.assertEqual(clip_polygon(self.polygon(),0,2.,keep_above=True),[])
        self.assertEqual(len(split_by_planes(self.polygon(),0,[0.,2.])),1)


if __name__ == '__main__':
    unittest.main()
