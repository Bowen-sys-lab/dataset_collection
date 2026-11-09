import importlib.util,unittest
import numpy as np


class FullFactoryGeometryTests(unittest.TestCase):
    def test_native_area_bound_accounts_for_float32_cad_coordinates_and_scale(self):
        m=self.module();p=np.array([[13386.1007,132645.536,200.],[14016.1007,132645.536,200.],[13386.1007,133324.541,200.]])
        quantized=p.astype('float32');matrix=np.diag([.001,.001,.001,1.]);indices=np.array([[0,1,2]])
        area=lambda q:np.linalg.norm(np.cross(q[1]-q[0],q[2]-q[0]))/2.*1e-6
        error=abs(area(p)-area(quantized.astype(float)))
        bound=m.native_area_rounding_bound(quantized,matrix,indices)
        self.assertGreater(error,1e-6);self.assertLess(error,bound);self.assertLess(bound,.001)

    def test_frame_rear_reaches_measured_carrier_without_using_adjacent_wall(self):
        m=self.module()
        wall=[np.array(p,dtype=float) for p in [[0,0,-.03],[3,0,-.03],[3,4,-.03],[0,4,-.03]]]
        adjacent=[p+np.array([10.,0.,.02]) for p in wall]
        depth=m.opening_support_offset([{'poly':wall},{'poly':adjacent}],[0,0,1],[1,0,0],[0,1,0],(1.,2.,1.,3.),0.)
        self.assertAlmostEqual(depth,-.03)
        self.assertIsNone(m.opening_support_offset([{'poly':adjacent}],[0,0,1],[1,0,0],[0,1,0],(1.,2.,1.,3.),0.))

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.blender.model02_full_factory_common'))
        from scripts.blender import model02_full_factory_common
        return model02_full_factory_common

    def test_thermal_wall_regions_keep_same_plane_and_footprint(self):
        m=self.module()
        poly=[np.array(p,dtype=float) for p in [[0,0,0],[4,0,0],[4,3,0],[0,3,0]]]
        result=m.partition_surface(poly,np.eye(3),2.)
        from scripts.blender.model02_geometry import polygon_area
        self.assertAlmostEqual(sum(polygon_area(p) for _,p in result),12.)
        self.assertTrue(all(abs(v[2])<1e-12 for _,p in result for v in p))

    def test_opening_frames_and_recess_are_measured_in_surface_coordinates(self):
        m=self.module();n=np.array([0.,-1.,0.]);u=np.array([1.,0.,0.])
        vertices,faces=m.box_geometry((1.,2.,.5,2.5),n,u,4.,-.018,0.)
        p=np.asarray(vertices)
        self.assertAlmostEqual(np.min(p@n),3.982);self.assertAlmostEqual(np.max(p@n),4.)
        self.assertEqual(len(faces),6)

    def test_clipping_retains_uv_at_shared_boundaries(self):
        m=self.module()
        poly=[np.array(p,dtype=float) for p in [[0,0,0,0,0],[4,0,0,1,0],[4,2,0,1,1],[0,2,0,0,1]]]
        result=m.partition_surface(poly,np.eye(3),2.)
        seam=[v for _,p in result for v in p if abs(v[0]-2.)<1e-9]
        self.assertTrue(seam);self.assertTrue(all(abs(v[3]-.5)<1e-9 for v in seam))

    def test_noncircular_section_is_taken_from_actual_edges(self):
        m=self.module()
        poly=[np.array(p,dtype=float) for p in [[2,0,0],[2,1,0],[2,1,3],[2,0,3]]]
        segment=m.section_segment(poly,1.5)
        self.assertEqual(len(segment),2)
        self.assertTrue(all(abs(p[0]-2.)<1e-9 and abs(p[2]-1.5)<1e-9 for p in segment))

    def test_bare_and_painted_metal_are_not_masonry_or_shared_ir_material(self):
        m=self.module()
        bare=m.surface_kind('11-29-gc020.jpg','building',[0.,0.,1.],8.,0.,8.)
        painted=m.surface_kind('11-29-gc030.jpg','building',[0.,0.,1.],8.,0.,8.)
        self.assertNotEqual(bare[1],painted[1]);self.assertNotEqual(bare[1],'etics_render')

    def test_axial_sectors_cut_actual_shell_without_new_radial_faces(self):
        m=self.module()
        p=[np.array(v,dtype=float) for v in [[1,-1,0],[1,1,0],[1,1,2],[1,-1,2]]]
        parts=m.partition_axial(p,np.eye(3),np.zeros(3),1.,8)
        from scripts.blender.model02_geometry import polygon_area
        self.assertAlmostEqual(sum(polygon_area(poly) for _,poly in parts),4.)
        self.assertTrue(all(abs(v[0]-1.)<1e-9 for _,poly in parts for v in poly))
        self.assertTrue(all(np.ptp(np.asarray(poly)[:,2])<=1.+1e-9 for _,poly in parts))

    def test_nanometre_cut_sliver_is_not_a_mesh_object(self):
        m=self.module()
        p=[np.array(v,dtype=float) for v in [[711.5566754437,-325.5690742399,5.2642819338],
            [711.5566754406,-325.5690742428,5.2642819338],[711.428843645,-325.4367005898,5.264281928]]]
        self.assertEqual(m.partition_surface(p,np.eye(3),2.),[])

    def test_nearly_coplanar_roof_does_not_gain_area_at_grid_plane(self):
        m=self.module()
        from scripts.blender.model02_route_refinement import BUILDING_BASIS
        from scripts.blender.model02_geometry import polygon_area
        p=[np.array(v) for v in [[718.675685786,-310.934073368,17.9999996725],
            [707.894714571,-321.345134409,18.0000001275],[749.912415785,-364.855738038,18.0000020294]]]
        parts=m.partition_surface(p,BUILDING_BASIS,2.)
        self.assertAlmostEqual(sum(polygon_area(poly) for _,poly in parts),polygon_area(p),places=5)

    def test_opening_halves_join_across_rgb_slots_and_reversed_source_winding(self):
        m=self.module()
        a={'poly':[np.array(v,dtype=float) for v in [[0,0,0],[2,0,0],[2,0,3]]],'material':0,'face':0}
        b={'poly':[np.array(v,dtype=float) for v in [[0,0,0],[0,0,3],[2,0,3]]],'material':1,'face':1}
        groups=m.connected_opening_groups([a,b],{0:'glass.jpg',1:'glass.jpg'})
        self.assertEqual(len(groups),1);self.assertEqual(len(groups[0]),2)

    def test_pipe_cardinal_side_is_not_split_into_two_diagonal_sectors(self):
        m=self.module();p=[np.array(v,dtype=float) for v in [[1,-1,0],[1,1,0],[1,1,2],[1,-1,2]]]
        parts=m.partition_axial(p,np.eye(3),np.zeros(3),1.,4)
        self.assertEqual({key[1] for key,_ in parts},{2})

    def test_bare_wall_keeps_wall_temperature_component(self):
        m=self.module();component,material=m.surface_kind('11-29-gc013.jpg','building',[1.,0.,0.],4.,0.,8.)
        self.assertEqual(component,'Wall');self.assertEqual(material,'bare_aluminium')
