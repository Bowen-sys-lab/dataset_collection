import copy,unittest
from scripts.validate_model02_v6 import validate_view_evidence


class AcceptanceTests(unittest.TestCase):
    def test_surface_views_do_not_require_unavailable_nadir_gimbal(self):
        import math,numpy as np
        from scripts.validate_model02_v6 import diagnostic_eye
        for center in [[10,20,.2],[10,20,20.]]:
            eye=diagnostic_eye(center,[0,0,1],[10,20,0],2.5,'near',center[2]<1.)
            delta=eye-np.array(center);pitch=math.degrees(math.atan2(delta[2],np.linalg.norm(delta[:2])))
            self.assertLessEqual(pitch,35.+1e-8)

    def test_ground_diagnostic_pose_stays_above_surface_and_two_metres(self):
        import numpy as np
        from scripts.validate_model02_v6 import diagnostic_eye
        for normal in [[0,0,-1],[0,0,1],[1,0,0]]:
            eye=diagnostic_eye([10,20,.2],normal,[10,20,0],1.2,'interface',True)
            self.assertGreaterEqual(eye[2],2.)
            self.assertGreaterEqual(np.linalg.norm(eye-np.array([10,20,.2])),2.5)

    def test_small_detail_pixel_estimate_is_not_exaggerated(self):
        from scripts.validate_model02_v6 import projected_pixels
        self.assertAlmostEqual(projected_pixels(.008,10.,160,18.,36.),.064)

    def test_response_roi_contains_only_selected_native_sensor_footprint(self):
        import numpy as np
        from scripts.validate_model02_v6 import native_roi
        mask=np.zeros((4,6),dtype=bool);mask[1,3]=True
        expected=np.zeros((2,3),dtype=bool);expected[0,1]=True
        np.testing.assert_array_equal(native_roi(mask,(2,3)),expected)
        with self.assertRaises(ValueError):native_roi(mask,(3,3))

    def data(self):
        registry={'sources':[{'zone_id':'Z11','role':'building'}]}
        views=[{'zone_id':'Z11','scale':scale,'resolution':[w,h],'role_pixels':{'building':40},
            'native_capture':True,'manual_success':None,'intrinsics':{'focal_length_mm':18.},'distance_m':10.}
            for w,h in [(160,128),(640,512)] for scale in ['overall','mid','near']]
        return views,registry

    def test_rejects_missing_zone(self):
        v,r=self.data();r['sources'].append({'zone_id':'Z12','role':'ground'})
        with self.assertRaises(ValueError):validate_view_evidence(v,r)

    def test_rejects_component_label_without_pixels(self):
        v,r=self.data()
        for row in v:row['role_pixels']['building']=0
        with self.assertRaises(ValueError):validate_view_evidence(v,r)

    def test_high_resolution_cannot_replace_live_evidence(self):
        v,r=self.data();v=[x for x in v if x['resolution']==[640,512]]
        with self.assertRaises(ValueError):validate_view_evidence(v,r)

    def test_static_capture_cannot_claim_manual_success(self):
        v,r=self.data();v[0]['manual_success']=True
        with self.assertRaises(ValueError):validate_view_evidence(v,r)
        v[0]['manual_success']=None
        self.assertTrue(validate_view_evidence(v,r)['complete'])
