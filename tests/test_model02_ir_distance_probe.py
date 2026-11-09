import copy
import unittest

from scripts.validate_model02_ir_distance_probe import validate_distance_evidence
from scripts.validate_model02_ir_surface_channels import final_probe_result


class DistanceEvidenceTests(unittest.TestCase):
    def fixture(self):
        return [dict(role=role,scale=scale,native_capture=True,manual_success=None,
                     intrinsics_px={'rgb':{'fx':137.},'thermal':{'fx':137.}},
                     pair_pose_max_error=0.,coverage_complete=True,distance_to_focus_center_m=distance,
                     focus_pixels=1 if scale!='overall' else 0,assembly_pixels=10)
                for role in ['opening','equipment','ground']
                for scale,distance in [('overall',70.),('mid',12.),('near',3.5)]]

    def test_accepts_subpixel_overall_focus_with_observed_assembly(self):
        self.assertTrue(validate_distance_evidence(self.fixture())['complete'])

    def test_rejects_missing_requested_focus_or_assembly(self):
        for index,key in [(1,'focus_pixels'),(0,'assembly_pixels')]:
            views=self.fixture();views[index][key]=0
            with self.assertRaises(ValueError):validate_distance_evidence(views)

    def test_rejects_missing_scales_and_unordered_actual_distances(self):
        views=self.fixture()
        with self.assertRaises(ValueError):validate_distance_evidence(views[1:])
        views[2]['distance_to_focus_center_m']=80.
        with self.assertRaises(ValueError):validate_distance_evidence(views)

    def test_combined_probe_cannot_pass_with_distance_failure_or_caught_error(self):
        report={'result':'PASS_CONTROLLED_PROBE','experiments':{case:{'result':'controlled'}
            for case in ['repeat','normal_off','normal_restored','color_only']}}
        report['distance_probe']={'result':'FAIL'}
        self.assertEqual(final_probe_result(report,True),'INCONCLUSIVE_DISTANCE_PROBE')
        report['distance_probe']['result']='PASS_NATIVE_DISTANCE_PROBE'
        self.assertEqual(final_probe_result(report,True),'PASS_NATIVE_SURFACE_AND_DISTANCE_PROBE')
        report['error']='injected distance exception'
        self.assertEqual(final_probe_result(report,True),'FAIL')

    def test_rejects_duplicate_role_scale_and_nonfinite_pose_error(self):
        views=self.fixture();views.append(copy.deepcopy(views[0]))
        with self.assertRaises(ValueError):validate_distance_evidence(views)
        views=self.fixture();views[0]['pair_pose_max_error']=float('nan')
        with self.assertRaises(ValueError):validate_distance_evidence(views)

    def test_rejects_unaligned_or_incomplete_native_capture(self):
        for key,value in [('native_capture',False),('manual_success',True),('coverage_complete',False),('pair_pose_max_error',.1)]:
            views=self.fixture();views[0][key]=value
            with self.assertRaises(ValueError):validate_distance_evidence(views)
        views=copy.deepcopy(self.fixture());views[0]['intrinsics_px']['thermal']['fx']=138.
        with self.assertRaises(ValueError):validate_distance_evidence(views)
