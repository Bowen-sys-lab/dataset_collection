import unittest
import numpy as np
from scene_intake.forest_replay_pose import check_body_replay
from scene_intake.verify_dataset import requested_distance_contract


class ForestRecordedContractTests(unittest.TestCase):
    def test_trajectory_does_not_claim_a_requested_static_distance(self):
        self.assertTrue(requested_distance_contract([{'requested_distance_m':None}],recorded=True)[0])
        self.assertFalse(requested_distance_contract([{'requested_distance_m':None}],recorded=False)[0])
        self.assertTrue(requested_distance_contract([{'requested_distance_m':8}],recorded=False)[0])
        self.assertFalse(requested_distance_contract([{'requested_distance_m':8}],recorded=True)[0])

    def test_body_pose_readback_is_checked_not_just_camera_placement(self):
        matrix=np.eye(4);matrix[3,:3]=[10,20,30]
        row={'recorded_body_world_matrix':matrix.tolist()}
        self.assertTrue(check_body_replay(matrix,row)['restored'])
        wrong=matrix.copy();wrong[3,0]+=1
        with self.assertRaises(ValueError):check_body_replay(wrong,row)


if __name__=='__main__':unittest.main()
