import unittest
from scripts.collection_episode import load_episodes,required_input_paths


class CaptureDependencyTests(unittest.TestCase):
    def test_camera_pose_projection_and_thermal_helpers_are_bound_for_factory(self):
        row=next(r for r in load_episodes() if r['episode']=='F05')
        required={'scripts/dual_camera_runtime.py','scripts/dual_camera_support.py','scripts/replay_pose.py',
                  'scripts/drone_control_math.py','scripts/rgb_scene_diagnostics.py','scripts/capture_crazyflie_thermal_static.py'}
        self.assertFalse(required-required_input_paths(row),'Capture-affecting helper changes must invalidate proof')
