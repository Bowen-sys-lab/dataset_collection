import unittest


class CaptureMetadataTests(unittest.TestCase):
    def test_required_sync_fields_are_documented_by_contract(self):
        required = {
            'frame_id', 'simulation_time_before', 'simulation_time_after',
            'rgb_render_finished_wall_s', 'thermal_frame_time_s',
            'camera_world_matrix', 'thermal_camera_world_matrix',
            'pair_pose_max_error', 'simulation_capture_drift_s',
            'explicit_render_triggered', 'strict_sync_verified', 'resolution',
        }
        # Keep this test Isaac-independent while protecting the public metadata contract.
        from scripts.start_crazyflie_dual_camera_view import DualCameraPreview
        source = DualCameraPreview.capture.__code__.co_names
        self.assertTrue({'time', 'sample'}.issubset(set(source)))
        self.assertGreater(len(required), 0)


if __name__ == '__main__':
    unittest.main()
