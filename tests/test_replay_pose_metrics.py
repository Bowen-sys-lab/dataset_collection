import math
import unittest

import numpy as np

from scripts.replay_pose import (body_world_from_camera_world, compare_pose_matrices,
                                 pose_within_tolerance, resolve_recorded_body_world)


class ReplayPoseMetricTests(unittest.TestCase):
    def test_legacy_replay_pose_can_be_recovered_from_recorded_rgb_camera(self):
        body_world = np.eye(4)
        body_world[:3, :3] = 3.0 * np.array([
            [math.cos(0.2), math.sin(0.2), 0.0],
            [-math.sin(0.2), math.cos(0.2), 0.0],
            [0.0, 0.0, 1.0],
        ])
        body_world[3, :3] = [5.0, 6.0, 7.0]
        camera_in_body = np.eye(4)
        camera_in_body[3, :3] = [0.4, -0.2, 0.1]
        camera_world = camera_in_body @ body_world

        recovered = body_world_from_camera_world(camera_world, camera_in_body)

        np.testing.assert_allclose(recovered, body_world, atol=1e-12)

    def test_recorded_body_transform_has_priority_over_legacy_camera_fallback(self):
        recorded_body = np.eye(4)
        recorded_body[3, :3] = [1.0, 2.0, 3.0]
        camera_in_body = np.eye(4)
        camera_in_body[3, 0] = 10.0
        record = {
            'recorded_body_world_matrix': recorded_body.tolist(),
            'rgb_camera_world_matrix': (np.eye(4) * 2.0).tolist(),
        }

        matrix, source = resolve_recorded_body_world(record, camera_in_body)

        np.testing.assert_array_equal(matrix, recorded_body)
        self.assertEqual(source, 'recorded_body_world_transform')

    def test_legacy_without_camera_matrix_preserves_existing_pose_fallback(self):
        matrix, source = resolve_recorded_body_world({'position': [1, 2, 3]}, np.eye(4))

        self.assertIsNone(matrix)
        self.assertEqual(source, 'legacy_position_orientation')

    def test_scaled_camera_rotation_drift_is_judged_by_angle_not_matrix_elements(self):
        # A 3x scaled camera transform makes a 0.004129 degree rotation
        # produce a 2.16e-4 element delta, despite negligible image-space impact.
        angle = math.radians(0.004129)
        expected = np.diag([3.0, 3.0, 3.0, 1.0])
        actual = np.array([
            [3.0 * math.cos(angle), -3.0 * math.sin(angle), 0.0, 0.0],
            [3.0 * math.sin(angle), 3.0 * math.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.00000367, 0.0, 0.0, 1.0],
        ])

        errors = compare_pose_matrices(actual, expected, meters_per_unit=1.0)

        self.assertGreater(errors['matrix_max_abs_difference'], 1e-4)
        self.assertAlmostEqual(errors['translation_error_m'], 0.00000367, places=10)
        self.assertAlmostEqual(errors['rotation_error_degrees'], 0.004129, places=6)
        self.assertTrue(pose_within_tolerance(errors, max_translation_error_m=1e-4,
                                              max_rotation_error_degrees=0.01))

    def test_stage_units_are_converted_to_meters(self):
        expected = np.eye(4)
        actual = np.eye(4)
        actual[3, 0] = 0.01

        errors = compare_pose_matrices(actual, expected, meters_per_unit=0.01)

        self.assertAlmostEqual(errors['translation_error_m'], 0.0001, places=12)
        self.assertEqual(errors['rotation_error_degrees'], 0.0)

    def test_physically_meaningful_limits_still_reject_real_pose_mismatch(self):
        expected = np.eye(4)
        translated = np.eye(4)
        translated[3, 0] = 0.02
        errors = compare_pose_matrices(translated, expected, meters_per_unit=0.01)

        self.assertFalse(pose_within_tolerance(errors, max_translation_error_m=1e-4,
                                               max_rotation_error_degrees=0.01))


if __name__ == '__main__':
    unittest.main()
