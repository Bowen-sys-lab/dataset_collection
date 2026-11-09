import math
import unittest

import numpy as np

from scripts.sync_validation_support import compare_recorded_pose


class RecordedPoseComparisonTests(unittest.TestCase):
    def test_three_x_scaled_camera_uses_translation_and_rotation_tolerances(self):
        angle = math.radians(0.004129)
        expected = np.diag([3.0, 3.0, 3.0, 1.0])
        actual = np.array([
            [3.0 * math.cos(angle), -3.0 * math.sin(angle), 0.0, 0.0],
            [3.0 * math.sin(angle), 3.0 * math.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.00000367, -0.00000216, -0.00000075, 1.0],
        ])

        result = compare_recorded_pose(actual, expected, meters_per_unit=1.0)

        self.assertGreater(result['matrix_max_abs_difference'], 1e-4)
        self.assertAlmostEqual(result['translation_error_m'], 4.324e-6, places=10)
        self.assertAlmostEqual(result['rotation_error_degrees'], 0.004129, places=6)
        self.assertTrue(result['within_tolerance'])

    def test_pose_comparison_still_rejects_a_real_translation_error(self):
        expected = np.diag([3.0, 3.0, 3.0, 1.0])
        actual = expected.copy()
        actual[3, 0] += 0.02

        result = compare_recorded_pose(actual, expected, meters_per_unit=1.0)

        self.assertAlmostEqual(result['translation_error_m'], 0.02, places=12)
        self.assertFalse(result['within_tolerance'])


if __name__ == '__main__':
    unittest.main()
