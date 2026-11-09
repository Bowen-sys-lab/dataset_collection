import math
import unittest
from unittest.mock import patch

from scripts.drone_control_math import MotionConfig, smooth_velocity
from scripts import start_crazyflie_dual_camera_view as preview_app
from scripts.start_crazyflie_dual_camera_view import preview_motion_config


class PreviewMotionOptionsTests(unittest.TestCase):
    def test_defaults_preserve_existing_limits(self):
        self.assertEqual(preview_motion_config(), MotionConfig())

    def test_demo_cli_defaults_to_search_speed(self):
        config = self._cli_motion_config([])
        self.assertEqual(config.max_horizontal_speed, 5.0)
        self.assertEqual(config.horizontal_acceleration, 10.0)
        self.assertEqual(config.max_vertical_speed, 0.5)
        self.assertAlmostEqual(config.max_yaw_speed, math.radians(45.0))

    def test_demo_cli_can_use_slower_horizontal_speed(self):
        config = self._cli_motion_config(['--horizontal-speed', '3'])
        self.assertEqual(config.max_horizontal_speed, 3.0)
        self.assertEqual(config.horizontal_acceleration, 6.0)
        self.assertEqual(config.max_vertical_speed, 0.5)
        self.assertAlmostEqual(config.max_yaw_speed, math.radians(45.0))

    def _cli_motion_config(self, argv):
        configs = []

        def record_config(*args):
            config = preview_motion_config(*args)
            configs.append(config)
            return config

        # Stop after real CLI parsing and motion configuration, before Kit starts.
        with patch.object(preview_app, 'preview_motion_config', side_effect=record_config), \
                patch.object(preview_app, 'load_scene_config', side_effect=RuntimeError('stop before Kit')):
            with self.assertRaisesRegex(RuntimeError, 'stop before Kit'):
                preview_app.main(argv)
        self.assertEqual(len(configs), 1)
        return configs[0]

    def test_legacy_scale_still_scales_all_axes(self):
        config = preview_motion_config(1.5)
        self.assertEqual(config.max_horizontal_speed, 1.5)
        self.assertEqual(config.max_vertical_speed, 0.75)
        self.assertAlmostEqual(config.max_yaw_speed, math.radians(67.5))

    def test_horizontal_override_leaves_yaw_height_and_attitude_alone(self):
        original = preview_motion_config(1.5)
        config = preview_motion_config(1.5, horizontal_speed=10.0)
        self.assertEqual(config.max_horizontal_speed, 10.0)
        self.assertEqual(config.horizontal_acceleration, 20.0)
        self.assertEqual(config.max_vertical_speed, original.max_vertical_speed)
        self.assertEqual(config.vertical_acceleration, original.vertical_acceleration)
        self.assertEqual(config.max_yaw_speed, original.max_yaw_speed)
        self.assertEqual(config.yaw_acceleration, original.yaw_acceleration)
        velocity = smooth_velocity((0., 0., 0., 0.), (10., 0., 0., 0.), 0.25, config)
        self.assertEqual(velocity[0], 5.0)
        velocity = smooth_velocity(velocity, (10., 0., 0., 0.), 0.25, config)
        self.assertEqual(velocity[0], 10.0)
        self.assertEqual(smooth_velocity(velocity, (0., 0., 0., 0.), 0.5, config)[0], 0.0)

    def test_rejects_invalid_limits_before_kit_startup(self):
        for value in (0.0, -1.0, 20.1, float('nan'), float('inf')):
            with self.subTest(horizontal_speed=value):
                with self.assertRaises(ValueError):
                    preview_motion_config(horizontal_speed=value)
        for value in (0.0, 4.1, float('nan'), float('inf')):
            with self.subTest(speed_scale=value):
                with self.assertRaises(ValueError):
                    preview_motion_config(speed_scale=value)


if __name__ == '__main__':
    unittest.main()
