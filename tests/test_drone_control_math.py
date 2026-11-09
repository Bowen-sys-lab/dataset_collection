import math
import unittest

from scripts.drone_control_math import (
    MotionConfig,
    body_command_from_keys,
    clamp_horizontal,
    integrate_pose,
    root_orientation_wxyz_from_yaw,
    smooth_velocity,
    target_velocity_from_command,
    world_heading_from_yaw,
)


class DroneControlMathTests(unittest.TestCase):
    def test_keys_form_a_continuous_body_command(self):
        command = body_command_from_keys({"W", "D", "R", "E"})

        self.assertAlmostEqual(command.forward, math.sqrt(0.5))
        self.assertAlmostEqual(command.right, -math.sqrt(0.5))
        self.assertEqual(command.vertical, 1.0)
        self.assertEqual(command.yaw, -1.0)

    def test_opposite_keys_cancel_each_other(self):
        command = body_command_from_keys({"W", "S", "A", "D", "Q", "E"})

        self.assertEqual(command.forward, 0.0)
        self.assertEqual(command.right, 0.0)
        self.assertEqual(command.vertical, 0.0)
        self.assertEqual(command.yaw, 0.0)

    def test_horizontal_vector_is_limited_to_unit_length(self):
        forward, right = clamp_horizontal(1.0, 1.0)

        self.assertAlmostEqual(math.hypot(forward, right), 1.0)

    def test_velocity_approaches_target_using_dt(self):
        config = MotionConfig(horizontal_acceleration=2.0)
        current = (0.0, 0.0, 0.0, 0.0)
        target = (1.0, 0.0, 0.0, 0.0)

        next_velocity = smooth_velocity(current, target, dt=0.25, config=config)

        self.assertEqual(next_velocity, (0.5, 0.0, 0.0, 0.0))

    def test_pose_respects_minimum_height(self):
        position, yaw = integrate_pose(
            position=(2.0, 3.0, 1.0),
            yaw=0.0,
            velocity=(0.0, 0.0, -1.0, 0.0),
            dt=2.0,
            minimum_height=0.5,
        )

        self.assertEqual(position, (2.0, 3.0, 0.5))
        self.assertEqual(yaw, 0.0)

    def test_command_becomes_velocity_using_configured_limits(self):
        config = MotionConfig(
            max_horizontal_speed=2.0,
            max_vertical_speed=0.5,
            max_yaw_speed=math.radians(30.0),
        )
        command = body_command_from_keys({"W", "D", "R", "Q"})

        target = target_velocity_from_command(command, config)

        self.assertAlmostEqual(target[0], math.sqrt(0.5) * 2.0)
        self.assertAlmostEqual(target[1], -math.sqrt(0.5) * 2.0)
        self.assertEqual(target[2], 0.5)
        self.assertAlmostEqual(target[3], math.radians(30.0))

    def test_heading_vector_follows_yaw(self):
        heading = world_heading_from_yaw(math.pi / 2.0)

        self.assertAlmostEqual(heading[0], 0.0)
        self.assertAlmostEqual(heading[1], 1.0)

    def test_articulation_orientation_uses_wxyz_order(self):
        orientation = root_orientation_wxyz_from_yaw(math.pi)

        self.assertAlmostEqual(orientation[0], 0.0)
        self.assertEqual(orientation[1:3], (0.0, 0.0))
        self.assertAlmostEqual(orientation[3], 1.0)


if __name__ == "__main__":
    unittest.main()
