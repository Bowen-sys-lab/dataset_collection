import unittest

from scripts.drone_keyboard_controller import (
    DEFAULT_EMERGENCY_KEY,
    DroneKeyboardController,
    KeyState,
)


class KeyStateTests(unittest.TestCase):
    def test_press_and_release_track_held_keys(self):
        state = KeyState()

        state.press("W")
        state.press("D")
        self.assertEqual(state.pressed, {"W", "D"})

        state.release("W")
        self.assertEqual(state.pressed, {"D"})

    def test_emergency_stop_clears_all_keys(self):
        state = KeyState({"W", "R", "E"})

        state.emergency_stop()

        self.assertEqual(state.pressed, set())

    def test_key_names_are_normalized(self):
        state = KeyState()

        state.press("w")
        state.release("W")

        self.assertEqual(state.pressed, set())

    def test_emergency_key_does_not_use_space_shortcut(self):
        self.assertEqual(DEFAULT_EMERGENCY_KEY, "X")

    def test_direct_pose_controller_starts_above_ground(self):
        controller = DroneKeyboardController()

        self.assertEqual(controller.minimum_height, 1.0)

    def test_stop_resets_update_subscription(self):
        class FakeSubscription:
            def __init__(self):
                self.reset_called = False

            def reset(self):
                self.reset_called = True

        controller = DroneKeyboardController()
        subscription = FakeSubscription()
        controller._update_sub = subscription

        controller.stop()

        self.assertTrue(subscription.reset_called)
        self.assertIsNone(controller._update_sub)


if __name__ == "__main__":
    unittest.main()
