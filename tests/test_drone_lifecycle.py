import unittest
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from scripts.drone_keyboard_controller import DroneKeyboardController


class FakePhysics:
    valid = True
    ready = True

    def __init__(self):
        self.position = (2., 3., 4.)
        self.orientation = (1., 0., 0., 0.)
        self.gravity = True
        self.velocity = (0., 0., -2.)

    def snapshot(self):
        return {'position': self.position, 'orientation': self.orientation,
                'linear_velocity': self.velocity, 'gravity_enabled': [self.gravity]}

    def prepare_hover(self):
        self.gravity = False

    def write_pose(self, position, orientation):
        self.position, self.orientation = position, orientation
        self.velocity = (0., 0., 0.)


class LifecycleTests(unittest.TestCase):
    def test_keyboard_focus_repeat_and_emergency_release(self):
        carb = ModuleType('carb')
        carb.input = ModuleType('carb.input')
        carb.input.KeyboardEventType = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
        c = self.controller()
        c._on_physics_pre(.01, None)
        def event(key, kind):
            return SimpleNamespace(input=SimpleNamespace(name=key), type=kind)
        with patch.dict(sys.modules, {'carb': carb, 'carb.input': carb.input}):
            self.assertFalse(c._on_keyboard_event(event('I', 1)))
            self.assertIn('I', c.keys.pressed)
            c._on_keyboard_event(event('X', 1))
            c._on_keyboard_event(event('W', 1))
            self.assertFalse(c.keys.pressed)
            self.assertTrue(c._emergency_held)
            c._on_keyboard_event(event('X', 3))
            c._on_keyboard_event(event('W', 2))
            self.assertFalse(c.keys.pressed)
            c._has_control_focus = lambda: False
            self.assertTrue(c._on_keyboard_event(event('W', 1)))
            self.assertFalse(c.keys.pressed)

    def test_success_key_marks_task_once_only_while_controller_has_focus(self):
        carb = ModuleType('carb')
        carb.input = ModuleType('carb.input')
        carb.input.KeyboardEventType = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
        marked = []
        c = self.controller()
        c._on_physics_pre(.01, None)
        c.task_success_callback = lambda: marked.append('success') or True
        def event(key, kind):
            return SimpleNamespace(input=SimpleNamespace(name=key), type=kind)
        with patch.dict(sys.modules, {'carb': carb, 'carb.input': carb.input}):
            c._on_keyboard_event(event('T', 1))
            c._on_keyboard_event(event('T', 2))
            c._on_keyboard_event(event('T', 3))
            c._on_keyboard_event(event('T', 1))
        self.assertEqual(marked, ['success'])
        self.assertTrue(c.task_success_marked)

    def test_success_key_does_not_mark_without_focus_or_recording_callback(self):
        carb = ModuleType('carb')
        carb.input = ModuleType('carb.input')
        carb.input.KeyboardEventType = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
        marked = []
        c = self.controller()
        c._on_physics_pre(.01, None)
        c.task_success_callback = lambda: marked.append('success') or True
        def event():
            return SimpleNamespace(input=SimpleNamespace(name='T'), type=1)
        with patch.dict(sys.modules, {'carb': carb, 'carb.input': carb.input}):
            c._has_control_focus = lambda: False
            self.assertTrue(c._on_keyboard_event(event()))
            self.assertEqual(marked, [])
            c._has_control_focus = lambda: True
            c.task_success_callback = None
            self.assertFalse(c._on_keyboard_event(event()))
        self.assertFalse(c.task_success_marked)

    def test_tilt_does_not_change_translation_height(self):
        c = self.controller()
        c._on_physics_pre(.01, None)
        c.keys.press('I')
        c.keys.press('J')
        c.keys.press('W')
        c.yaw = 1.5707963267948966
        for _ in range(100):
            c._on_physics_pre(.01, None)
        self.assertAlmostEqual(c.position[0], 2)
        self.assertGreater(c.position[1], 3)
        self.assertEqual(c.position[2], 4)

    def test_not_ready_timeout_stops_once(self):
        c = self.controller()
        c._runtime.ready = False
        with patch('scripts.drone_keyboard_controller.time.monotonic', side_effect=[1., 12.]):
            c._on_update(None)
            c._on_update(None)
        self.assertFalse(c._running)
        self.assertIn('10 seconds', c._error)

    def test_lost_physics_view_times_out_even_without_physics_callbacks(self):
        c = self.controller()
        c._on_physics_pre(.01, None)
        c._runtime.ready = False
        with patch('scripts.drone_keyboard_controller.time.monotonic', side_effect=[1., 12.]):
            c._on_update(None)
            c._on_update(None)
        self.assertFalse(c._running)

    def controller(self):
        c = DroneKeyboardController()
        c._runtime = FakePhysics()
        c._running = True
        c._playing = lambda: True
        c._has_control_focus = lambda: True
        return c

    def test_first_step_clears_falling_velocity_and_holds_actual_world_pose(self):
        c = self.controller()
        for _ in range(1800):
            c._on_physics_pre(1/60, None)
            c._on_physics_post(1/60, None)
        self.assertEqual(c.position, (2., 3., 4.))
        self.assertEqual(c._runtime.position, (2., 3., 4.))
        self.assertEqual(c._runtime.velocity, (0., 0., 0.))
        self.assertFalse(c._runtime.gravity)

    def test_no_target_drift_before_physics_ready(self):
        c = self.controller()
        c._runtime.ready = False
        original = c.position
        c.keys.press('R')
        c._on_physics_pre(1., None)
        self.assertEqual(c.position, original)

    def test_emergency_freezes_tilt_until_release(self):
        c = self.controller()
        c._on_physics_pre(.01, None)
        c.keys.press('I')
        for _ in range(40):
            c._on_physics_pre(.01, None)
        self.assertGreater(c.pitch, 0)
        c.emergency_stop()
        pose = (c.position, c.roll, c.pitch, c.yaw)
        for _ in range(100):
            c._on_physics_pre(.01, None)
        self.assertEqual((c.position, c.roll, c.pitch, c.yaw), pose)
        c._emergency_held = False
        for _ in range(300):
            c._on_physics_pre(.01, None)
        self.assertAlmostEqual(c.pitch, 0)

    def test_pause_and_focus_loss_clear_inputs(self):
        c = self.controller()
        c._on_physics_pre(.01, None)
        c.keys.press('R')
        c._suspend()
        self.assertFalse(c.keys.pressed)
        c.keys.press('W')
        c._has_control_focus = lambda: False
        original = c.position
        c._on_physics_pre(.01, None)
        self.assertFalse(c.keys.pressed)
        self.assertEqual(c.position, original)

    def test_stop_play_reinitializes_from_actual_pose(self):
        c = self.controller()
        c._on_physics_pre(.01, None)
        c._suspend(stopped=True)
        c._runtime.position = (5., 6., 7.)
        c._runtime.gravity = True
        c._on_physics_pre(.01, None)
        self.assertEqual(c.position, (5., 6., 7.))
        self.assertFalse(c._runtime.gravity)

    def test_invalid_stage_stops_without_writing_old_pose(self):
        c = self.controller()
        physics = c._runtime
        physics.valid = False
        c._on_physics_pre(.01, None)
        self.assertFalse(c._running)
        self.assertEqual(physics.position, (2., 3., 4.))
