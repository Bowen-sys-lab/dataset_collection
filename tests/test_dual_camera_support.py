import unittest
from dataclasses import dataclass
import numpy as np
from scripts import dual_camera_support as support


@dataclass(frozen=True)
class _Intrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width: int
    height: int


@dataclass(frozen=True)
class _CameraOptics:
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    intrinsics: _Intrinsics
    supersample: int


class ImageTests(unittest.TestCase):
    def test_array_upload_preserves_flipped_pixel_order_and_alpha(self):
        data, size = support.ui_rgba_array(np.array([[[1,2,3],[4,5,6]]], dtype=np.uint8)[:, ::-1])
        np.testing.assert_array_equal(data, np.array([4,5,6,255,1,2,3,255], dtype=np.uint8))
        self.assertEqual(size, [2, 1])
        self.assertTrue(data.flags.c_contiguous)

    def test_gray_is_repeated_into_rgb_with_opaque_alpha(self):
        data, size = support.rgba_bytes(np.array([[0, 90, 255]], dtype=np.uint8))
        self.assertEqual(size, [3, 1])
        self.assertEqual(data, bytes([0,0,0,255,90,90,90,255,255,255,255,255]))

    def test_noncontiguous_rgb_preserves_pixel_order(self):
        img = np.array([[[1,2,3],[4,5,6]]], dtype=np.uint8)[:, ::-1]
        data, size = support.rgba_bytes(img)
        self.assertEqual(data, bytes([4,5,6,255,1,2,3,255]))
        self.assertEqual(size, [2, 1])

    def test_rejects_float_raw_temperature_instead_of_silent_wrapping(self):
        with self.assertRaises(ValueError):
            support.rgba_bytes(np.array([[293.15]], dtype=np.float32))

    def test_rejects_empty_and_two_channel_data(self):
        for img in [np.zeros((0,2), dtype=np.uint8), np.zeros((2,2,2), dtype=np.uint8)]:
            with self.assertRaises(ValueError):
                support.rgba_bytes(img)

    def test_ui_payload_uses_integer_sequence_accepted_by_isaac_61(self):
        data, size = support.rgba_bytes(np.array([[1, 2]], dtype=np.uint8))
        ui_data, ui_size = support.ui_rgba_payload(np.array([[1, 2]], dtype=np.uint8))
        self.assertIsInstance(ui_data, list)
        self.assertEqual(ui_data, list(data))
        self.assertEqual(ui_size, size)


class ClockTests(unittest.TestCase):
    def test_capture_clock_tracks_simulation_not_number_of_frames(self):
        clock = support.PreviewClock()
        self.assertEqual(clock.sample(7.0), (0.0, 0))
        self.assertAlmostEqual(clock.sample(7.3)[0], .3)
        self.assertAlmostEqual(clock.sample(7.3)[0], .3)
        self.assertAlmostEqual(clock.sample(8.0)[0], 1.0)

    def test_stop_replay_starts_new_epoch_without_reversing_thermal_solver(self):
        clock = support.PreviewClock()
        clock.sample(0.0)
        clock.sample(1.0)
        self.assertEqual(clock.sample(0.0), (1.0, 1))
        self.assertEqual(clock.sample(.5), (1.5, 1))

    def test_invalid_time_does_not_poison_clock(self):
        clock = support.PreviewClock()
        clock.sample(1.0)
        with self.assertRaises(ValueError):
            clock.sample(float('nan'))
        self.assertEqual(clock.sample(2.0), (1.0, 0))


class PreviewConfigTests(unittest.TestCase):
    def test_preserves_horizontal_fov_without_modifying_source_config(self):
        original = {'sensor': {'fpa': {'width': 640, 'height': 512, 'pitch_um': 12.0},
                               'optics': {'focal_length_mm': 14., 'supersample_factor': 4}}}
        result = support.preview_sensor_data(original, 320, 240, 25., 20.)
        self.assertEqual(result['sensor']['fpa']['height'], 240)
        self.assertAlmostEqual(result['sensor']['optics']['focal_length_mm'], 4.8)
        self.assertEqual(result['sensor']['optics']['supersample_factor'], 1)
        self.assertEqual(original['sensor']['fpa']['height'], 512)
        self.assertEqual(original['sensor']['optics']['focal_length_mm'], 14.)

    def test_rejects_invalid_resolution_or_optics(self):
        for width, height, focal, aperture in [(0,480,25,20), (640,-1,25,20),
                                              (640,480,0,20), (640,480,25,float('nan'))]:
            with self.assertRaises(ValueError):
                support.preview_sensor_data({}, width, height, focal, aperture)

    def test_rgb_intrinsics_scale_to_the_higher_output_resolution_without_changing_fov(self):
        thermal_optics = _CameraOptics(
            3.84, 3.072, 2.46, _Intrinsics(320.0, 320.0, 128.0, 102.5, 256, 205), 1)

        rgb_optics = support.scale_camera_optics_for_resolution(thermal_optics, 512, 410)

        self.assertEqual(rgb_optics.intrinsics, _Intrinsics(640.0, 640.0, 256.0, 205.0, 512, 410))
        self.assertEqual(rgb_optics.focal_length_mm, thermal_optics.focal_length_mm)
        self.assertEqual(rgb_optics.horizontal_aperture_mm, thermal_optics.horizontal_aperture_mm)
        self.assertEqual(rgb_optics.vertical_aperture_mm, thermal_optics.vertical_aperture_mm)
        self.assertEqual(thermal_optics.intrinsics.width, 256)


class CleanupTests(unittest.TestCase):
    def test_reverse_order_and_idempotency(self):
        calls = []
        stack = support.CleanupStack()
        stack.add(lambda: calls.append('camera'))
        stack.add(lambda: calls.append('reader'))
        self.assertEqual(stack.close(), [])
        stack.close()
        self.assertEqual(calls, ['reader', 'camera'])

    def test_failed_cleanup_does_not_skip_remaining_resources(self):
        calls = []
        stack = support.CleanupStack()
        stack.add(lambda: calls.append('restored'))
        def fail():
            raise RuntimeError('detach failed')
        stack.add(fail)
        self.assertEqual(stack.close(), ['RuntimeError: detach failed'])
        self.assertEqual(calls, ['restored'])
        self.assertEqual(stack.close(), [])


class CaptureLifecycleTests(unittest.TestCase):
    def test_pause_discards_incomplete_pair_instead_of_publishing_old_buffer(self):
        self.assertEqual(support.capture_interruption_reason(
            app_running=True, window_visible=True, timeline_playing=False),
            'timeline_paused_or_stopped')

    def test_closing_window_takes_priority_over_pause(self):
        for app_running, window_visible in ((False, True), (True, False)):
            self.assertEqual(support.capture_interruption_reason(
                app_running=app_running, window_visible=window_visible,
                timeline_playing=False), 'window_closed')

    def test_playing_render_failure_is_not_hidden(self):
        self.assertIsNone(support.capture_interruption_reason(
            app_running=True, window_visible=True, timeline_playing=True))


if __name__ == '__main__':
    unittest.main()
