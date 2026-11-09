import unittest
from types import SimpleNamespace
from scene_intake.forest_source_clock import source_thermal_clock,recorded_animation_time_code


class SourceClockTests(unittest.TestCase):
    def test_episode_animation_matches_recorded_source_even_after_clip_end(self):
        self.assertAlmostEqual(recorded_animation_time_code(4.1,60),246)
        self.assertAlmostEqual(recorded_animation_time_code(40.1,60),2406)
        from scene_intake.forest_collection import animation_time_code
        self.assertLess(animation_time_code(4.1,24,0,96),96)
    def test_looping_animation_does_not_rewind_thermal_time(self):
        from scripts.dual_camera_support import PreviewClock
        preview=SimpleNamespace(clock=PreviewClock())
        original=preview.clock
        with source_thermal_clock(preview,3.9):
            first=preview.clock.sample(3.9)
        with source_thermal_clock(preview,4.1):
            second=preview.clock.sample(.1)
            repeated=preview.clock.sample(.1)
        self.assertAlmostEqual(second[0]-first[0],.2)
        self.assertEqual(second,repeated)
        self.assertIs(preview.clock,original)

    def test_exception_restores_original_clock(self):
        clock=SimpleNamespace(sample=lambda stamp:(stamp,0))
        preview=SimpleNamespace(clock=clock)
        with self.assertRaises(RuntimeError):
            with source_thermal_clock(preview,8):
                self.assertEqual(preview.clock.sample(0),(8,0))
                raise RuntimeError('capture failed')
        self.assertIs(preview.clock,clock)


if __name__=='__main__':unittest.main()
