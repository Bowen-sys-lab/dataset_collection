"""Guard the opt-in USD lighting settings consumed by the existing recorder."""
import unittest

from scripts.authored_lighting import validate_settings


class AuthoredLightingTests(unittest.TestCase):
    def profile(self):
        return {'/rtx/post/tonemap/op': 6,
                '/rtx/post/tonemap/filmIso': 400.,
                '/rtx/post/tonemap/cameraShutter': 30.,
                '/rtx/post/tonemap/fNumber': 2.8,
                '/rtx/post/histogram/enabled': False}

    def test_no_metadata_has_no_settings(self):
        self.assertEqual(validate_settings(None), {})

    def test_manual_exposure_is_preserved(self):
        settings = validate_settings(self.profile())
        self.assertEqual(settings['/rtx/post/tonemap/filmIso'], 400.)
        self.assertIs(settings['/rtx/post/histogram/enabled'], False)

    def test_unknown_runtime_setting_is_rejected(self):
        profile = self.profile()
        profile['/app/arbitrarySetting'] = True
        with self.assertRaises(ValueError):
            validate_settings(profile)

    def test_nonfinite_exposure_is_rejected(self):
        for value in (float('nan'), float('inf'), 0., -1.):
            with self.subTest(value=value):
                profile = self.profile()
                profile['/rtx/post/tonemap/fNumber'] = value
                with self.assertRaises(ValueError):
                    validate_settings(profile)

    def test_auto_exposure_cannot_brighten_the_night(self):
        profile = self.profile()
        profile['/rtx/post/histogram/enabled'] = True
        with self.assertRaises(ValueError):
            validate_settings(profile)


if __name__ == '__main__':
    unittest.main()
