"""The cache must preserve angles, interpolate radiance (not temperature), and roll forward."""
import unittest
import numpy as np
from scripts.preview_sky import PreviewSky


class LinearSky:
    # Analytic sky with nonlinear temperature conversion; no Isaac needed.
    class LUT:
        def apparent_temperature(self, radiance, quantity):
            return np.sqrt(radiance)
    lut = LUT()
    quantity = 'lb'

    def radiance(self, t, elevation):
        return 100.0 + 2*t + np.asarray(elevation)*10

    def effective_radiance(self, t, tilt):
        return 90.0 + 3*t + np.asarray(tilt)*5


class PreviewSkyTests(unittest.TestCase):
    def test_interpolates_radiance_then_inverts_and_preserves_angles(self):
        with PreviewSky(LinearSky(), 100.0, interval_s=10) as sky:
            angles = np.array([[0., np.pi/4, np.pi/2]])
            expected = np.sqrt(310.0 + angles*10)
            np.testing.assert_allclose(sky.apparent_temperature_k(105, angles), expected)
            np.testing.assert_allclose(sky.effective_radiance_from_sky_view(105, [1., .5, 0.]),
                                       [405., 405.+np.pi*2.5, 405.+np.pi*5])

    def test_rolls_forward_skips_time_and_accepts_replay_without_stale_results(self):
        with PreviewSky(LinearSky(), 100.0, interval_s=10) as sky:
            for t in [100., 110., 125., 1000., 105.]:
                np.testing.assert_allclose(sky.radiance(t, [0.]), [100.+2*t])
                self.assertLessEqual(sky.cached_samples, 3)

    def test_invalid_interval_is_rejected(self):
        for interval in [0., -1., float('nan')]:
            with self.assertRaises(ValueError):
                PreviewSky(LinearSky(), 100., interval_s=interval)


if __name__ == '__main__':
    unittest.main()
