"""Checks for the image statistics used by the RGB renderer A/B report."""
import unittest

import numpy as np

from scripts.compare_rgb_quality import image_statistics, array_difference


class RgbQualityComparisonTests(unittest.TestCase):
    def test_image_statistics_separates_luma_from_color_saturation(self):
        pixels = np.asarray([[[128, 128, 128, 255], [255, 0, 0, 255]]], dtype=np.uint8)

        stats = image_statistics(pixels)

        self.assertEqual(stats['mean_rgb'], [191.5, 64.0, 64.0])
        self.assertAlmostEqual(stats['mean_luma'], 91.1065, places=3)
        self.assertAlmostEqual(stats['mean_hsv_saturation'], 0.5, places=6)
        self.assertEqual(stats['clipped_channel_fraction'], 0.5)

    def test_array_difference_detects_changed_thermal_output(self):
        same, different = np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)

        self.assertTrue(array_difference(same, same)['exact_equal'])
        self.assertFalse(array_difference(same, different)['exact_equal'])
        self.assertEqual(array_difference(same, different)['max_absolute_difference'], 1.0)


if __name__ == '__main__':
    unittest.main()
