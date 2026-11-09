import unittest

import numpy as np

from scripts.crazyflie_thermal_capture import validate_frame_contract


class CrazyflieThermalCaptureTests(unittest.TestCase):
    def test_accepts_registered_rgb_and_thermal_planes(self):
        shape = (2, 3)
        result = validate_frame_contract(
            rgb=np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
            radiance=np.arange(6, dtype=np.float32).reshape(shape),
            apparent_t=(280.0 + np.arange(6, dtype=np.float32)).reshape(shape),
            dn16=np.arange(6, dtype=np.uint16).reshape(shape),
            display8=np.arange(6, dtype=np.uint8).reshape(shape),
            expected_shape=shape,
        )
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["rgb_shape"], [2, 3, 4])

    def test_rejects_nonfinite_thermal_plane(self):
        with self.assertRaisesRegex(ValueError, "radiance contains non-finite"):
            validate_frame_contract(
                rgb=np.ones((2, 3, 3), dtype=np.uint8),
                radiance=np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=np.float32),
                apparent_t=np.arange(6, dtype=np.float32).reshape(2, 3),
                dn16=np.arange(6, dtype=np.uint16).reshape(2, 3),
                display8=np.arange(6, dtype=np.uint8).reshape(2, 3),
                expected_shape=(2, 3),
            )


if __name__ == "__main__":
    unittest.main()
