import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'third_party/isaac-thermal-camera-simulation/src'))
from scripts.live_motion_optimization import compact_motion
from irsim.optics.motion import image_plane_motion
from irsim.optics.projection import Intrinsics
from irsim.config.sensor import DistortionSpec


class MotionEquivalenceTests(unittest.TestCase):
    def test_camera_motion_and_object_motion_are_preserved(self):
        points = np.zeros((12, 16, 3))
        points[..., 0] = np.linspace(-1, 1, 16)
        points[..., 1] = np.linspace(-1, 1, 12)[:, None]
        points[..., 2] = -4
        ids = np.full((12, 16), 400, dtype=np.uint32)
        ids[:3] = 0
        previous = np.broadcast_to(np.eye(4), (500, 4, 4)).copy()
        current = previous.copy()
        current[400, 3, 0] = .2
        cam_prev = np.eye(4)
        cam_cur = np.eye(4)
        cam_cur[3, 1] = .1
        intrinsics = Intrinsics(fx_px=30., fy_px=30., cx_px=8., cy_px=6., width=16, height=12)
        args = (points, ids, previous, current, cam_prev, cam_cur, intrinsics,
                DistortionSpec(model='brown_conrady', coeffs=[0., 0., 0., 0., 0.]))
        expected = image_plane_motion(*args)
        actual = compact_motion(*args)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.any(actual[3:] != 0))
