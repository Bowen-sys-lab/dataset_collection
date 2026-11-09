"""The collection HUD must not certify point distance or unobserved time as a hold."""
import hashlib
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.factory_pilot_hud import EquipmentClearance, ObservationTimer, projected_inside


class PilotHUDTests(unittest.TestCase):
    def test_headless_preview_with_target_does_not_touch_usd_or_ui(self):
        from scripts.start_crazyflie_dual_camera_view import DualCameraPreview
        preview = DualCameraPreview(None, None, None, None, None, width=160, height=128,
                                    cleanup=None, metadata={}, show_ui=False)
        preview.pilot_target = np.zeros(3)
        with patch.dict(sys.modules, {'pxr': None}):
            preview._update_pilot_hud(np.eye(4), 0.0)

    def test_equipment_surface_distance_subtracts_body_radius(self):
        # Triangle in plane x=0. Point x=7.2 has 7.0 m body clearance.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'tank.npz'
            np.savez(path, equipment_vertices=np.array([[0,-100,-100],[0,100,-100],[0,0,100]]),
                     equipment_triangles=np.array([[0,1,2]]))
            inspection = {'geometry_file': 'tank.npz',
                          'geometry_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'inspection_requirements': {'body_envelope_radius_m': 0.2}}
            geometry = EquipmentClearance.load(root, inspection)
            self.assertAlmostEqual(geometry.distance([7.2,0,0]), 7.0)
            inspection['geometry_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                EquipmentClearance.load(root, inspection)
            inspection['geometry_file'] = '../outside.npz'
            with self.assertRaises(ValueError):
                EquipmentClearance.load(root, inspection)

    def test_hold_requires_both_ends_of_every_interval_to_qualify(self):
        timer = ObservationTimer()
        timer.update(0.0, [0,0,0], 9.0, True)
        self.assertEqual(timer.update(0.1, [0,0,0], 7.0, True)['hold_s'], 0.0)
        self.assertAlmostEqual(timer.update(0.2, [0,0,0], 7.0, True)['hold_s'], 0.1)
        self.assertEqual(timer.update(0.3, [0,0,0], 7.0, False)['hold_s'], 0.0)

    def test_large_gaps_rewinds_and_motion_reset_timer(self):
        timer = ObservationTimer()
        for i in range(6):
            state = timer.update(i * 0.1, [0,0,0], 7.0, True)
        self.assertGreater(state['hold_s'], 0)
        self.assertEqual(timer.update(2.0, [0,0,0], 7.0, True)['hold_s'], 0.0)
        self.assertEqual(timer.update(0.0, [0,0,0], 7.0, True)['hold_s'], 0.0)
        self.assertEqual(timer.update(0.1, [1,0,0], 7.0, True)['hold_s'], 0.0)
        self.assertEqual(timer.update(0.2, [1,0,0], float('nan'), True)['hold_s'], 0.0)

    def test_three_seconds_needs_continuous_observed_intervals(self):
        timer = ObservationTimer()
        for i in range(32):
            state = timer.update(i * .1, [0,0,0], 7.0, True)
        self.assertAlmostEqual(state['hold_s'], 3.0)
        self.assertTrue(state['hold_complete'])

    def test_projection_uses_real_intrinsics_but_does_not_claim_occlusion(self):
        camera = np.eye(4)
        self.assertTrue(projected_inside([0,0,-10], camera, 160,128,18,20.955,16.764))
        self.assertFalse(projected_inside([0,0,10], camera, 160,128,18,20.955,16.764))
        self.assertFalse(projected_inside([100,0,-10], camera, 160,128,18,20.955,16.764))
        self.assertFalse(projected_inside([0,0,-10], camera, 160,128,0,20.955,16.764))


if __name__ == '__main__':
    unittest.main()
