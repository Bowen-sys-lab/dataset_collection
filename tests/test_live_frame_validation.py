import unittest
from types import SimpleNamespace
import numpy as np
from scripts.live_frame_validation import (LiveFrameValidator, PreviewContentTracker,
                                          driving_gates, match_pair_history)


def frame(x, noise=0):
    rgb = np.zeros((32, 48, 3), np.uint8) + noise
    rgb[12:19, x-3:x+4] = (0, 255, 0)
    mask = np.zeros((32, 48), bool)
    mask[12:19, x-3:x+4] = True
    t = np.full(mask.shape, 290.)
    t[mask] = 350.
    thermal = SimpleNamespace(display8=np.where(mask, 240, noise).astype(np.uint8), apparent_t=t)
    return rgb, thermal, mask


class LiveEvidenceTests(unittest.TestCase):
    def test_report_counts_repeated_hashes_without_claiming_rejected_frames(self):
        tracker = PreviewContentTracker()
        rgb, thermal, _ = frame(10)
        tracker.inspect(rgb, thermal.display8)
        tracker.inspect(rgb, thermal.display8)
        report = tracker.report()
        self.assertEqual(report['hash_repeated_pair_diagnostics'], 1)
        self.assertNotIn('exact_duplicate_content_rejections', report)
        self.assertFalse(report['old_aov_frames_verified'])

    def test_equal_hash_is_unknown_not_a_rejected_frame(self):
        tracker = PreviewContentTracker()
        rgb, thermal, _ = frame(10)
        tracker.inspect(rgb, thermal.display8)
        same = tracker.inspect(rgb, thermal.display8)
        self.assertEqual(same['freshness_status'], 'unknown_from_hash')
        self.assertFalse(same['reject_on_hash'])

    def test_pair_matches_same_history_without_claiming_current_pose(self):
        history = [{'physics_step': 0, 'simulation_time_s': .1, 'wall_s': 1., 'expected_uv': [10, 15]},
                   {'physics_step': 1, 'simulation_time_s': .2, 'wall_s': 1.04, 'expected_uv': [30, 15]}]
        sample = match_pair_history([10, 15], [10, 15], [10, 15], history, 1.08)
        self.assertTrue(sample['same_historical_marker_state_verified'])
        self.assertFalse(sample['current_marker_state_verified'])
        self.assertAlmostEqual(sample['marker_edit_to_read_ms'], 80.)
        self.assertEqual(sample['matched_marker_simulation_time_s'], .1)
        mismatch = match_pair_history([10, 15], [30, 15], [30, 15], history, 1.08)
        self.assertFalse(mismatch['same_historical_marker_state_verified'])
    def test_preview_duplicate_rejection_is_not_a_freshness_certificate(self):
        tracker = PreviewContentTracker()
        rgb, thermal, _ = frame(10)
        self.assertTrue(tracker.inspect(rgb, thermal.display8)['pair_content_changed'])
        self.assertFalse(tracker.inspect(rgb, thermal.display8)['pair_content_changed'])
        noisy_rgb, noisy_thermal, _ = frame(10, 2)
        sample = tracker.inspect(noisy_rgb, noisy_thermal.display8)
        self.assertTrue(sample['pair_content_changed'])
        self.assertFalse(sample['fresh_current_state_verified'])
        self.assertEqual(tracker.rejected, 0)
        self.assertEqual(tracker.repeated_pairs, 1)

    def test_current_and_old_state_even_when_noise_changes_hash(self):
        validator = LiveFrameValidator()
        self.assertTrue(validator.inspect(*frame(10), [10, 15])['fresh_current_state_verified'])
        stale = validator.inspect(*frame(10, 2), [30, 15])
        self.assertFalse(stale['rgb_hash_repeated'])
        self.assertTrue(stale['rgb_mask_repeated'])
        self.assertIsNone(stale['frame_id'])
        fresh = validator.inspect(*frame(30), [30, 15])
        self.assertEqual(fresh['frame_id'], 1)

    def test_one_sensor_stale_fails(self):
        v = LiveFrameValidator()
        rgb, _, _ = frame(30)
        _, thermal, geometry = frame(10)
        self.assertFalse(v.inspect(rgb, thermal, geometry, [30, 15])['fresh_current_state_verified'])

    def test_empty_or_bad_latency_cannot_pass(self):
        for latency, gaps in (([], []), ([.06], [.05]), ([.03], [.11])):
            gates = driving_gates(viewport_enabled=True, latencies=latency,
                loop_intervals=gaps, samples=[], has_thermal=True)
            self.assertFalse(all(gates.values()))


if __name__ == '__main__':
    unittest.main()
