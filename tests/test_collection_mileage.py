import tempfile
import unittest
from pathlib import Path

from scripts import collection_mileage as mileage


class MileageTests(unittest.TestCase):
    def test_live_distance_is_labelled_as_anchor_distance(self):
        tracker=mileage.MileageTracker()
        tracker.update(0,[0,0,0])
        self.assertIn('anchor 5.0m',tracker.status_text(target=[3,4,0]))
        self.assertEqual(tracker.length_3d_m,0)
    def test_counts_real_3d_motion_and_not_paused_states(self):
        tracker = mileage.MileageTracker()
        tracker.update(0, [0, 0, 0])
        tracker.update(.1, [.3, 0, .4])
        tracker.update(.1, [.3, 0, .4])
        self.assertAlmostEqual(tracker.length_3d_m, .5)
        self.assertAlmostEqual(tracker.length_xy_m, .3)
        self.assertEqual(tracker.summary()['measurement_status'], 'COMPLETE')

    def test_rollback_gap_jump_and_bad_pose_never_count_as_valid_distance(self):
        for time, position in [(-1, [2, 0, 0]), (2, [2, 0, 0]), (.1, [100, 0, 0]),
                               (.1, [float('nan'), 0, 0]), (0, [1, 0, 0])]:
            with self.subTest(time=time, position=position):
                tracker = mileage.MileageTracker()
                tracker.update(0, [0, 0, 0])
                tracker.update(time, position)
                self.assertEqual(tracker.length_3d_m, 0)
                self.assertEqual(tracker.summary()['measurement_status'], 'INCOMPLETE')
                self.assertTrue(tracker.summary()['issues'])

    def test_soft_length_labels_never_qualify_or_block_task(self):
        self.assertEqual(mileage.length_band(99), 'BELOW_TARGET')
        self.assertEqual(mileage.length_band(100), 'IN_RANGE')
        self.assertEqual(mileage.length_band(200), 'IN_RANGE')
        self.assertEqual(mileage.length_band(201), 'ABOVE_TARGET')
        tracker = mileage.MileageTracker()
        self.assertFalse(tracker.summary()['task_gate_applied'])

    def test_summary_includes_exact_t_point_and_excludes_post_confirmation(self):
        rows = [{'simulation_time': i*.1, 'position': [i*.2, 0, 0]} for i in range(5)]
        meta = {'task_result': {'status': 'success', 'marked_by': 'keyboard:T',
                'simulation_time_s': .15, 'drone_position_m': [.3, 0, 0]}}
        report = mileage.summarize_mileage(rows, meta)
        self.assertAlmostEqual(report['to_confirmation']['length_3d_m'], .3)
        self.assertAlmostEqual(report['raw_recording']['length_3d_m'], .8)
        self.assertEqual(report['to_confirmation']['endpoint_source'], 'recorded_keyboard_T_event')
        self.assertIsNone(mileage.summarize_mileage(rows, {})['to_confirmation'])

    def test_recorder_preserves_short_manual_confirmation_and_writes_separate_quality(self):
        import json
        from scripts.trajectory_recorder import TrajectoryRecorder
        with tempfile.TemporaryDirectory() as folder:
            recorder = TrajectoryRecorder.create(folder, {'task': {'task_id': 'example'}}, 'run')
            recorder.append({'frame_id': 0, 'simulation_time': 0, 'position': [0,0,0]})
            recorder.append({'frame_id': 1, 'simulation_time': .1, 'position': [.1,0,0]})
            self.assertTrue(recorder.mark_success(.15, [.15,0,0]))
            recorder.close()
            quality = json.loads((Path(folder)/'run/mileage_quality.json').read_text())
            self.assertEqual(quality['to_confirmation']['length_band'], 'BELOW_TARGET')
            self.assertEqual(recorder.metadata['task_result']['status'], 'success')


if __name__ == '__main__':
    unittest.main()
