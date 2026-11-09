import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.report_camera_acceptance import build_report


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'source'
        source.write_text('fixed input', encoding='utf-8')
        record = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        self.conditions = {'files': {key: record for key in
                           ('scene_config', 'thermal_config', 'sensor_config', 'usd', 'thermal_environment_config')},
                           'resolution': [640, 512], 'flat_field_enabled': True, 'sensor_seed': 1,
                           'ao_aov_enabled': False, 'default_occlusion': 1.0}
        self.sync = {'result': 'PASS', 'strict_per_frame_pixel_sync_verified': True,
                     'source_unchanged': True, 'failed_frames': [], 'frames': [{'pass': True}],
                     'experiment_conditions': self.conditions}
        self.replay = {'result': 'PASS_FILE_AND_ARRAY_CHECKS', 'failures': [],
                       'source_usd_unchanged': True, 'frames_validated': 8,
                       'experiment_conditions': self.conditions}
        self.gui = {'result': 'PASS', 'checks': {key: True for key in
                    ('forward', 'lateral', 'vertical', 'yaw', 'paired_updates',
                     'pause_resume', 'no_visible_stale_frames', 'camera_follow')}}
        self.args = SimpleNamespace(**{name: self.root / (name + '.json') for name in
                                     ('sync', 'replay', 'gui', 'sync_log', 'replay_log', 'gui_log')})
        for name in ('sync_log', 'replay_log', 'gui_log'):
            getattr(self.args, name).write_text('no critical errors', encoding='utf-8')

    def report(self):
        for name in ('sync', 'replay', 'gui'):
            getattr(self.args, name).write_text(json.dumps(getattr(self, name)), encoding='utf-8')
        return build_report(self.args)

    def test_all_evidence_required_for_promotion(self):
        self.assertEqual(self.report()['benchmark_status'], 'benchmark-ready')

    def test_gui_render_error_blocks_even_with_positive_human_feedback(self):
        self.args.gui_log.write_text('[Error] Explicit render did not complete', encoding='utf-8')
        self.assertEqual(self.report()['benchmark_status'], 'pilot/development data')

    def test_missing_gui_check_blocks(self):
        del self.gui['checks']['pause_resume']
        self.assertFalse(self.report()['gates']['gui_human_observation'])

    def test_failed_pixel_frame_blocks(self):
        self.sync['frames'][0]['pass'] = False
        self.assertFalse(self.report()['gates']['strict_marker_replay_sync'])

    def test_modified_input_hash_blocks(self):
        (self.root / 'source').write_text('changed', encoding='utf-8')
        self.assertFalse(self.report()['gates']['experiment_hashes_present_current_and_matching'])

    def test_runtime_resolution_mismatch_blocks(self):
        self.sync['experiment_conditions'] = {**self.conditions, 'resolution': [160, 128]}
        self.assertFalse(self.report()['gates']['capture_conditions_matching'])


if __name__ == '__main__':
    unittest.main()
