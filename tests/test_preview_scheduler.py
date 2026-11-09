import unittest
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from scripts.preview_scheduler import PreviewScheduler, verify_renderer_evidence


class SchedulerTests(unittest.TestCase):
    def test_due_and_single_flight(self):
        s = PreviewScheduler(10)
        self.assertTrue(s.begin(0, 1))
        self.assertFalse(s.begin(.5, 2))
        self.assertEqual(s.finish(0, .05, 1), 0)
        self.assertFalse(s.due(.09))
        self.assertTrue(s.due(.1))

    def test_expired_and_duplicate_do_not_get_ids(self):
        s = PreviewScheduler(10, .2)
        s.begin(0, 1)
        self.assertIsNone(s.finish(0, .3, 1))
        s.begin(.3, 2)
        self.assertEqual(s.finish(.3, .35, 2), 0)
        self.assertFalse(s.begin(.5, 2))
        s.begin(.7, 3)
        self.assertEqual(s.finish(.7, .75, 3), 1)
        self.assertGreaterEqual(s.dropped, 2)

    def test_invalid_and_slow_capture(self):
        s = PreviewScheduler(10)
        s.begin(0)
        self.assertIsNone(s.finish(0, .05, valid=False))
        s.begin(1)
        self.assertGreaterEqual(s.dropped, 9)
        self.assertEqual(s.finish(1, 1.05), 0)
        with self.assertRaises(RuntimeError):
            s.finish(1, 2)

    def test_invalid_parameters(self):
        for rate, age in ((-1, .1), (float('nan'), .1), (10, 0)):
            with self.assertRaises(ValueError):
                PreviewScheduler(rate, age)

    def test_zero_rate_is_uncapped_but_still_single_flight(self):
        s = PreviewScheduler(0)
        self.assertTrue(s.uncapped)
        self.assertTrue(s.begin(0, 1))
        self.assertFalse(s.begin(.001, 2))
        self.assertEqual(s.finish(0, .001, 1), 0)
        self.assertTrue(s.begin(.001, 2))

    def test_renderer_evidence_rejects_changed_sources_and_wrong_resolution(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source'
            source.write_text('immutable')
            config = SimpleNamespace(sha256='config', scene=SimpleNamespace(stage=source),
                sensors=SimpleNamespace(thermal_config=source, sensor_config=source))
            proof = {'result': 'PASS', 'strict_per_frame_pixel_sync_verified': True,
                     'renderer': 'RealTimePathTracing', 'resolution': [160, 128],
                     'scene_config_sha256': 'config', 'experiment_conditions': {'files': {
                         key: {'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
                         for key in ('usd', 'thermal_config', 'sensor_config')}}}
            path = root/'proof.json'
            path.write_text(json.dumps(proof))
            verify_renderer_evidence(path, config, 'RealTimePathTracing', 160, 128)
            with self.assertRaises(ValueError):
                verify_renderer_evidence(path, config, 'RealTimePathTracing', 320, 256)
            source.write_text('changed')
            with self.assertRaises(ValueError):
                verify_renderer_evidence(path, config, 'RealTimePathTracing', 160, 128)
            with self.assertRaises(ValueError):
                verify_renderer_evidence(None, config, 'RealTimePathTracing', 160, 128)


if __name__ == '__main__':
    unittest.main()
