"""Regression tests for isolated RGB replay render-quality diagnostics."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from scripts.replay_trajectory_highres import (
    TONEMAP_LOCK_SETTINGS,
    apply_tonemap_lock,
    build_parser,
)


class _Settings:
    def __init__(self, ignored_keys=()):
        self.values = {}
        self.ignored_keys = set(ignored_keys)

    def set(self, key, value):
        if key not in self.ignored_keys:
            self.values[key] = value

    def get(self, key):
        return self.values.get(key)


class ReplayRenderOptionsTests(unittest.TestCase):
    def _base_args(self, *extra):
        return build_parser().parse_args([
            '--trajectory', 'trajectory.jsonl', '--out', 'output', *extra,
        ])

    def test_default_replay_keeps_single_sample_path_tracing(self):
        args = self._base_args()

        self.assertEqual(args.samples_per_pixel, 1)
        self.assertFalse(args.lock_current_tonemap)

    def test_default_replay_uses_square_paired_sensor_resolutions(self):
        args = self._base_args()

        self.assertEqual((args.rgb_width, args.rgb_height), (256, 256))
        self.assertEqual((args.thermal_width, args.thermal_height), (256, 256))

    def test_diagnostic_accepts_higher_sample_count_without_changing_other_defaults(self):
        args = self._base_args('--samples-per-pixel', '8', '--lock-current-tonemap')

        self.assertEqual(args.samples_per_pixel, 8)
        self.assertTrue(args.lock_current_tonemap)
        self.assertEqual((args.rgb_width, args.rgb_height), (256, 256))
        self.assertEqual((args.thermal_width, args.thermal_height), (256, 256))
        self.assertEqual(args.sample_interval, 1.0)

    def test_legacy_resolution_override_sets_both_streams_to_the_same_size(self):
        args = self._base_args('--width', '640', '--height', '512')

        from scripts.replay_trajectory_highres import resolve_capture_resolutions
        self.assertEqual(resolve_capture_resolutions(args), ((640, 512), (640, 512)))

    def test_custom_stream_resolutions_must_keep_the_same_aspect_ratio(self):
        args = self._base_args('--rgb-width', '512', '--rgb-height', '410',
                               '--thermal-width', '256', '--thermal-height', '128')

        from scripts.replay_trajectory_highres import resolve_capture_resolutions
        with self.assertRaisesRegex(ValueError, 'aspect ratio'):
            resolve_capture_resolutions(args)

    def test_sample_count_must_be_positive(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self._base_args('--samples-per-pixel', '0')

    def test_tonemap_lock_sets_and_reads_back_every_recorded_current_value(self):
        settings = _Settings()

        actual = apply_tonemap_lock(settings)

        self.assertEqual(actual, TONEMAP_LOCK_SETTINGS)
        self.assertEqual(settings.values, TONEMAP_LOCK_SETTINGS)

    def test_tonemap_lock_fails_if_kit_does_not_apply_a_setting(self):
        first_key = next(iter(TONEMAP_LOCK_SETTINGS))
        settings = _Settings(ignored_keys=(first_key,))

        with self.assertRaisesRegex(RuntimeError, first_key):
            apply_tonemap_lock(settings)

    def test_changed_recorded_usd_is_rejected_before_starting_kit_with_diagnostics(self):
        from scripts.replay_trajectory_highres import main
        root = Path(__file__).resolve().parents[1]
        fake_isaac = ModuleType('isaacsim')
        fake_isaac.SimulationApp = lambda options: self.fail('Kit started before checking the recorded USD')
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trajectory = directory / 'trajectory.jsonl'
            trajectory.write_text('{"frame_id":0,"simulation_time":0.0}\n', encoding='utf-8')
            (directory / 'run_metadata.json').write_text(
                json.dumps({'stage_sha256': '0' * 64}), encoding='utf-8')
            output = directory / 'replay'
            with patch.dict(sys.modules, {'isaacsim': fake_isaac}):
                with self.assertRaisesRegex(RuntimeError, 'recorded.*USD'):
                    main(['--config', str(root / 'configs/crazyflie_scene.yaml'),
                          '--trajectory', str(trajectory), '--out', str(output)])
            self.assertIn('recorded', (output / 'failure.txt').read_text(encoding='utf-8'))
            self.assertFalse((output / 'replay_metadata.json').exists())


if __name__ == '__main__':
    unittest.main()
