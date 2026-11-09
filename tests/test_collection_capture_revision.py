"""Collection defaults and cadence regressions, without starting Isaac."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


class CollectionCaptureRevisionTests(unittest.TestCase):
    def test_live_default_records_one_hz_without_lowering_preview_rates(self):
        import argparse
        from unittest.mock import patch
        from scripts import start_crazyflie_dual_camera_view as live
        parse_args = argparse.ArgumentParser.parse_args
        parsed = []

        class ParsedBeforeStartup(Exception):
            pass

        def capture(parser, argv):
            parsed.append(parse_args(parser, argv))
            raise ParsedBeforeStartup()

        with patch.object(argparse.ArgumentParser, 'parse_args', capture):
            with self.assertRaises(ParsedBeforeStartup):
                live.main([])
        self.assertEqual(parsed[0].trajectory_period, 1.)
        self.assertEqual(parsed[0].preview_fps, 15.)
        self.assertEqual(parsed[0].thermal_preview_fps, 2.)
        from scripts.replay_trajectory_highres import build_parser
        offline = build_parser().parse_args(['--trajectory', 'raw.jsonl', '--out', 'out'])
        self.assertEqual(offline.sample_interval, 1.)

    def test_scene_specific_launchers_use_one_hz_and_keep_preview_rates(self):
        from unittest.mock import patch
        from scripts.collection_capture_defaults import TRAJECTORY_PERIOD_S
        from scripts.city_collection_config import resolve_selection
        from scripts.city_bc_pilot import build_launch as city_launch
        from scripts import factory_bc_pilot
        from scripts.factory_collection_config import resolve_selection as factory_selection

        self.assertEqual(TRAJECTORY_PERIOD_S, 1.)
        self.assertEqual(resolve_selection()['task']['environment_evidence']['city_target']['record_period_s'], 1.)
        city = city_launch('city-bc-probe-cadence-test', smoke_frames=1)
        # The factory readiness hash guards native collection, not this CLI cadence contract.
        with patch.object(factory_bc_pilot, 'resolve_selection',
                          side_effect=lambda *args: factory_selection(*args, require_ready=False)):
            factory = factory_bc_pilot.build_launch('factory-bc-probe-cadence-test', smoke_frames=1)
        for launch in (city, factory):
            args = launch['preview_args']
            period = args[args.index('--trajectory-period') + 1]
            preview_fps = args[args.index('--preview-fps') + 1]
            thermal_preview_fps = args[args.index('--thermal-preview-fps') + 1]
            self.assertEqual(float(period), 1.)
            self.assertEqual(float(preview_fps), 15.)
            self.assertEqual(float(thermal_preview_fps), 2.)

    def test_dense_raw_trajectory_is_thinned_only_for_replay_and_preserves_exact_t(self):
        from scripts.episode_replay_source import select_source
        rows = [{'frame_id': i, 'simulation_time': i/10, 'position': [i/10, 0., 0.]}
                for i in range(26)]
        terminal = {'frame_id': None, 'simulation_time': 2.35, 'position': [2.35, 0., 0.]}
        for key in ('recorded_body_world_matrix', 'rgb_camera_world_matrix', 'thermal_camera_world_matrix'):
            terminal[key] = np.eye(4).tolist()
            terminal[key][3][:3] = terminal['position']
        metadata = {'sample_period_s': .1, 'trajectory_source': 'manual_driving',
                    'task_result': {'status': 'success', 'marked_by': 'keyboard:T',
                                    'simulation_time_s': 2.35, 'drone_position_m': [2.35, 0., 0.]},
                    'terminal_snapshot': terminal}
        selected = select_source(rows, metadata, 1.)
        self.assertEqual([row['simulation_time'] for row in selected], [0., 1., 2., 2.35])
        self.assertEqual(selected[-1]['action_type'], 'STOP')
        self.assertEqual(len(rows), 26)
        self.assertEqual(rows[1]['simulation_time'], .1)

    def test_new_offline_defaults_use_square_paired_framing(self):
        from scripts.replay_trajectory_highres import build_parser, resolve_capture_resolutions
        from scripts.replay_factory_inspection import RGB_SIZE, THERMAL_SIZE
        args = build_parser().parse_args(['--trajectory', 'raw.jsonl', '--out', 'out'])
        self.assertEqual(resolve_capture_resolutions(args), ((256, 256), (256, 256)))
        self.assertEqual((RGB_SIZE, THERMAL_SIZE), ((256, 256), (256, 256)))
        self.assertEqual(args.sample_interval, 1.)

    def test_one_second_mileage_requires_declared_cadence_and_rejects_missing_sample(self):
        from scripts.collection_mileage import summarize_mileage
        rows = [{'simulation_time': t, 'position': [t, 0, 0]} for t in [0., 1., 2.]]
        report = summarize_mileage(rows, {'sample_period_s': 1.})['raw_recording']
        self.assertEqual(report['measurement_status'], 'COMPLETE')
        self.assertEqual(report['length_3d_m'], 2.)
        self.assertEqual(summarize_mileage(rows, {})['raw_recording']['measurement_status'], 'INCOMPLETE')
        self.assertEqual(summarize_mileage(rows[::2], {'sample_period_s': 1.})['raw_recording']['issues'], {'sampling_gap': 1})

    def test_live_recorder_uses_same_cadence_as_saved_quality(self):
        from scripts.trajectory_recorder import TrajectoryRecorder
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {'sample_period_s': 1.}, 'run')
            for i in range(3):
                recorder.append({'frame_id': i, 'simulation_time': float(i), 'position': [i, 0, 0]})
            self.assertEqual(recorder.mileage.summary()['measurement_status'], 'COMPLETE')
            self.assertEqual(recorder.mileage.length_3d_m, 2.)
            recorder.close()

    def test_quality_uses_recorded_cadence_and_rejects_a_missing_one_hz_sample(self):
        from scripts.factory_bc_pilot import recording_quality as factory_quality
        from scripts.city_bc_pilot import recording_quality as city_quality
        rows = [{'frame_id': i, 'simulation_time': t,
                 **{key: np.eye(4).tolist() for key in (
                     'recorded_body_world_matrix', 'rgb_camera_world_matrix', 'thermal_camera_world_matrix')}}
                for i, t in enumerate([0., 1.016666667, 2.033333334])]
        for quality in (factory_quality, city_quality):
            self.assertEqual(quality(rows, sample_period_s=1.)['result'], 'PASS')
            self.assertEqual(quality(rows)['result'], 'FAIL')
            missing = [dict(rows[0]), dict(rows[2], frame_id=1)]
            self.assertEqual(quality(missing, sample_period_s=1.)['result'], 'FAIL')

    def test_bc_accepts_declared_one_second_samples_and_exact_off_grid_terminal(self):
        from scripts.bc_pose_targets import select_future_endpoint
        states = [{'simulation_time': t} for t in [0., 1., 2., 2.3]]
        states[-1].update(action_type='STOP', source_event_ref='run_metadata.json#terminal_snapshot')
        self.assertIs(select_future_endpoint(states, 0, sample_period_s=1.), states[1])
        self.assertIs(select_future_endpoint(states, 2, sample_period_s=1.), states[-1])
        with self.assertRaises(ValueError):
            select_future_endpoint(states, 0)
        with self.assertRaises(ValueError):
            select_future_endpoint(states[::2] + [states[-1]], 0, sample_period_s=1.)

    def test_rounded_thermal_height_scales_actual_rgb_intrinsics(self):
        from scripts.dual_camera_support import scale_camera_optics_for_resolution
        from tests.test_dual_camera_support import _CameraOptics, _Intrinsics
        optics = _CameraOptics(1., 1., 1., _Intrinsics(100., 100., 64., 51.5, 128, 103), 1)
        rgb = scale_camera_optics_for_resolution(optics, 256, 205)
        self.assertEqual((rgb.intrinsics.width, rgb.intrinsics.height), (256, 205))
        self.assertAlmostEqual(rgb.intrinsics.fx_px, 200.)
        self.assertAlmostEqual(rgb.intrinsics.fy_px, 100.*205/103)
        self.assertEqual((rgb.intrinsics.cx_px, rgb.intrinsics.cy_px), (128., 102.5))

    def test_factory_validation_uses_legacy_bundle_metadata_dimensions(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts import replay_factory_inspection as replay
        provenance = {'trajectory_source': 'manual_driving', 'task_result': {'status': 'unjudged'}}
        metadata = dict(replay.development_labels(provenance, manual_only=True),
                        trajectory_provenance=provenance, rgb_resolution=[512,410],
                        thermal_resolution=[256,205], observation_frame_ids=[], source_input_hashes={},
                        task={'environment_evidence': {'factory_inspection': {'inspection_requirements':
                             {'maximum_sample_gap_s': .25, 'hold_duration_s': 3.}}}})
        frame = SimpleNamespace(trajectory_frame_id=0, index=0, metadata_path='unused')
        bundle = SimpleNamespace(metadata=metadata, frames=[frame], trajectory_path=Path('unused'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'visual_evidence.jsonl').write_text('', encoding='utf-8')
            with patch('scripts.generate_replay_video.load_replay', return_value=bundle), \
                 patch('scripts.trajectory_recorder.read_trajectory', return_value=[{'frame_id':0, 'simulation_time':0.}]), \
                 patch.object(replay, 'manual_replay_records', side_effect=lambda rows, meta: rows), \
                 patch('scripts.frame_metadata.FrameMetadataReader.read', return_value={
                     'strict_sync_verified':False, 'strict_per_frame_pixel_sync_verified':False}), \
                 patch.object(replay, 'validate_capture'), \
                 patch.object(replay, 'validate_saved_frame') as saved:
                self.assertEqual(replay.validate_bundle(root)['result'], 'PASS')
                saved.assert_called_once_with(root, 0, rgb_size=(512,410), thermal_size=(256,205))

    def test_legacy_saved_frame_uses_explicit_stored_sizes(self):
        from scripts.replay_factory_inspection import validate_saved_frame
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = dict(rgb=np.zeros((410, 512, 4), dtype=np.uint8),
                          display8=np.zeros((205, 256), dtype=np.uint8),
                          radiance=np.ones((205, 256)), apparent_t=np.full((205, 256), 300.),
                          dn16=np.zeros((205, 256), dtype=np.uint16))
            for suffix, array in arrays.items():
                np.save(root / f'frame_000000_{suffix}.npy', array)
            Image.fromarray(arrays['rgb']).save(root / 'frame_000000_rgb.png')
            Image.fromarray(arrays['display8']).save(root / 'frame_000000_thermal.png')
            validate_saved_frame(root, 0, rgb_size=(512, 410), thermal_size=(256, 205))
            with self.assertRaisesRegex(ValueError, 'resolution'):
                validate_saved_frame(root, 0)


if __name__ == '__main__':
    unittest.main()
