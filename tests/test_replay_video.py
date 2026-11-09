"""Tests for preserving frame provenance and time in presentation videos."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess

from PIL import Image


class GeneratorAvailabilityTests(unittest.TestCase):
    def test_reusable_generator_is_available_without_isaac(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.generate_replay_video'),
                             'Reusable replay-video generator has not been implemented')


class ReplayVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.find_spec('scripts.generate_replay_video')
        if spec is None:
            raise unittest.SkipTest('Generator implementation is not present yet')
        from scripts import generate_replay_video
        cls.video = generate_replay_video

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.replay = self.root / 'replay'
        self.replay.mkdir()
        self.trajectory = self.root / 'trajectory.jsonl'
        records = [
            {'frame_id': 0, 'simulation_time': 5.0, 'position': [0, 0, 1]},
            {'frame_id': 2, 'simulation_time': 5.2, 'position': [1, 0, 1]},
            {'frame_id': 5, 'simulation_time': 5.55, 'position': [1, 2, 1]},
        ]
        self.trajectory.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
        metadata = {
            'result': 'PASS', 'frames': 3, 'resolution': [96, 64],
            'trajectory': str(self.trajectory), 'sample_interval_s': 0.2,
            'strict_per_frame_pixel_sync_verified': False,
            'data_usage': 'pilot/development data',
        }
        (self.replay / 'replay_metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        for index, r in enumerate(records):
            meta = dict(r, frame_id=index, trajectory_frame_id=r['frame_id'],
                        trajectory_index=index, replay_simulation_time=r['simulation_time'],
                        replay_source=str(self.trajectory), resolution=[96, 64],
                        strict_sync_verified=False)
            (self.replay / f'frame_{index:06d}.json').write_text(json.dumps(meta), encoding='utf-8')
            Image.new('RGB', (96, 64), (30 + index * 60, 80, 100)).save(
                self.replay / f'frame_{index:06d}_rgb.png')
            Image.new('L', (96, 64), 50 + index * 70).save(
                self.replay / f'frame_{index:06d}_thermal.png')

    def change_meta(self, index, **updates):
        path = self.replay / f'frame_{index:06d}.json'
        data = json.loads(path.read_text())
        data.update(updates)
        path.write_text(json.dumps(data), encoding='utf-8')

    def compact_frames(self):
        from scripts.frame_metadata import FrameMetadataWriter
        from scripts.trajectory_recorder import trajectory_provenance
        metadata_path = self.replay / 'replay_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        provenance = trajectory_provenance(self.trajectory)
        metadata['trajectory_provenance'] = provenance
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        frames = []
        for index in range(3):
            path = self.replay / f'frame_{index:06d}.json'
            frame = json.loads(path.read_text(encoding='utf-8'))
            frame.update(task_id=None, task_result_status='unjudged',
                         trajectory_sha256=provenance['trajectory_sha256'])
            path.unlink()
            frames.append(frame)
        writer = FrameMetadataWriter(self.replay, frames[0])
        for index, frame in enumerate(frames):
            writer.write(index, frame)
        return self.replay / 'common_metadata.json'

    def prepare_validatable_replay(self):
        import hashlib
        import numpy as np
        stage = self.root / 'source.usd'
        stage.write_text('#usda 1.0\n', encoding='utf-8')
        metadata_path = self.replay / 'replay_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(source_stage_sha256=hashlib.sha256(stage.read_bytes()).hexdigest(),
                        scene_config_path=str(self.root / 'configs' / 'scene.yaml'))
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        for index in range(3):
            matrix = np.eye(4)
            matrix[3, 0] = index
            self.change_meta(index, stage_path=str(stage), camera_world_matrix=matrix.tolist())
            for kind in ('rgb', 'display8', 'dn16', 'radiance', 'apparent_t'):
                dtype = np.uint16 if kind == 'dn16' else np.float32
                np.save(self.replay / f'frame_{index:06d}_{kind}.npy', np.ones((64, 96), dtype=dtype))
        self.compact_frames()
        return stage

    def test_compact_frames_preserve_provenance_and_trajectory_alignment(self):
        self.compact_frames()
        bundle = self.video.load_replay(self.replay)
        self.assertEqual([f.trajectory_frame_id for f in bundle.frames], [0, 2, 5])
        self.assertEqual([f.video_time_s for f in bundle.frames], [0.0, 0.2, 0.55])
        self.assertFalse(bundle.strict_sync_verified)

    def test_manual_preview_video_has_visible_identity_without_altering_sensor_panels(self):
        bundle=self.video.load_replay(self.replay)
        original,panels=self.video.compose_frame(bundle,0)
        bundle.metadata['confirmation_mode']='manual_only'
        preview,_=self.video.compose_frame(bundle,0)
        self.assertTrue(original.crop((0,0,1000,96)).tobytes()!=preview.crop((0,0,1000,96)).tobytes())
        for rectangle in panels.values():
            self.assertEqual(original.crop(rectangle).tobytes(),preview.crop(rectangle).tobytes())

    def test_video_tracks_shared_metadata_in_source_hashes(self):
        common_path = self.compact_frames()
        report = self.video.generate_video(self.replay, self.root / 'compact-video')
        source_inputs = {Path(path).resolve().as_posix().casefold(): digest
                         for path, digest in report['source_input_hashes'].items()}
        self.assertIn(common_path.resolve().as_posix().casefold(), source_inputs)
        self.assertEqual(report['encoded_frame_count'], 3)
        self.assertTrue(report['source_files_unchanged'])

    def test_video_inherits_pipeline_validation_without_claiming_new_pixel_checks(self):
        replay_metadata_path = self.replay / 'replay_metadata.json'
        metadata = json.loads(replay_metadata_path.read_text(encoding='utf-8'))
        from scripts.sync_validation_metadata import _load_verified_evidence, SYNC_CONTRACT_CHECKS
        evidence = _load_verified_evidence()
        checks = dict.fromkeys(SYNC_CONTRACT_CHECKS, True)
        metadata.update(
            strict_sync_verified=True,
            strict_per_frame_pixel_sync_verified=False,
            sync_status='synchronized_pipeline_validated',
            sync_validation_status='validated',
            sync_validation_scope='pipeline_and_frame_capture_contract',
            sync_validation_evidence=evidence,
        )
        replay_metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        for index in range(3):
            frame_path = self.replay / f'frame_{index:06d}.json'
            frame = json.loads(frame_path.read_text(encoding='utf-8'))
            frame.update(
                strict_sync_verified=True,
                strict_per_frame_pixel_sync_verified=False,
                sync_status='synchronized_pipeline_validated',
                sync_validation_status='validated',
                sync_validation_scope='pipeline_and_frame_capture_contract',
                sync_validation_evidence=evidence,
                sync_validation_checks=checks,
            )
            frame_path.write_text(json.dumps(frame), encoding='utf-8')

        output = self.root / 'validated-video'
        report = self.video.generate_video(self.replay, output)

        self.assertTrue(report['strict_sync_verified'])
        self.assertFalse(report['strict_per_frame_pixel_sync_verified'])
        self.assertEqual(report['sync_validation_status'], 'validated')
        self.assertEqual(report['sync_validation_scope'], 'pipeline_and_frame_capture_contract')
        self.assertEqual(report['sync_validation_evidence'], evidence)
        row = json.loads((output / 'video_frames.jsonl').read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(row['sync_status'], 'synchronized_pipeline_validated')
        self.assertFalse(row['strict_per_frame_pixel_sync_verified'])

    def test_output_validator_reads_compact_stage_path_and_all_arrays(self):
        from scripts.validate_replay_outputs import validate_replay
        stage = self.prepare_validatable_replay()
        report = validate_replay(self.replay)
        self.assertEqual(report['result'], 'PASS_FILE_AND_ARRAY_CHECKS')
        self.assertEqual(report['frames_validated'], 3)
        self.assertTrue(report['source_usd_unchanged'])
        self.assertFalse(report['strict_per_frame_pixel_sync_verified'])

    def test_output_validator_preserves_pipeline_sync_validation_scope(self):
        self.prepare_validatable_replay()
        metadata_path = self.replay / 'replay_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        from scripts.sync_validation_metadata import _load_verified_evidence, SYNC_CONTRACT_CHECKS
        evidence = _load_verified_evidence()
        checks = dict.fromkeys(SYNC_CONTRACT_CHECKS, True)
        metadata.update(
            strict_sync_verified=True,
            strict_per_frame_pixel_sync_verified=False,
            sync_status='synchronized_pipeline_validated',
            sync_validation_status='validated',
            sync_validation_scope='pipeline_and_frame_capture_contract',
            sync_validation_evidence=evidence,
        )
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

        for frame_path in sorted(self.replay.glob('frame_*.json')):
            frame = json.loads(frame_path.read_text(encoding='utf-8'))
            frame.update(strict_sync_verified=True,
                         sync_status='synchronized_pipeline_validated',
                         sync_validation_status='validated',
                         sync_validation_evidence=evidence,
                         sync_validation_checks=checks)
            frame_path.write_text(json.dumps(frame), encoding='utf-8')

        from scripts.validate_replay_outputs import validate_replay
        report = validate_replay(self.replay)

        self.assertTrue(report['strict_sync_verified'])
        self.assertFalse(report['strict_per_frame_pixel_sync_verified'])
        self.assertEqual(report['sync_status'], 'synchronized_pipeline_validated')
        self.assertEqual(report['sync_validation_status'], 'validated')
        self.assertEqual(report['sync_validation_scope'], 'pipeline_and_frame_capture_contract')

    def test_unverified_frame_or_failed_files_cannot_produce_verified_summary(self):
        self.prepare_validatable_replay()
        metadata_path = self.replay / 'replay_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        from scripts.sync_validation_metadata import _load_verified_evidence, SYNC_CONTRACT_CHECKS
        evidence = _load_verified_evidence()
        checks = dict.fromkeys(SYNC_CONTRACT_CHECKS, True)
        metadata.update(strict_sync_verified=True,
                        sync_validation_status='validated',
                        sync_validation_evidence=evidence,
                        sync_status='synchronized_pipeline_validated')
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        for path in self.replay.glob('frame_*.json'):
            frame = json.loads(path.read_text(encoding='utf-8'))
            frame.update(strict_sync_verified=True,
                         sync_validation_status='validated',
                         sync_status='synchronized_pipeline_validated',
                         sync_validation_evidence=evidence,
                         sync_validation_checks=checks)
            path.write_text(json.dumps(frame), encoding='utf-8')
        frame_path = self.replay / 'frame_000001.json'
        frame = json.loads(frame_path.read_text(encoding='utf-8'))
        frame['strict_sync_verified'] = False
        frame_path.write_text(json.dumps(frame), encoding='utf-8')
        from scripts.validate_replay_outputs import validate_replay
        report = validate_replay(self.replay)
        self.assertFalse(report['strict_sync_verified'])

        frame['strict_sync_verified'] = True
        frame_path.write_text(json.dumps(frame), encoding='utf-8')
        (self.replay / 'frame_000001_dn16.npy').unlink()
        report = validate_replay(self.replay)
        self.assertEqual(report['result'], 'FAIL')
        self.assertFalse(report['strict_sync_verified'])

    def test_output_validator_reports_missing_common_metadata_without_crashing(self):
        from scripts.validate_replay_outputs import validate_replay
        self.prepare_validatable_replay()
        (self.replay / 'common_metadata.json').unlink()

        report = validate_replay(self.replay)

        self.assertEqual(report['result'], 'FAIL')
        self.assertTrue(any('common metadata' in failure.lower() or 'common_metadata' in failure
                            for failure in report['failures']))
        self.assertTrue((self.replay / 'validation_summary.json').is_file())

    def test_uses_original_trajectory_ids_not_sample_indices_and_actual_time_gaps(self):
        bundle = self.video.load_replay(self.replay)
        self.assertEqual([f.trajectory_frame_id for f in bundle.frames], [0, 2, 5])
        self.assertEqual([f.video_time_s for f in bundle.frames], [0.0, 0.2, 0.55])
        self.assertEqual([round(f.duration_s, 9) for f in bundle.frames], [0.2, 0.35, 0.2])
        self.assertFalse(bundle.strict_sync_verified)

    def test_refuses_incorrect_trajectory_timestamp_or_position(self):
        self.change_meta(1, replay_simulation_time=5.3)
        with self.assertRaisesRegex(ValueError, 'trajectory.*time'):
            self.video.load_replay(self.replay)
        self.change_meta(1, replay_simulation_time=5.2, position=[99, 0, 1])
        with self.assertRaisesRegex(ValueError, 'trajectory.*position'):
            self.video.load_replay(self.replay)

    def test_legacy_pose_recovery_keeps_replay_pose_and_source_pose_distinct(self):
        self.change_meta(1, position=[1.001, 0, 1],
                         source_trajectory_pose={'position': [1, 0, 1]})

        bundle = self.video.load_replay(self.replay)

        self.assertEqual(bundle.frames[1].position, [1.001, 0, 1])

    def test_refuses_missing_pair_and_wrong_dimensions(self):
        path = self.replay / 'frame_000001_thermal.png'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.video.load_replay(self.replay)
        Image.new('L', (95, 64)).save(path)
        with self.assertRaisesRegex(ValueError, 'resolution'):
            self.video.load_replay(self.replay)

    def test_refuses_duplicate_times_and_extra_frame_files(self):
        self.change_meta(1, replay_simulation_time=5.0)
        with self.assertRaises(ValueError):
            self.video.load_replay(self.replay)
        self.change_meta(1, replay_simulation_time=5.2)
        (self.replay / 'frame_000004.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'count'):
            self.video.load_replay(self.replay)

    def test_missing_sync_evidence_never_inherits_a_pass_label(self):
        path = self.replay / 'replay_metadata.json'
        m = json.loads(path.read_text())
        m['strict_per_frame_pixel_sync_verified'] = True
        m['data_usage'] = 'benchmark-ready'
        path.write_text(json.dumps(m))
        self.assertFalse(self.video.load_replay(self.replay).strict_sync_verified)

    def test_rejects_failed_replay_nonfinite_time_and_mismatched_orientation(self):
        path = self.replay / 'replay_metadata.json'
        m = json.loads(path.read_text())
        m['result'] = 'FAIL'
        path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'successful'):
            self.video.load_replay(self.replay)
        m['result'] = 'PASS'
        path.write_text(json.dumps(m))
        self.change_meta(0, replay_simulation_time=float('nan'))
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.video.load_replay(self.replay)
        self.change_meta(0, replay_simulation_time=5.0, orientation=[0, 1, 0, 0])
        records = [json.loads(line) for line in self.trajectory.read_text().splitlines()]
        records[0]['orientation'] = [1, 0, 0, 0]
        self.trajectory.write_text('\n'.join(json.dumps(r) for r in records))
        with self.assertRaisesRegex(ValueError, 'orientation'):
            self.video.load_replay(self.replay)

    def test_trajectory_index_means_original_row_not_sampled_index(self):
        self.change_meta(1, trajectory_index=99)
        bundle = self.video.load_replay(self.replay)
        self.assertEqual(bundle.frames[1].trajectory_index, 1)
        self.assertEqual(bundle.frames[1].replay_trajectory_index, 99)

    def test_refuses_extra_png_pair_not_accounted_for_in_metadata(self):
        Image.new('RGB', (96, 64)).save(self.replay / 'frame_000099_rgb.png')
        with self.assertRaisesRegex(ValueError, 'count'):
            self.video.load_replay(self.replay)

    def test_composition_preserves_native_source_pixels(self):
        bundle = self.video.load_replay(self.replay)
        image, panels = self.video.compose_frame(bundle, 1)
        with Image.open(self.replay / 'frame_000001_rgb.png') as source:
            self.assertEqual(image.crop(panels['rgb']).tobytes(), source.convert('RGB').tobytes())
        with Image.open(self.replay / 'frame_000001_thermal.png') as source:
            self.assertEqual(image.crop(panels['thermal']).tobytes(), source.convert('RGB').tobytes())

    def test_independent_stream_resolutions_are_validated_and_thermal_is_resized_for_display_only(self):
        metadata_path = self.replay / 'replay_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['rgb_resolution'] = [96, 64]
        metadata['thermal_resolution'] = [48, 32]
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        for index in range(3):
            frame_path = self.replay / f'frame_{index:06d}.json'
            frame_metadata = json.loads(frame_path.read_text())
            frame_metadata['rgb_resolution'] = [96, 64]
            frame_metadata['thermal_resolution'] = [48, 32]
            frame_path.write_text(json.dumps(frame_metadata), encoding='utf-8')
            Image.new('L', (48, 32), 50 + index * 70).save(
                self.replay / f'frame_{index:06d}_thermal.png')

        bundle = self.video.load_replay(self.replay)
        self.assertEqual((bundle.rgb_width, bundle.rgb_height), (96, 64))
        self.assertEqual((bundle.thermal_width, bundle.thermal_height), (48, 32))
        image, panels = self.video.compose_frame(bundle, 1)
        self.assertEqual(image.crop(panels['rgb']).size, (96, 64))
        self.assertEqual(image.crop(panels['thermal']).size, (96, 64))
        with Image.open(self.replay / 'frame_000001_thermal.png') as source:
            expected_display = source.convert('RGB').resize((96, 64), Image.Resampling.NEAREST)
            self.assertEqual(image.crop(panels['thermal']).tobytes(), expected_display.tobytes())
            self.assertEqual(source.size, (48, 32))

    def test_real_mp4_has_one_encoded_frame_per_pair_and_matching_timestamps(self):
        ffmpeg = self.video.find_ffmpeg(None)
        out = self.root / 'video'
        report = self.video.generate_video(self.replay, out, ffmpeg=ffmpeg)
        self.assertEqual(report['encoded_frame_count'], 3)
        self.assertTrue(report['source_files_unchanged'])
        self.assertTrue((out / 'paired_replay.mp4').is_file())
        rows = [json.loads(line) for line in (out / 'video_frames.jsonl').read_text().splitlines()]
        self.assertEqual([r['trajectory_frame_id'] for r in rows], [0, 2, 5])
        self.assertEqual([round(r['video_time_s'], 9) for r in rows], [0.0, 0.2, 0.55])
        self.assertEqual(report['encoded_pts_s'], [0.0, 0.2, 0.55])
        self.assertFalse(report['strict_per_frame_pixel_sync_verified'])
        probe = subprocess.run([str(ffmpeg.with_name('ffprobe.exe')), '-v', 'error',
                                '-show_entries', 'stream=duration', '-of', 'json',
                                str(out / 'paired_replay.mp4')],
                               capture_output=True, text=True, check=True)
        duration = float(json.loads(probe.stdout)['streams'][0]['duration'])
        self.assertAlmostEqual(duration, 0.75, places=5)
        self.assertAlmostEqual(report['video_duration_s'], duration, places=5)
        with self.assertRaises(FileExistsError):
            self.video.generate_video(self.replay, out, ffmpeg=ffmpeg)

    def test_video_keeps_the_recorded_instruction_and_unjudged_result(self):
        run_path = self.trajectory.parent / 'run_metadata.json'
        run_path.write_text(json.dumps({
            'task': {'task_id': 'spot-search', 'instruction': 'the recorded instruction'},
            'success': None, 'task_result': {'status': 'unjudged'},
        }), encoding='utf-8')
        report = self.video.generate_video(self.replay, self.root / 'with-task')
        self.assertEqual(report.get('task', {}).get('instruction'), 'the recorded instruction')
        self.assertEqual(report.get('task_result', {}).get('status'), 'unjudged')
        self.assertIsNone(report['task_success'])

    def test_changed_run_labels_after_replay_are_rejected(self):
        from scripts.trajectory_recorder import trajectory_provenance
        run_path = self.trajectory.parent / 'run_metadata.json'
        run_path.write_text(json.dumps({'task': {'task_id': 'original'}, 'success': None}), encoding='utf-8')
        meta_path = self.replay / 'replay_metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['trajectory_provenance'] = trajectory_provenance(self.trajectory)
        meta_path.write_text(json.dumps(meta), encoding='utf-8')
        run_path.write_text(json.dumps({'task': {'task_id': 'changed'}, 'success': True}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.video.load_replay(self.replay)

    def test_video_rejects_a_replay_from_a_different_recorded_scene_version(self):
        run_path = self.trajectory.parent / 'run_metadata.json'
        run_path.write_text(json.dumps({'stage_sha256': '1' * 64}), encoding='utf-8')
        meta_path = self.replay / 'replay_metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['source_stage_sha256'] = '2' * 64
        meta_path.write_text(json.dumps(meta), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'recorded.*USD'):
            self.video.load_replay(self.replay)

    def test_older_replay_provenance_remains_valid_when_optional_hash_fields_are_added(self):
        from scripts.trajectory_recorder import trajectory_provenance
        meta_path = self.replay / 'replay_metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        legacy = trajectory_provenance(self.trajectory)
        legacy.pop('recording_sensor_config_sha256', None)
        legacy.pop('recording_thermal_config_sha256', None)
        meta['trajectory_provenance'] = legacy
        meta_path.write_text(json.dumps(meta), encoding='utf-8')
        for index in range(3):
            self.change_meta(index, task_id=None, task_result_status='unjudged',
                             trajectory_sha256=legacy['trajectory_sha256'])
        bundle = self.video.load_replay(self.replay)
        self.assertEqual(len(bundle.frames), 3)
        self.assertIsNone(bundle.metadata['trajectory_provenance']['recording_sensor_config_sha256'])


if __name__ == '__main__':
    unittest.main()
