import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.trajectory_recorder import TrajectoryRecorder, read_trajectory
from scripts import trajectory_recorder


class TrajectoryRecorderTests(unittest.TestCase):
    def test_recorded_replay_pose_comes_from_same_usd_sample_as_camera(self):
        yaw_target = np.deg2rad(0.025)
        controller_state = {
            'position': [10.0, 20.0, 30.0],
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': yaw_target,
            'orientation': [np.cos(yaw_target / 2), 0.0, 0.0, np.sin(yaw_target / 2)],
            'command': {'keys': ['Q']},
        }
        usd_body_world = np.eye(4)
        usd_body_world[:3, :3] *= 3.0
        usd_body_world[3, :3] = [10.0, 20.0, 30.0]

        sample = trajectory_recorder.trajectory_state_from_world_transform(
            controller_state, usd_body_world)

        self.assertEqual(sample['position'], [10.0, 20.0, 30.0])
        self.assertEqual(sample['orientation'], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(sample['yaw'], 0.0)
        self.assertEqual(sample['controller_target_pose']['yaw'], yaw_target)
        self.assertEqual(sample['recorded_body_world_matrix'], usd_body_world.tolist())
        self.assertEqual(sample['trajectory_pose_source'], 'usd_body_world_transform')
        self.assertEqual(sample['command'], {'keys': ['Q']})

    def test_world_pose_extraction_removes_drone_scale_without_losing_yaw(self):
        yaw = np.deg2rad(0.025)
        usd_body_world = np.eye(4)
        usd_body_world[:3, :3] = 3.0 * np.array([
            [np.cos(yaw), np.sin(yaw), 0.0],
            [-np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])

        sample = trajectory_recorder.trajectory_state_from_world_transform(
            {'position': [0.0, 0.0, 0.0], 'roll': 0.0, 'pitch': 0.0,
             'yaw': 0.0, 'orientation': [1.0, 0.0, 0.0, 0.0]}, usd_body_world)

        self.assertAlmostEqual(sample['orientation'][0], np.cos(yaw / 2), places=12)
        self.assertAlmostEqual(sample['orientation'][3], np.sin(yaw / 2), places=12)
        self.assertAlmostEqual(sample['yaw'], yaw, places=12)

    def test_writes_jsonl_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {'task': {'task_id': 'demo'}}, run_id='run1')
            recorder.append({'frame_id': 0, 'simulation_time': 0.0, 'position': (1, 2, 3)})
            recorder.append({'frame_id': 1, 'simulation_time': 0.2, 'position': (2, 2, 3)})
            trajectory = recorder.close()
            self.assertEqual(read_trajectory(trajectory)[1]['position'], [2, 2, 3])
            metadata = json.loads((Path(directory) / 'run1' / 'run_metadata.json').read_text())
            self.assertEqual(metadata['record_count'], 2)

    def test_recording_without_task_does_not_invent_a_task_result(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {}, run_id='no-task')
            recorder.close()
            metadata = json.loads((Path(directory) / 'no-task' / 'run_metadata.json').read_text())
        self.assertNotIn('task_result', metadata)

    def test_unmarked_task_is_explicitly_unjudged_when_recording_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(
                directory, {'task': {'task_id': 'school_find_spot_01',
                                     'instruction': '寻找 Spot'}}, run_id='unjudged')
            recorder.append({'frame_id': 0, 'simulation_time': 0.0})
            recorder.close()

            metadata = json.loads((Path(directory) / 'unjudged' / 'run_metadata.json').read_text())
        self.assertEqual(metadata['task_result']['status'], 'unjudged')
        self.assertIsNone(metadata['success'])
        self.assertNotIn('success', metadata['task'])

    def test_rejects_duplicate_or_missing_frame_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {}, run_id='run2')
            with self.assertRaises(ValueError):
                recorder.append({'simulation_time': 0.0})
            recorder.append({'frame_id': 0, 'simulation_time': 0.0})
            with self.assertRaises(ValueError):
                recorder.append({'frame_id': 0, 'simulation_time': 0.1})

    def test_mark_success_records_time_position_and_nearest_trajectory_frame_once(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {'task': {'task_id': 'spot-search'}}, run_id='run3')
            recorder.append({'frame_id': 4, 'simulation_time': 1.0, 'position': (1, 2, 3)})
            recorder.append({'frame_id': 5, 'simulation_time': 1.2, 'position': (2, 2, 3)})

            self.assertTrue(recorder.mark_success(1.25, (2.5, 2.0, 3.0)))
            self.assertFalse(recorder.mark_success(1.30, (3.0, 2.0, 3.0)))

            path = recorder.close()
            metadata = json.loads((Path(directory) / 'run3' / 'run_metadata.json').read_text())
            self.assertTrue(metadata['success'])
            self.assertEqual(metadata['success_time'], 1.25)
            self.assertEqual(metadata['success_position'], [2.5, 2.0, 3.0])
            self.assertEqual(metadata['success_trajectory_frame_id'], 5)
            self.assertEqual(metadata['task_result']['marked_by'], 'keyboard:T')
            self.assertEqual(metadata['task_result']['status'], 'success')
            self.assertNotIn('success', metadata['task'], 'Run result must not mutate task definition')

    def test_recorder_takes_an_independent_snapshot_of_task_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            task = {'task_id': 'school_find_spot_01', 'instruction': 'find Spot',
                    'environment_landmarks': ['trees']}
            recorder = TrajectoryRecorder.create(directory, {'task': task}, run_id='snapshot')
            task['instruction'] = 'changed after recording started'
            task['environment_landmarks'].append('changed')
            recorder.close()

            metadata = json.loads((Path(directory) / 'snapshot' / 'run_metadata.json').read_text())
        self.assertEqual(metadata['task']['instruction'], 'find Spot')
        self.assertEqual(metadata['task']['environment_landmarks'], ['trees'])

    def test_success_keeps_duration_separate_from_absolute_simulation_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {'task': {'task_id': 'spot'}}, run_id='elapsed')
            recorder.mark_success(14.25, (1, 2, 3), elapsed_time=4.25)
            recorder.close()
            saved = json.loads((recorder.run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['success_time'], 4.25)
        self.assertEqual(saved['success_simulation_time_s'], 14.25)
        self.assertEqual(saved['task_result']['simulation_time_s'], 14.25)

    def test_stop_flush_preserves_unjudged_and_later_success_can_replace_it(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {'task': {'task_id': 'spot'}}, run_id='stop')
            recorder.append({'frame_id': 0, 'simulation_time': 10.0})
            recorder.flush()
            saved = json.loads((recorder.run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['task_result']['status'], 'unjudged')
            self.assertTrue(recorder.mark_success(10.1, (1, 2, 3)))
            recorder.close()
            saved = json.loads((recorder.run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['task_result']['status'], 'success')

    def test_closed_recording_cannot_be_marked_success_after_timeline_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {'task': {'task_id': 'spot'}}, run_id='stop')
            recorder.append({'frame_id': 0, 'simulation_time': 20.0})
            recorder.close()
            self.assertFalse(recorder.mark_success(5.0, (1, 2, 3)))
            recorder.close()
            saved = json.loads((recorder.run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['task_result']['status'], 'unjudged')

    def test_closed_recording_rejects_new_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {}, run_id='closed')
            recorder.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                recorder.append({'frame_id': 0, 'simulation_time': 0.0})

    def test_provenance_distinguishes_recording_sensor_and_thermal_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {
                'sensor_config_sha256': 'recorded-sensor-hash',
                'thermal_config_sha256': 'recorded-thermal-hash'}, run_id='configs')
            recorder.append({'frame_id': 0, 'simulation_time': 0.0})
            provenance = trajectory_recorder.trajectory_provenance(recorder.close())
        self.assertEqual(provenance.get('recording_sensor_config_sha256'), 'recorded-sensor-hash')
        self.assertEqual(provenance.get('recording_thermal_config_sha256'), 'recorded-thermal-hash')

    def test_replay_provenance_preserves_the_recorded_task_and_unjudged_label(self):
        loader = getattr(trajectory_recorder, 'trajectory_provenance', None)
        self.assertTrue(callable(loader), 'Replay has no frozen task provenance')
        with tempfile.TemporaryDirectory() as directory:
            recorder = TrajectoryRecorder.create(directory, {
                'task': {'task_id': 'old-task', 'instruction': 'original instruction'},
                'success': None}, run_id='original')
            recorder.append({'frame_id': 0, 'simulation_time': 10.0})
            path = recorder.close()
            provenance = loader(path)
        self.assertEqual(provenance['task']['instruction'], 'original instruction')
        self.assertEqual(provenance['task_result']['status'], 'unjudged')
        self.assertEqual(provenance['source_run_id'], 'original')
        self.assertEqual(len(provenance['trajectory_sha256']), 64)
        self.assertEqual(len(provenance['source_run_metadata_sha256']), 64)

    def test_missing_legacy_task_is_never_filled_from_current_scene_config(self):
        loader = getattr(trajectory_recorder, 'trajectory_provenance', None)
        self.assertTrue(callable(loader), 'Replay has no frozen task provenance')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trajectory.jsonl'
            path.write_text('{"frame_id":0,"simulation_time":0.0}\n', encoding='utf-8')
            provenance = loader(path)
        self.assertIsNone(provenance['task'])
        self.assertEqual(provenance['task_result']['status'], 'unjudged')
        self.assertEqual(provenance['task_metadata_status'], 'missing')


if __name__ == '__main__':
    unittest.main()
