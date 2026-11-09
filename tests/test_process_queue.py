import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import process_queue as queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.addCleanup(patch.stopall)
        patch.object(queue, 'ROOT', self.root).start()
        self.path = self.root / 'queues/test.json'
        self.calls = []

    def check(self, run, interval):
        return {'source_hashes': {str(run): 'original'}, 'selected_frames': 3}

    def verify(self, output, run_id):
        return {'status': 'valid'}

    def runner(self, run, output, log, interval):
        self.calls.append((run, output, log, interval))
        return 0

    def create(self, runs=('first', 'second')):
        return queue.create_queue(runs, self.path, checker=self.check)

    def execute(self, **kwargs):
        return queue.execute(self.path, checker=self.check, runner=self.runner,
                             busy=lambda: [], verify=self.verify, **kwargs)

    def test_serial_success_resume_skips_complete_and_keeps_interval(self):
        self.create()
        self.assertEqual(self.execute(), 0)
        self.assertEqual([call[0].name for call in self.calls], ['first', 'second'])
        self.assertTrue(all(call[3] == 1.0 for call in self.calls))
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.calls), 2)

    def test_failure_continues_and_resume_retries_with_new_output_and_log(self):
        self.create()
        def fail_first(run, output, log, interval):
            self.runner(run, output, log, interval)
            return int(run.name == 'first')
        self.assertEqual(queue.execute(self.path, checker=self.check, runner=fail_first,
                                      busy=lambda: [], verify=self.verify), 1)
        state = queue.read_json(self.path)
        self.assertEqual([r['status'] for r in state['items']], ['FAILED', 'COMPLETE'])
        first_output, first_log = self.calls[0][1:3]
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.calls), 3)
        self.assertNotEqual(first_output, self.calls[2][1])
        self.assertNotEqual(first_log, self.calls[2][2])
        self.assertEqual(len(queue.read_json(self.path)['items'][0]['attempts']), 2)

    def test_zero_exit_without_completion_is_failed(self):
        self.create(('first',))
        def missing(*args):
            raise FileNotFoundError('final metadata missing')
        self.assertEqual(queue.execute(self.path, checker=self.check, runner=self.runner,
                                      busy=lambda: [], verify=missing), 1)
        self.assertEqual(queue.read_json(self.path)['items'][0]['status'], 'FAILED')

    def test_busy_worker_blocks_without_launching(self):
        self.create()
        result = queue.execute(self.path, checker=self.check, runner=self.runner,
                               busy=lambda: [{'ProcessId': 123}], verify=self.verify)
        self.assertEqual(result, 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(queue.read_json(self.path)['status'], 'BLOCKED')

    def test_source_change_refused_before_launch(self):
        self.create(('first',))
        self.assertEqual(queue.execute(self.path, checker=lambda *args: {'source_hashes': 'changed'},
                                      runner=self.runner, busy=lambda: [], verify=self.verify), 1)
        self.assertEqual(self.calls, [])

    def test_invalid_item_does_not_stop_valid_item(self):
        def check(run, interval):
            if run.name == 'first':
                raise ValueError('missing T')
            return self.check(run, interval)
        queue.create_queue(('first', 'second'), self.path, checker=check)
        self.assertEqual(queue.execute(self.path, checker=check, runner=self.runner,
                                      busy=lambda: [], verify=self.verify), 1)
        self.assertEqual([r[0].name for r in self.calls], ['second'])

    def test_os_lock_prevents_second_worker(self):
        with queue.exclusive(self.root / 'worker.lock'):
            with self.assertRaisesRegex(RuntimeError, 'Another queue'):
                with queue.exclusive(self.root / 'worker.lock'):
                    self.fail('second worker acquired lock')

    def test_process_detection_sees_mock_renderer_without_starting_isaac(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)',
                                    'start_crazyflie_dual_camera_view.py'])
        try:
            self.assertIn(process.pid, [row['ProcessId'] for row in queue.busy_processes()])
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_interrupted_attempt_retained_before_retry(self):
        state = self.create(('first',))
        state['items'][0].update(status='RUNNING', attempts=[{'status': 'RUNNING', 'output': 'old'}])
        queue.save(self.path, state)
        self.assertEqual(self.execute(), 0)
        attempts = queue.read_json(self.path)['items'][0]['attempts']
        self.assertEqual(attempts[0]['status'], 'INTERRUPTED')
        self.assertEqual(len(attempts), 2)

    def test_changed_completed_artifact_not_silently_skipped(self):
        self.create(('first',)); self.execute()
        self.assertEqual(queue.execute(self.path, checker=self.check, runner=self.runner,
                                      busy=lambda: [], verify=lambda *args: {'status': 'modified'}), 1)
        self.assertEqual(len(self.calls), 1)

    def test_duplicate_run_and_existing_queue_refused(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.create(('first', 'first'))
        self.create()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.create()

    def test_nonpositive_or_nonfinite_interval_refused(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                queue.create_queue(('first',), self.path, value, checker=self.check)

    def test_completion_requires_correct_run_pass_and_index(self):
        output = self.root / 'out'
        (output / 'replay').mkdir(parents=True)
        (output / 'dataset').mkdir()
        queue.save(output / 'pipeline_status.json', {'status': 'COMPLETE', 'run_id': 'first'})
        queue.save(output / 'replay/replay_metadata.json', {'result': 'PASS'})
        queue.save(output / 'dataset/episode.json', {'run_id': 'first'})
        self.assertEqual(len(queue.completion(output, 'first')), 3)
        with self.assertRaisesRegex(ValueError, 'RunId|RunId|this RunId'):
            queue.completion(output, 'second')
        queue.save(output / 'replay/replay_metadata.json', {'result': 'FAIL'})
        with self.assertRaisesRegex(ValueError, 'not PASS'):
            queue.completion(output, 'first')

    def test_real_powershell_child_invocation_and_completion_without_gpu(self):
        # Exercise actual subprocess/PowerShell argument and logging boundaries.
        scripts = self.root / 'scripts'; scripts.mkdir()
        (scripts / 'process.ps1').write_text('''param([string]$Run,[string]$Out,[double]$SampleInterval)
$ErrorActionPreference='Stop'
if ($SampleInterval -ne 1.0) { exit 9 }
$id=Split-Path -Leaf $Run
New-Item -ItemType Directory -Path "$Out/replay","$Out/dataset" -Force | Out-Null
@{status='COMPLETE';run_id=$id} | ConvertTo-Json | Set-Content "$Out/pipeline_status.json" -Encoding UTF8
@{result='PASS';frames=3} | ConvertTo-Json | Set-Content "$Out/replay/replay_metadata.json" -Encoding UTF8
@{run_id=$id} | ConvertTo-Json | Set-Content "$Out/dataset/episode.json" -Encoding UTF8
Write-Host 'mock replay complete; no Isaac launched'
exit 0
''', encoding='utf-8')
        self.create()
        self.assertEqual(queue.execute(self.path, checker=self.check, busy=lambda: []), 0)
        state = queue.read_json(self.path)
        self.assertEqual([i['attempts'][0]['frames'] for i in state['items']], [3, 3])
        self.assertIn('mock replay complete', Path(state['items'][0]['attempts'][0]['log']).read_text())
        self.assertEqual(queue.execute(self.path, checker=self.check, busy=lambda: []), 0)

    def test_preflight_checks_terminal_and_raw_time_without_writing_sources(self):
        run = self.root / 'first'; run.mkdir()
        config = self.root / 'scene.yaml'; config.write_text('scene: test')
        stage = self.root / 'scene.usda'; stage.write_text('#usda 1.0')
        matrix = [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
        def row(t, frame):
            return {'frame_id': frame, 'simulation_time': t, 'position': [0.,0.,0.],
                    **{k: copy.deepcopy(matrix) for k in ('recorded_body_world_matrix',
                       'rgb_camera_world_matrix', 'thermal_camera_world_matrix')}}
        rows = [row(0., 0), row(.1, 1)]
        trajectory = run / 'trajectory.jsonl'
        trajectory.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        metadata = {'run_id': 'first', 'scene_config_path': str(config), 'scene_config_sha256': queue.digest(config),
                    'stage_path': str(stage), 'stage_sha256': queue.digest(stage), 'trajectory_source': 'manual_driving',
                    'task': {'environment_evidence': {'collection_episode': {'scene': 'factory', 'episode_id': 'F01'}}},
                    'task_result': {'status': 'success', 'marked_by': 'keyboard:T', 'simulation_time_s': .15,
                                    'drone_position_m': [0.,0.,0.], 'nearest_trajectory_frame_id': 1},
                    'terminal_snapshot': row(.15, None)}
        queue.save(run / 'run_metadata.json', metadata)
        before = queue.digest(trajectory)
        result = queue.preflight(run, .1)
        self.assertEqual(result['selected_frames'], 3)
        self.assertEqual(before, queue.digest(trajectory))
        metadata['task_result']['simulation_time_s'] = .05
        metadata['terminal_snapshot']['simulation_time'] = .05
        queue.save(run / 'run_metadata.json', metadata)
        with self.assertRaisesRegex(ValueError, 'anchor'):
            queue.preflight(run, .1)
        del metadata['terminal_snapshot']
        queue.save(run / 'run_metadata.json', metadata)
        with self.assertRaisesRegex(ValueError, 'T confirmation'):
            queue.preflight(run, .1)

    def test_preflight_accepts_city_collection_recording(self):
        run = self.root / 'city-pilot30-C02-20261108T132307728798Z'; run.mkdir()
        config = self.root / 'city.yaml'; config.write_text('scene: city')
        stage = self.root / 'city.usda'; stage.write_text('#usda 1.0')
        task_config = self.root / 'C02.yaml'; task_config.write_text('task: C02')
        matrix = [[1., 0., 0., 0.], [0., 1., 0., 0.],
                  [0., 0., 1., 0.], [0., 0., 0., 1.]]
        def row(stamp, frame):
            return {'frame_id': frame, 'simulation_time': stamp,
                    'position': [0., 0., 0.],
                    **{key: copy.deepcopy(matrix) for key in (
                        'recorded_body_world_matrix', 'rgb_camera_world_matrix',
                        'thermal_camera_world_matrix')}}
        rows = [row(0., 0), row(1., 1)]
        (run / 'trajectory.jsonl').write_text(''.join(json.dumps(value) + '\n' for value in rows))
        metadata = {
            'run_id': run.name, 'scene_config_path': str(config),
            'scene_config_sha256': queue.digest(config),
            'stage_path': str(stage), 'stage_sha256': queue.digest(stage),
            'trajectory_source': 'manual_driving',
            'task_config_path': str(task_config),
            'task_config_sha256': queue.digest(task_config),
            'task': {'environment_evidence': {'city_collection': {
                'version': 'city20_long_start_remaining_v1', 'episode': 'C02',
                'split': 'train'}}},
            'task_result': {'status': 'success', 'marked_by': 'keyboard:T',
                            'simulation_time_s': 1.15, 'drone_position_m': [0., 0., 0.],
                            'nearest_trajectory_frame_id': 1},
            'terminal_snapshot': row(1.15, None),
        }
        queue.save(run / 'run_metadata.json', metadata)

        result = queue.preflight(run, 1.0)
        self.assertEqual((result['scene'], result['episode'], result['selected_frames']),
                         ('city', 'C02', 3))


if __name__ == '__main__':
    unittest.main()
