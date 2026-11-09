import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ReplayReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.episode = self.root / 'completed'
        self.recording = self.root / 'run-1'
        self.recording.mkdir()
        self.trajectory = self.recording / 'trajectory.jsonl'
        self.trajectory.write_bytes(b'{"frame_id":0}\n')
        self.metadata = self.recording / 'run_metadata.json'
        self.manual_result = {'marked_by': 'keyboard:T', 'status': 'success'}
        self.write(self.metadata, {'run_id': 'run-1', 'trajectory_source': 'manual_driving',
                                   'success': True, 'task_result': self.manual_result,
                                   'terminal_snapshot': {'simulation_time': 1.0},
                                   'task': {'task_id': 'tank-1', 'environment_evidence': {
                                       'collection_episode': {'episode': 'F09'}}}})
        replay = self.episode / 'replay' / 'replay_metadata.json'
        self.write(replay, {'result': 'PASS', 'frames': 2})
        self.write(replay.parent / 'validation_summary.json', {'result': 'PASS', 'frames_checked': 2})
        self.index = self.episode / 'dataset' / 'episode.json'
        self.write(self.index, {'scene': 'factory', 'run_id': 'run-1', 'frames': 2,
                                'episode': 'F09', 'task_id': 'tank-1', 'manual_task_result': self.manual_result,
                                'original_trajectory': str(self.trajectory),
                                'original_trajectory_sha256': self.sha(self.trajectory),
                                'original_metadata_sha256': self.sha(self.metadata),
                                'replay_metadata_sha256': self.sha(replay)})
        self.pipeline = self.episode / 'pipeline_status.json'
        self.write(self.pipeline, {'status': 'COMPLETE', 'run_id': 'run-1', 'index': str(self.index)})

    @staticmethod
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_completed_sources_bind_to_index(self):
        from scripts.review_factory_replay import load_completed_episode
        _, metadata, trajectory, hashes = load_completed_episode(self.episode)
        self.assertEqual(trajectory, self.trajectory.resolve())
        self.assertEqual(metadata['run_id'], 'run-1')
        self.assertEqual(hashes[str(self.trajectory.resolve())], self.sha(self.trajectory))

    def test_modified_recording_is_rejected(self):
        from scripts.review_factory_replay import load_completed_episode
        self.trajectory.write_bytes(b'{"frame_id":4}\n')
        with self.assertRaisesRegex(ValueError, 'hash'):
            load_completed_episode(self.episode)

    def test_incomplete_or_misbound_pipeline_is_rejected(self):
        from scripts.review_factory_replay import load_completed_episode
        for change in ({'status': 'RUNNING'}, {'index': str(self.index.with_name('other.json'))}):
            pipeline = {'status': 'COMPLETE', 'run_id': 'run-1', 'index': str(self.index)}
            self.write(self.pipeline, {**pipeline, **change})
            with self.assertRaisesRegex(ValueError, 'pipeline'):
                load_completed_episode(self.episode)

    def test_relabelled_episode_is_rejected(self):
        from scripts.review_factory_replay import load_completed_episode
        index = json.loads(self.index.read_bytes())
        self.write(self.index, {**index, 'episode': 'F10'})
        with self.assertRaisesRegex(ValueError, 'task'):
            load_completed_episode(self.episode)

    def test_output_inside_original_recording_is_rejected_before_writing(self):
        from scripts.review_factory_replay import main
        output = self.recording / 'audit'
        with patch('sys.argv', ['review_factory_replay', '--episodes', str(self.episode), '--out', str(output)]):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
