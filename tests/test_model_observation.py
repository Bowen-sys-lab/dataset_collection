import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.episode_dataset import write_episode_index
from scripts.trajectory_recorder import trajectory_provenance


class FirstObservationTests(unittest.TestCase):
    def fixture(self, root):
        run = root/'raw'; run.mkdir()
        replay = root/'processed/replay'; replay.mkdir(parents=True)
        matrix = np.eye(4).tolist()
        metadata = {'task': {'task_id': 'task', 'instruction': 'Inspect the tank.'},
                    'task_result': {'status': 'unjudged'}, 'trajectory_source': 'manual_driving'}
        (run/'run_metadata.json').write_text(json.dumps(metadata))
        (run/'trajectory.jsonl').write_text(json.dumps({'frame_id': 0, 'simulation_time': 8.1,
            'recorded_body_world_matrix': matrix, 'rgb_camera_world_matrix': matrix})+'\n')
        (replay/'replay_metadata.json').write_text(json.dumps({'result': 'PASS',
            'trajectory': str(run/'trajectory.jsonl'), 'trajectory_provenance': trajectory_provenance(run/'trajectory.jsonl')}))
        (replay/'frame_000000.json').write_text(json.dumps({'trajectory_frame_id': 0, 'frame_id': 0,
            'replay_simulation_time': 8.1, 'restored_body_world_matrix': matrix, 'camera_world_matrix': matrix}))
        for kind in ['rgb', 'thermal']:
            (replay/f'frame_000000_{kind}.png').write_bytes(b'reference-image')
        write_episode_index(run, replay, replay.parent/'dataset')
        return run, replay

    def test_first_frame_is_checked_through_existing_index(self):
        from scripts.model_observation import validate_first_observation
        with tempfile.TemporaryDirectory() as folder:
            run, replay = self.fixture(Path(folder))
            result = validate_first_observation(run, replay/'frame_000000_rgb.png')
            self.assertEqual(result['frame']['replay_simulation_time'], 8.1)
            self.assertIn('episode.json', result['evidence'])

    def test_changed_source_or_pose_is_rejected(self):
        from scripts.model_observation import validate_first_observation
        for changed in ['source', 'pose', 'camera', 'replay']:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                run, replay = self.fixture(Path(folder))
                if changed == 'source':
                    (run/'run_metadata.json').write_text('{}')
                elif changed in ['pose', 'camera']:
                    path = replay/'frame_000000.json'; value = json.loads(path.read_text())
                    value['restored_body_world_matrix' if changed == 'pose' else 'camera_world_matrix'][3][0] = 1
                    path.write_text(json.dumps(value))
                else:
                    path = replay/'replay_metadata.json'; value = json.loads(path.read_text())
                    value['frames'] = 2; path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    validate_first_observation(run, replay/'frame_000000_rgb.png')


if __name__ == '__main__':
    unittest.main()
