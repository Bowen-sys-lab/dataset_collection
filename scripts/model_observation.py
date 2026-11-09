"""Validate a selected recorded start view using its existing immutable-source index."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scene_intake.capture_integrity import verify_video_input
from scripts.frame_metadata import FrameMetadataReader
from scripts.trajectory_recorder import trajectory_provenance


def validate_first_observation(run, rgb_path):
    run, rgb_path = Path(run).resolve(), Path(rgb_path).resolve()
    replay = rgb_path.parent
    index = replay.parent/'dataset'

    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    verify_video_input(replay)
    header = read(index/'episode.json')
    if (Path(header['original_trajectory']).resolve() != run/'trajectory.jsonl'
        or header['original_trajectory_sha256'] != sha(run/'trajectory.jsonl')
        or header['original_metadata_sha256'] != sha(run/'run_metadata.json')
        or header['replay_metadata_sha256'] != sha(replay/'replay_metadata.json')):
        raise ValueError('Observation index no longer matches its recorded sources')
    replay_metadata = read(replay/'replay_metadata.json')
    if replay_metadata.get('result') != 'PASS' or (replay/'failure.txt').exists():
        raise ValueError('Observation requires a completed successful replay')
    trajectory = Path(replay_metadata['trajectory'])
    if not trajectory.is_absolute():
        trajectory = replay/trajectory
    actual_provenance = trajectory_provenance(trajectory)
    saved = header['replay_source_provenance']
    required = {'trajectory_sha256', 'source_run_metadata_sha256', 'task', 'task_result'}
    if not required.issubset(saved) or any(actual_provenance.get(key) != value for key, value in saved.items()):
        raise ValueError('Replay source provenance changed')
    with (index/'frames.jsonl').open(encoding='utf-8') as stream:
        first = json.loads(next(stream))
    with (run/'trajectory.jsonl').open(encoding='utf-8') as stream:
        initial = json.loads(next(stream))
    if (first['frame_index'] != 0 or Path(first['rgb']).resolve() != rgb_path
        or first['original_source_frame_id'] != initial['frame_id']
        or abs(float(first['observation_time_s']) - float(initial['simulation_time'])) > 1e-8):
        raise ValueError('Observation is not the recorded first frame')
    frame_path = replay/'frame_000000.json'
    frame = FrameMetadataReader().read(frame_path)
    actual = np.asarray(frame.get('restored_body_world_matrix'), dtype=float)
    expected = np.asarray(initial.get('recorded_body_world_matrix'), dtype=float)
    if (actual.shape != (4,4) or expected.shape != (4,4) or not np.isfinite(actual).all()
        or not np.isfinite(expected).all() or not np.allclose(actual, expected, rtol=0, atol=1e-5)
        or abs(float(frame['replay_simulation_time']) - float(initial['simulation_time'])) > 1e-8):
        raise ValueError('Observation pose or time differs from recorded start')
    camera = np.asarray(frame.get('camera_world_matrix'), dtype=float)
    recorded_camera = np.asarray(initial.get('rgb_camera_world_matrix'), dtype=float)
    if (camera.shape != (4,4) or recorded_camera.shape != (4,4) or not np.isfinite(camera).all()
        or not np.isfinite(recorded_camera).all() or not np.allclose(camera, recorded_camera, rtol=0, atol=1e-5)):
        raise ValueError('Observation camera differs from recorded start')
    return {'frame': frame, 'evidence': {path.name: sha(path) for path in
            [index/'episode.json', index/'frames.jsonl', replay/'replay_metadata.json', frame_path]},
            'scope': 'Selected first-view source and pose checks; no new training or pixel-sync certification'}
