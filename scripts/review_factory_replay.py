"""Audit completed factory episodes and prepare endpoint images for review."""

import argparse
from copy import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RADIUS_M = 0.08


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def load_completed_episode(episode):
    """Bind completed pipeline, original manual record and replay index."""
    episode = Path(episode).resolve()
    hashes = {}

    def load(path):
        raw = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    index = episode / 'dataset' / 'episode.json'
    header = load(index)
    pipeline = load(episode / 'pipeline_status.json')
    if (header['scene'] != 'factory' or pipeline.get('status') != 'COMPLETE'
            or pipeline.get('run_id') != header['run_id']
            or Path(pipeline.get('index') or '.').resolve() != index):
        raise ValueError('Completed factory pipeline must point to this episode index')
    trajectory = Path(header['original_trajectory']).resolve()
    if trajectory.name != 'trajectory.jsonl' or trajectory.parent.name != header['run_id']:
        raise ValueError('Original recording does not match the indexed run')
    metadata_path = trajectory.parent / 'run_metadata.json'
    metadata = load(metadata_path)
    if (metadata.get('run_id') != header['run_id'] or metadata.get('trajectory_source') != 'manual_driving'
            or metadata.get('success') is not True or metadata.get('task_result', {}).get('marked_by') != 'keyboard:T'
            or not metadata.get('terminal_snapshot')):
        raise ValueError('Original manual recording with exact T snapshot required')
    task = metadata['task']
    recorded_episode = task['environment_evidence']['collection_episode']
    if (header['episode'] != recorded_episode['episode'] or header['task_id'] != task['task_id']
            or header['manual_task_result'] != metadata['task_result']):
        raise ValueError('Indexed task identity or T confirmation differs from the original recording')
    replay_path = episode / 'replay' / 'replay_metadata.json'
    replay = load(replay_path)
    validation = load(replay_path.with_name('validation_summary.json'))
    if (replay.get('result') != 'PASS' or validation.get('result') != 'PASS'
            or not (0 < header['frames'] == replay['frames'] == validation['frames_checked'])):
        raise ValueError('Replay completion and validated frame counts disagree')
    hashes[str(trajectory)] = file_hash(trajectory)
    for path, field in [(trajectory, 'original_trajectory_sha256'), (metadata_path, 'original_metadata_sha256'),
                        (replay_path, 'replay_metadata_sha256')]:
        if hashes[str(path.resolve())] != header[field]:
            raise ValueError(f'Indexed hash differs from {path}')
    return header, metadata, trajectory, hashes


def merge_hashes(destination, additional):
    for path, digest in additional.items():
        if path in destination and destination[path] != digest:
            raise ValueError(f'Conflicting source hashes for {path}')
        destination[path] = digest


def audit_episode(episode, completed, out, row, sources):
    from scripts.bc_action_protocol import encode_action
    from scripts.episode_dataset import write_episode_index
    from scripts.episode_replay_source import validate_recorded_scene_assets
    from scripts.prepare_bc_samples import prepare_candidates
    from scripts.replay_factory_inspection import validate_saved_frame
    from scripts.scene_config import load_scene_config
    from scripts.trajectory_recorder import build_training_sequence, read_trajectory

    header, metadata, trajectory, hashes = completed
    row.update(run_id=header['run_id'], episode=header['episode'], split=header['split'], integrity='RUNNING')
    environment = metadata['task']['environment_evidence']
    episode_contract = environment['collection_episode']
    if not episode_contract.get('source_snapshot_version'):
        raise ValueError('Recorded scene asset snapshot is required for collision review')
    validate_recorded_scene_assets(metadata)
    scene_assets = Path(episode_contract['scene_assets_path']).resolve()
    merge_hashes(hashes, read_json(scene_assets))
    hashes[str(scene_assets)] = episode_contract['scene_assets_sha256']
    inspection = environment['factory_inspection']
    geometry_reference = Path(inspection['geometry_file'])
    geometry = (ROOT / geometry_reference).resolve()
    if (geometry_reference.is_absolute() or not geometry.is_relative_to(ROOT)
            or file_hash(geometry) != inspection['geometry_sha256']):
        raise ValueError('Factory target geometry differs from the original recording')
    merge_hashes(hashes, {str(geometry): inspection['geometry_sha256']})
    config = load_scene_config(Path(metadata['scene_config_path']))
    stage = config.scene.stage.resolve()
    if (config.sha256 != metadata['scene_config_sha256'] or stage != Path(metadata['stage_path']).resolve()
            or file_hash(stage) != metadata['stage_sha256']):
        raise ValueError('Scene config or stage differs from the original recording')
    merge_hashes(hashes, {str(config.source_path.resolve()): config.sha256, str(stage): metadata['stage_sha256']})

    replay = episode / 'replay'
    for suffix in ('.json', '_rgb.png', '_thermal.png'):
        if sum(1 for _ in replay.glob(f'frame_*{suffix}')) != header['frames']:
            raise ValueError(f'Replay {suffix} frame count differs from index')
    replay_metadata = read_json(replay / 'replay_metadata.json')
    merge_hashes(hashes, replay_metadata['source_input_hashes'])
    with TemporaryDirectory(prefix='rebuild-', dir=out) as temp:
        scratch = Path(temp)
        # The normal indexer walks and authenticates the chain back to this exact raw run.
        index = write_episode_index(trajectory.parent, replay, scratch / 'index')
        frames = episode / 'dataset' / 'frames.jsonl'
        indexed_bytes = frames.read_bytes()
        merge_hashes(hashes, {str(frames.resolve()): hashlib.sha256(indexed_bytes).hexdigest()})
        original_rows = [json.loads(line) for line in indexed_bytes.splitlines()]
        rebuilt_rows = [json.loads(line) for line in (scratch / 'index' / 'frames.jsonl').read_bytes().splitlines()]
        if index['frames'] != header['frames'] or original_rows != rebuilt_rows:
            raise ValueError('Saved frame index differs from the index rebuilt from replay and recording')
        rebuilt = scratch / 'candidates'
        candidate = prepare_candidates(trajectory.parent, replay, rebuilt)
        samples = [json.loads(line) for line in (rebuilt / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
        merge_hashes(hashes, read_json(rebuilt / 'source_manifest.json'))

    for index, sample in enumerate(samples):
        for suffix in ('rgb', 'display8', 'radiance', 'apparent_t', 'dn16'):
            path = (replay / f'frame_{index:06d}_{suffix}.npy').resolve()
            merge_hashes(hashes, {str(path): file_hash(path)})
        validate_saved_frame(replay, index)
        encode_action(sample['pose_target'])
    if len(samples) != header['frames'] or candidate['counts']['STOP'] != 1:
        raise ValueError('Expected one exact T action and a complete indexed sample sequence')

    states = build_training_sequence(read_trajectory(trajectory), metadata)
    last = samples[-1]
    approach_time = last['observation_time_s'] - 10.0
    approach = min(samples[:-1] or samples, key=lambda sample: abs(sample['observation_time_s'] - approach_time))
    row.update(integrity='PASS', frames=len(samples), action_counts=candidate['counts'],
               candidate_samples_sha256=candidate['samples_sha256'], terminal_time_s=last['observation_time_s'],
               target=candidate['target'], instruction=header['instruction'], endpoint_review='PENDING_HUMAN',
               endpoint_images={'approach_rgb': approach['images']['rgb'], 'terminal_rgb': last['images']['rgb'],
                                'terminal_thermal': last['images']['thermal']})
    merge_hashes(sources, hashes)
    contract = (stage, metadata['stage_sha256'], config.drone.prim_path, config.drone.body_path,
                config.scene.static_targets)
    return config, contract, states, samples


def audit_collision(preview, row, config, states, samples):
    import numpy as np
    from scripts.collision_scene import build_collision_scene, check_path
    from scripts.drone_envelope import measure_drone_envelope
    from scripts.replay_pose import FrozenPoseRestorer, compare_pose_matrices, world_matrix

    expected, _ = FrozenPoseRestorer(preview.stage, config).restore(states[0])
    actual = world_matrix(preview.stage.GetPrimAtPath(config.drone.body_path))
    errors = compare_pose_matrices(actual, expected)
    if errors['translation_error_m'] > 1e-4 or errors['rotation_error_degrees'] > .01:
        raise ValueError('Recorded start pose could not be restored')
    envelope = measure_drone_envelope(preview.stage, config.drone.prim_path, config.drone.body_path)
    if envelope['rotation_envelope_radius_m'] >= RADIUS_M:
        raise ValueError('Composed robot exceeds the declared collision sphere')
    row['drone_envelope'] = envelope
    positions = np.asarray([state['position'] for state in states])
    bounds = np.array([positions.min(0) - 1.0, positions.max(0) + 1.0])
    scene = build_collision_scene(preview.stage, start=positions[0], bounds=bounds, radius=RADIUS_M,
                                  drone_root=config.drone.prim_path)
    row['collision_scene'] = scene.report
    source = check_path(scene, positions)
    row['source_path_check'] = source
    if source['status'] != 'CLEAR':
        row['collision'] = 'SOURCE_PATH_FINDING_OR_UNPROVEN'
        if 'path_interval' in source:
            row['first_source_interval_s'] = [states[index]['simulation_time'] for index in source['path_interval']]
        return

    by_time = {state['simulation_time']: state for state in states}
    local_scene = copy(scene)
    for sample in samples[:-1]:
        matrix_position = np.asarray(sample['body_world_matrix'])[3, :3]
        stamp = sample['observation_time_s']
        if stamp not in by_time or not np.allclose(matrix_position, by_time[stamp]['position'], rtol=0, atol=1e-5):
            raise ValueError('Candidate action does not start at a checked source observation')
        start = np.asarray(by_time[stamp]['position'])
        local_scene.start = start
        check = check_path(local_scene, [start, sample['future_endpoint']['position_m']])
        row['candidate_segments_checked'] += 1
        if check['status'] != 'CLEAR':
            row['candidate_findings'].append({'sample_id': sample['sample_id'], 'start_time_s': stamp,
                                             'end_time_s': sample['future_endpoint']['simulation_time_s'], 'check': check})
    row['collision'] = 'FINDINGS' if row['candidate_findings'] else 'CLEAR'


def write_endpoint_sheet(out, rows):
    lines = ['# 工厂回放终点复核', '', '结合接近画面确认设备身份，再用终点 RGB 和红外确认异常是否清晰。', '',
             '| 任务 | 目标 | 接近 RGB | 终点 RGB | 终点红外 | 画面结论 |', '|---|---|---|---|---|---|']
    for row in rows:
        if 'endpoint_images' not in row:
            continue
        links = [f'[打开](</{Path(image["path"]).as_posix()}>)' for image in row['endpoint_images'].values()]
        target = {'tank': '储罐壁的圆形热异常', 'pipe': '设备外壁竖管上的短热段'}.get(row['target'], row['target'])
        lines.append(f'| {row["episode"]}（{row["run_id"]}） | {target} | ' + ' | '.join(links) + ' | 待确认 |')
    for row in rows:
        if 'instruction' in row:
            lines.extend(['', f'**{row["episode"]}** {row["instruction"]}'])
    (out / 'endpoint-review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    episodes = [path.resolve() for path in args.episodes]
    out = args.out.resolve()
    if len(set(episodes)) != len(episodes) or any(out.is_relative_to(path) for path in episodes):
        parser.error('Supply distinct episode directories and a separate output directory')
    try:
        completed = [load_completed_episode(path) for path in episodes]
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    if any(out.is_relative_to(trajectory.parent) for _, _, trajectory, _ in completed):
        parser.error('Output must be outside the original recording directories')
    out.mkdir(parents=True, exist_ok=False)
    report = {'result': 'RUNNING', 'radius_m': RADIUS_M, 'runs': [], 'training_ready': False,
              'collision_scope': 'Continuous sphere sweep against composed static meshes'}
    sources = {str(Path(__file__).resolve()): file_hash(__file__)}
    app = preview = None
    try:
        inputs = []
        for episode, recording in zip(episodes, completed):
            row = {'path': str(episode), 'episode': recording[0]['episode'], 'integrity': 'RUNNING', 'collision': 'PENDING',
                   'candidate_segments_checked': 0, 'candidate_findings': []}
            report['runs'].append(row)
            write_json(out / 'report.json', report)
            inputs.append(audit_episode(episode, recording, out, row, sources))
            print(f'{row["episode"]}: integrity PASS, {row["frames"]} frames', flush=True)
        write_endpoint_sheet(out, report['runs'])
        config, contract, _, _ = inputs[0]
        if any(other != contract for _, other, _, _ in inputs):
            raise ValueError('Episodes have different recorded geometry; review them separately')
        busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        if busy:
            report.update(result='GPU_BUSY', gpu_processes=busy)
            return 2
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 160, 'height': 128})
        from scripts.start_crazyflie_dual_camera_view import open_preview
        preview, _ = open_preview(app, scene_config=config, start_controller=False, show_ui=False, width=160, height=128,
                                  fast=True, renderer='RayTracedLighting', disable_external_viewport=True, defer_thermal=True)
        root_layer = preview.stage.GetRootLayer().realPath
        if not root_layer or Path(root_layer).resolve() != contract[0]:
            raise ValueError('Composed collision stage differs from the recorded USD')
        report['collision_stage'] = str(contract[0])
        for row, (run_config, _, states, samples) in zip(report['runs'], inputs):
            row['collision'] = 'RUNNING'
            write_json(out / 'report.json', report)
            audit_collision(preview, row, run_config, states, samples)
            print(f'{row["episode"]}: collision {row["collision"]}', flush=True)
        for path, digest in sources.items():
            if file_hash(path) != digest:
                raise ValueError(f'Review input changed during the audit: {path}')
        manifest = out / 'source-sha256.json'
        write_json(manifest, sources)
        clear = all(row['collision'] == 'CLEAR' for row in report['runs'])
        report.update(result='PASS_AUTOMATIC_CHECKS' if clear else 'COLLISION_FINDINGS_OR_UNPROVEN',
                      source_files=len(sources), source_manifest_sha256=file_hash(manifest))
        return 0 if clear else 1
    except Exception as error:
        report.update(result='FAIL', error=f'{type(error).__name__}: {error}')
        for row in report['runs']:
            if row['integrity'] == 'RUNNING':
                row['integrity'] = 'FAIL'
            if row['collision'] == 'RUNNING':
                row['collision'] = 'UNPROVEN'
        (out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return 1
    finally:
        write_json(out / 'report.json', report)
        print(report['result'], flush=True)
        try:
            if preview is not None:
                preview.close()
        finally:
            if app is not None:
                app.close()


if __name__ == '__main__':
    raise SystemExit(main())
