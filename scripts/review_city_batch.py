"""Read-only technical review for a completed City replay queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.episode_replay_source import select_source
from scripts.trajectory_recorder import build_training_sequence
from scripts.bc_pose_targets import future_pose_target
from scripts.bc_action_protocol import checked_action
from scripts.frame_metadata import FrameMetadataReader


ROOT = Path(__file__).resolve().parents[1]
ACTION_MAX_S = 1.25


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def path_from_recorded(value):
    return Path(value.replace('\\\\', '\\'))


def mileage(rows):
    length = 0.0
    for left, right in zip(rows, rows[1:]):
        a, b = left.get('position'), right.get('position')
        if a is None or b is None:
            continue
        dt = float(right['simulation_time']) - float(left['simulation_time'])
        if dt > 0 and dt <= 1.25 + 1e-8:
            length += math.dist(a, b)
    return length


def review_item(item, task_by_episode):
    run = Path(item['run'])
    metadata_path = run / 'run_metadata.json'
    trajectory_path = run / 'trajectory.jsonl'
    metadata = read_json(metadata_path)
    raw = read_jsonl(trajectory_path)
    evidence = metadata['task']['environment_evidence']['city_collection']
    episode = evidence['episode']
    task = task_by_episode[episode]
    issues = []

    if metadata.get('run_id') != item['run_id']:
        issues.append('RunId mismatch')
    if metadata['task'].get('task_id') != task['task_id']:
        issues.append('task_id differs from manifest')
    if evidence.get('split') != task['split']:
        issues.append('split differs from manifest')
    if evidence.get('version') != task['task_version']:
        issues.append('task version differs from manifest')
    instruction = metadata['task'].get('instruction', '')
    if instruction != task['instruction_en']:
        issues.append('English instruction differs from manifest')
    task_file = path_from_recorded(metadata['task_config_path'])
    task_file_hash = sha256(task_file) if task_file.is_file() else None
    if task_file_hash != task['task_sha256'] or metadata.get('task_config_sha256') != task['task_sha256']:
        issues.append('task configuration hash differs from manifest')
    if metadata.get('sample_period_s') != 1.0:
        issues.append('recording sample period is not 1.0 s')
    if metadata.get('capture_contract', {}).get('contract_id') != 'paired_offline_v3':
        issues.append('recording capture contract is unexpected')
    if metadata.get('task_result', {}).get('marked_by') != 'keyboard:T' or not metadata.get('terminal_snapshot'):
        issues.append('exact keyboard:T terminal event missing')
    if metadata.get('collision_pending'):
        issues.append('unresolved collision')
    if math.dist(metadata['task']['start_position'], task['start_position_m']) > 1e-6:
        issues.append('start position differs from manifest')

    raw_times = [float(row['simulation_time']) for row in raw]
    raw_dts = [b - a for a, b in zip(raw_times, raw_times[1:])]
    if any(not math.isfinite(dt) or dt <= 0 for dt in raw_dts):
        issues.append('raw time is not strictly increasing')
    if any(b['frame_id'] <= a['frame_id'] for a, b in zip(raw, raw[1:])):
        issues.append('raw frame IDs are not strictly increasing')
    for row in raw + [metadata['terminal_snapshot']]:
        for key in ('recorded_body_world_matrix', 'rgb_camera_world_matrix', 'thermal_camera_world_matrix'):
            matrix = row.get(key)
            if not (isinstance(matrix, list) and len(matrix) == 4
                    and all(isinstance(line, list) and len(line) == 4 for line in matrix)
                    and all(math.isfinite(float(value)) for line in matrix for value in line)):
                issues.append('raw or terminal camera/body matrix is invalid')
                break
    regular_raw = [dt for dt in raw_dts if abs(dt - 1.0) <= 0.05]
    raw_off_period = [dt for dt in raw_dts if abs(dt - 1.0) > 0.05]

    source = select_source(raw, metadata, 1.0)
    training = read_jsonl(run / 'training_trajectory.jsonl')
    if training != build_training_sequence(raw, metadata):
        issues.append('persisted training trajectory differs from exact T/corrected derivation')
    training_metadata = read_json(run / 'training_sequence_metadata.json')
    if (training_metadata.get('source_trajectory_sha256') != sha256(trajectory_path)
            or training_metadata.get('source_run_metadata_sha256') != sha256(metadata_path)):
        issues.append('persisted training sequence provenance hash mismatch')
    source_dts = [float(b['simulation_time']) - float(a['simulation_time']) for a, b in zip(source, source[1:])]
    if any(dt > ACTION_MAX_S + 1e-8 for dt in source_dts):
        issues.append('corrected action interval exceeds 1.25 s')
    stops = [row for row in source if row.get('action_type') == 'STOP']
    if len(stops) != 1 or stops[-1].get('source_event_ref') != 'run_metadata.json#terminal_snapshot':
        issues.append('derived source does not have one exact terminal STOP')
    action_diagnostics = []
    for number, (first, later) in enumerate(zip(source, source[1:])):
        try:
            candidate = future_pose_target(first, later, initial_yaw=source[0]['yaw'])
            checked_action(candidate, schema='factory_pose_action_v2')
        except ValueError as error:
            action_diagnostics.append({'source_index': number, 'error': str(error)})

    mileage_file = read_json(run / 'mileage_quality.json')
    edits = metadata.get('collection_edit_events', [])
    collisions = sum(event.get('type') == 'collision' for event in edits)
    undos = sum(event.get('type') == 'undo' for event in edits)
    attempt = item['attempts'][-1] if item.get('attempts') else {}
    output = Path(attempt['output']) if attempt.get('output') else None
    output_summary = None
    pairs = 0
    if item.get('status') == 'COMPLETE' and output:
        pipeline = read_json(output / 'pipeline_status.json')
        replay = read_json(output / 'replay/replay_metadata.json')
        validation = read_json(output / 'replay/validation_summary.json')
        index = read_json(output / 'dataset/episode.json')
        frames = read_jsonl(output / 'dataset/frames.jsonl')
        pairs = len(frames)
        for name, expected in attempt.get('completion_hashes', {}).items():
            if not Path(name).is_file() or sha256(name) != expected:
                issues.append('completed artifact hash changed after queue validation')
        if pipeline.get('status') != 'COMPLETE' or any(
                stage.get('status') != 'PASS' for stage in pipeline.get('stages', [])
                if stage.get('name') in ('replay', 'validation', 'dataset_index')):
            issues.append('pipeline stage status is not PASS')
        if replay.get('result') != 'PASS':
            issues.append('replay result is not PASS')
        if not str(validation.get('result', '')).startswith('PASS'):
            issues.append('validation_summary failed')
        if index.get('run_id') != item['run_id'] or index.get('episode') != episode:
            issues.append('dataset index identity mismatch')
        if (index.get('original_trajectory_sha256') != sha256(trajectory_path)
                or index.get('original_metadata_sha256') != sha256(metadata_path)
                or index.get('replay_metadata_sha256') != sha256(output / 'replay/replay_metadata.json')):
            issues.append('dataset provenance hash mismatch')
        if index.get('instruction') != instruction or index.get('split') != task['split']:
            issues.append('dataset index task/split mismatch')
        if index.get('sample_interval_s') != 1.0:
            issues.append('dataset index sample interval is not 1.0 s')
        if index.get('rgb_resolution') != [256, 256] or index.get('thermal_resolution') != [256, 256]:
            issues.append('dataset index image size is unexpected')
        if index.get('training_ready') is not False or index.get('action_alignment_status') != 'NOT_CERTIFIED':
            issues.append('dataset index readiness flags need review')
        if len(frames) != len(source) or len(frames) != replay.get('frames'):
            issues.append('derived source, frame index, and replay frame counts differ')
        if frames:
            frame_times = [float(row['observation_time_s']) for row in frames]
            if any(b <= a for a, b in zip(frame_times, frame_times[1:])):
                issues.append('frame observation timestamps are not increasing')
            last = frames[-1]
            if last.get('action_type') != 'STOP' or last.get('source_event_ref') != 'run_metadata.json#terminal_snapshot':
                issues.append('last indexed frame is not exact T STOP')
            if sum(frame.get('action_type') == 'STOP' for frame in frames) != 1:
                issues.append('indexed STOP count differs from one')
        image_failures = []
        expected_ids = list(range(len(frames)))
        for kind in ('rgb', 'thermal'):
            actual_ids = sorted(int(path.name.split('_')[1])
                                for path in (output / 'replay').glob(f'frame_*_{kind}.png'))
            if actual_ids != expected_ids:
                image_failures.append(f'{kind} PNG count/sequence differs from frame index')
        import numpy as np
        from PIL import Image
        pose_reader = FrameMetadataReader()
        pose_failures = []
        for number, (frame, source_row) in enumerate(zip(frames, source)):
            if frame.get('frame_index') != number or abs(frame['observation_time_s'] - source_row['simulation_time']) > 1e-8:
                issues.append(f'frame {number} time/index differs from selected source')
            if frame.get('original_source_frame_id') != source_row.get('source_frame_id'):
                issues.append(f'frame {number} original frame ID differs from selected source')
            if frame.get('source_event_ref') != source_row.get('source_event_ref'):
                issues.append(f'frame {number} event reference differs from selected source')
            try:
                rendered = pose_reader.read(output / 'replay' / f'frame_{number:06d}.json', fields={
                    'frame_id', 'trajectory_sha256', 'task_id', 'task_result_status',
                    'trajectory_frame_id', 'replay_simulation_time', 'position',
                    'camera_world_matrix', 'thermal_camera_world_matrix',
                    'pose_restore_validation', 'restored_body_pose_max_error', 'recorded_camera_pose_max_error'})
                if (rendered['frame_id'] != number or rendered['trajectory_frame_id'] != source_row['frame_id']
                        or rendered['trajectory_sha256'] != replay['trajectory_provenance']['trajectory_sha256']
                        or rendered['task_id'] != task['task_id']
                        or abs(rendered['replay_simulation_time'] - source_row['simulation_time']) > 1e-8):
                    raise ValueError('rendered source identity differs from selected source')
                if not np.allclose(rendered['position'], source_row['position'], rtol=0, atol=1e-5):
                    raise ValueError('rendered body position differs from source')
                for actual, expected in [('camera_world_matrix', 'rgb_camera_world_matrix'),
                                         ('thermal_camera_world_matrix', 'thermal_camera_world_matrix')]:
                    if not np.allclose(rendered[actual], source_row[expected], rtol=0, atol=1e-5):
                        raise ValueError(f'{actual} differs from source')
                for name in ('restored_body_pose_max_error', 'recorded_camera_pose_max_error'):
                    if not math.isfinite(rendered[name]) or rendered[name] > 1e-5:
                        raise ValueError(f'{name} exceeds tolerance')
            except (ValueError, KeyError, OSError) as error:
                pose_failures.append(f'frame {number}: {error}')
            for kind in ('rgb', 'thermal'):
                image_path = path_from_recorded(frame[kind])
                if image_path.resolve().parent != (output / 'replay').resolve():
                    image_failures.append(f'frame {number} {kind} points outside replay')
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.load()
                        if image.size != (256, 256):
                            image_failures.append(f'frame {number} {kind} size {image.size}')
                except (OSError, ValueError) as error:
                    image_failures.append(f'frame {number} {kind}: {error}')
            for kind in ('dn16', 'radiance', 'apparent_t'):
                path = output / 'replay' / f'frame_{number:06d}_{kind}.npy'
                try:
                    array = np.load(path, allow_pickle=False)
                    if array.shape != (256, 256) or not np.isfinite(array).all():
                        image_failures.append(f'frame {number} {kind} invalid shape or values')
                except (OSError, ValueError) as error:
                    image_failures.append(f'frame {number} {kind}: {error}')
        if image_failures:
            issues.append('PNG/array independent inspection failed')
        if pose_failures:
            issues.append('restored pose/camera correspondence inspection failed')
        output_summary = {
            'pipeline_status': pipeline.get('status'),
            'stage_statuses': {stage.get('name'): stage.get('status') for stage in pipeline.get('stages', [])},
            'replay_result': replay.get('result'),
            'validation_result': validation.get('result'),
            'validation_failures': validation.get('failures', []),
            'frames': pairs,
            'capture_contract': index.get('replay_capture_contract'),
            'training_ready': index.get('training_ready'),
            'action_alignment_status': index.get('action_alignment_status'),
            'terminal_source_event_ref': frames[-1].get('source_event_ref') if frames else None,
            'png_and_array_failures': image_failures,
            'png_pairs_decoded': len(frames),
            'pose_and_camera_correspondence_failures': pose_failures,
        }
    else:
        issues.append('replay output is not complete')

    source_hashes = item.get('preflight', {}).get('source_hashes', {})
    hash_mismatches = []
    for name, expected in source_hashes.items():
        path = Path(name)
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            hash_mismatches.append(name)
    if hash_mismatches:
        issues.append('recording/configuration source hashes changed')

    corrected = mileage_file.get('corrected_branch', {}).get('raw_recording', {})
    return {
        'episode': episode,
        'run_id': item['run_id'],
        'split': task['split'],
        'task_id': task['task_id'],
        'instruction_match': instruction == task['instruction_en'],
        'task_config_sha256': task_file_hash,
        'raw_records': len(raw),
        'raw_regular_1s_intervals': len(regular_raw),
        'raw_off_period_intervals': raw_off_period,
        'selected_source_records': len(source),
        'selected_action_interval_max_s': max(source_dts, default=0.0),
        'selected_action_intervals_over_1_25_s': [dt for dt in source_dts if dt > ACTION_MAX_S + 1e-8],
        'future_pose_diagnostic_failures': action_diagnostics,
        'future_pose_diagnostic_count': max(0, len(source) - 1),
        'future_pose_diagnostic_semantics': 'read-only adjacent-pose legality probe; does not certify BC export or causal actions',
        'terminal_event': metadata.get('task_result'),
        'collision_events': collisions,
        'undo_events': undos,
        'raw_mileage': mileage_file.get('to_confirmation'),
        'corrected_branch_mileage': corrected,
        'corrected_sampled_length_to_stop_m': mileage(source),
        'output': str(output) if output else None,
        'replay': output_summary,
        'source_hashes_unchanged': not hash_mismatches,
        'source_hash_mismatches': hash_mismatches,
        'issues': issues,
    }


def review(queue_path, manifest_path):
    queue = read_json(queue_path)
    manifest = read_json(manifest_path)
    task_by_episode = {task['episode']: task for task in manifest['tasks']}
    items = [review_item(item, task_by_episode) for item in queue['items']]
    return {
        'schema_version': 1,
        'queue_id': queue['queue_id'],
        'queue_status': queue['status'],
        'sample_interval_s': queue['sample_interval_s'],
        'item_count': len(items),
        'complete_count': sum(not item['issues'] for item in items),
        'issue_count': sum(bool(item['issues']) for item in items),
        'independent_recordings': len({item['run_id'] for item in items}),
        'items': items,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = review(args.queue, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('queue_status', 'item_count', 'complete_count', 'issue_count', 'independent_recordings')}, ensure_ascii=False))
    for item in report['items']:
        print(f"{item['episode']} {item['run_id']} pairs={item['replay']['frames'] if item['replay'] else '-'} issues={'; '.join(item['issues']) or 'none'}")
    return 0 if report['queue_status'] == 'COMPLETE' and report['issue_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
