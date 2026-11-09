"""Lightweight JSONL flight recorder and replay-file helpers.

The recorder is Isaac-independent.  It buffers records in memory and writes one
JSON object per line at shutdown, so disk I/O does not run in the driving loop.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


def configure_recording_timeline(timeline, minimum_remaining_s=86400.):
    """Extend the process's recording range so authored animation cannot loop its clock."""
    current = float(timeline.get_current_time())
    original_end = float(timeline.get_end_time())
    remaining = float(minimum_remaining_s)
    if not all(math.isfinite(v) for v in (current, original_end, remaining)) or remaining <= 0:
        raise ValueError('Recording timeline range must be finite and positive')
    original_looping = bool(timeline.is_looping())
    timeline.set_looping(False)
    timeline.set_end_time(max(original_end, current + remaining))
    timeline.commit()
    end = float(timeline.get_end_time())
    if timeline.is_looping() or end < current + remaining - 1e-8:
        raise ValueError('Recording timeline range was not applied')
    return {'original_end_time_s':original_end, 'original_looping':original_looping,
            'recording_end_time_s':end, 'looping':False,
            'scope':'runtime timeline only; source USD is not saved'}


def validate_recording_time(records, simulation_time):
    timestamp = float(simulation_time)
    if not math.isfinite(timestamp):
        raise ValueError('Recording time must be finite')
    if records and timestamp < float(records[-1]['simulation_time']) - 1e-8:
        raise ValueError('Recording time moved backwards; timeline may have looped or been rewound. '
                         'Raw samples are preserved; start a new recording epoch.')


def prepare_frozen_capture_timeline(timeline, simulation_time):
    """Keep Replicator's zero-time step inside a non-looping runtime range."""
    timestamp = float(simulation_time)
    current = float(timeline.get_current_time())
    if not math.isfinite(timestamp) or not math.isfinite(current):
        raise ValueError('Frozen capture time must be finite')
    required_end = max(timestamp, current) + 1.
    if not timeline.is_looping() and float(timeline.get_end_time()) >= required_end:
        return None
    return configure_recording_timeline(timeline, required_end-current)


def _json_value(value: Any):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def orientation_wxyz_from_world_transform(world_transform):
    """Extract the rigid WXYZ rotation from a scaled USD row-vector transform."""
    import numpy as np

    matrix = np.asarray(world_transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError('body world transform must be a 4x4 matrix')
    if not np.isfinite(matrix).all():
        raise ValueError('body world transform must contain only finite values')

    left, singular_values, right = np.linalg.svd(matrix[:3, :3])
    if singular_values[-1] <= 1e-12:
        raise ValueError('body world transform has a singular rotation/scale component')
    row_rotation = left @ right
    if np.linalg.det(row_rotation) < 0:
        left[:, -1] *= -1
        row_rotation = left @ right
    rotation = row_rotation.T

    # Stable matrix-to-quaternion conversion for a standard column-vector
    # rotation matrix. Canonicalize the sign so logs remain reproducible.
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
        quaternion = np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
        ])
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
        quaternion = np.array([
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
        ])
    else:
        scale = np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
        quaternion = np.array([
            (rotation[1, 0] - rotation[0, 1]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
        ])
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError('body world transform did not produce a valid orientation')
    quaternion /= norm
    if quaternion[0] < 0:
        quaternion *= -1
    return [float(value) for value in quaternion]


def trajectory_state_from_world_transform(controller_state: dict[str, Any], world_transform):
    """Use one evaluated USD body transform as the trajectory's canonical pose.

    The controller pose is retained separately because it is a command/target
    state, not necessarily the pose that the cameras rendered at this sample.
    """
    import numpy as np
    from scripts.drone_control_math import rpy_from_orientation_wxyz

    matrix = np.asarray(world_transform, dtype=np.float64)
    orientation = orientation_wxyz_from_world_transform(matrix)
    roll, pitch, yaw = rpy_from_orientation_wxyz(orientation)

    state = copy.deepcopy(controller_state)
    state['controller_target_pose'] = {
        key: copy.deepcopy(controller_state[key])
        for key in ('position', 'roll', 'pitch', 'yaw', 'orientation')
        if key in controller_state
    }
    state.update({
        'position': [float(value) for value in matrix[3, :3]],
        'roll': float(roll),
        'pitch': float(pitch),
        'yaw': float(yaw),
        'orientation': orientation,
        'recorded_body_world_matrix': matrix.tolist(),
        'trajectory_pose_source': 'usd_body_world_transform',
    })
    return state


def _validate_terminal_snapshot(snapshot, timestamp, position):
    import numpy as np
    if (not math.isfinite(float(timestamp)) or snapshot.get('simulation_time') != timestamp
            or snapshot.get('position') != list(position)):
        raise ValueError('terminal event time/position differs from sampled pose')
    for key in ('recorded_body_world_matrix', 'rgb_camera_world_matrix', 'thermal_camera_world_matrix'):
        value = np.asarray(snapshot.get(key), dtype=float)
        if value.shape != (4,4) or not np.isfinite(value).all():
            raise ValueError(f'complete terminal {key} required')
    if not np.allclose(np.asarray(snapshot['recorded_body_world_matrix'])[3,:3], position, atol=1e-6):
        raise ValueError('terminal body matrix does not match position')
    json.dumps(snapshot, allow_nan=False)


def build_training_sequence(records, metadata):
    """Derive a terminal sequence without modifying or relabelling the raw run."""
    result = metadata.get('task_result', {})
    terminal = metadata.get('terminal_snapshot')
    if (metadata.get('trajectory_source') != 'manual_driving' or result.get('status') != 'success'
            or result.get('marked_by') != 'keyboard:T' or not terminal):
        raise ValueError('manual T confirmation and complete terminal snapshot required')
    timestamp = result['simulation_time_s']
    _validate_terminal_snapshot(terminal, timestamp, result['drone_position_m'])
    anchor_id = result.get('nearest_trajectory_frame_id')
    if anchor_id is not None:
        anchor = next((r for r in records if r['frame_id'] == anchor_id), None)
        if anchor is None or float(anchor['simulation_time']) > float(timestamp) + 1e-8:
            raise ValueError('Manual confirmation time precedes its recorded trajectory anchor; '
                             'timeline epoch is inconsistent. Do not export this terminal sequence.')
    from scripts.recording_undo import active_branch
    if metadata.get('collection_branch_applied'):
        raise ValueError('Corrected source already derived; use its recorded sequence without applying undo twice')
    if metadata.get('collision_pending'):
        raise ValueError('Resolve collision before exporting a successful branch')
    branch,offset=active_branch(records,metadata)
    selected = [copy.deepcopy(r) for r in branch
                if float(r.get('source_simulation_time_s',r['simulation_time'])) < timestamp - 1e-9]
    endpoint=copy.deepcopy(terminal)
    if metadata.get('collection_edit_events'):
        endpoint['source_simulation_time_s']=timestamp
        endpoint['simulation_time']=float(timestamp)-offset
        selected=[row for row in selected if float(row['simulation_time'])<endpoint['simulation_time']-1e-9]
    selected.append(endpoint)
    previous = -math.inf
    for i, row in enumerate(selected):
        t = float(row['simulation_time'])
        if not math.isfinite(t) or t <= previous:
            raise ValueError('training sequence timestamps must strictly increase')
        previous = t
        row['source_frame_id'] = row['frame_id'] if i < len(selected)-1 else None
        row['frame_id'] = i
        row['action_type'] = 'STOP' if i == len(selected)-1 else 'MOTION'
        if row['action_type']=='STOP': row['source_event_ref']='run_metadata.json#terminal_snapshot'
    return selected


@dataclass
class TrajectoryRecorder:
    run_dir: Path
    metadata: dict[str, Any]
    records: list[dict[str, Any]]
    _closed: bool = False
    _mileage: Any = None

    @property
    def mileage(self):
        if self._mileage is None:
            from scripts.collection_mileage import MileageTracker
            from scripts.collection_capture_defaults import maximum_record_gap
            self._mileage = MileageTracker(max_gap_s=maximum_record_gap(self.metadata.get('sample_period_s', .1)))
        return self._mileage

    @classmethod
    def create(cls, root: str | Path, metadata: dict[str, Any], run_id: str | None = None):
        stamp = run_id or time.strftime('%Y%m%d-%H%M%S')
        run_dir = Path(root).resolve() / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        return cls(run_dir, copy.deepcopy(metadata), [])

    def append(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError('trajectory recorder is closed')
        normalized = {key: _json_value(value) for key, value in record.items()}
        if not isinstance(normalized.get('frame_id'), int):
            raise ValueError('trajectory record requires an integer frame_id')
        if 'simulation_time' not in normalized:
            raise ValueError('trajectory record requires simulation_time')
        if self.records and normalized['frame_id'] <= self.records[-1]['frame_id']:
            raise ValueError('trajectory frame_id must increase strictly')
        if self.records and float(normalized['simulation_time']) < float(self.records[-1]['simulation_time']):
            raise ValueError('trajectory simulation_time must be monotonic')
        self.records.append(normalized)
        self.mileage.update(normalized['simulation_time'], normalized.get('position'))

    def mark_success(self, simulation_time: float, position, *, elapsed_time: float | None = None,
                     terminal_snapshot: dict | None = None) -> bool:
        """Record one manually confirmed task success; never overwrite a prior result."""
        if self._closed or self.metadata.get('success') is not None:
            return False
        if self.metadata.get('collision_pending'):
            raise ValueError('Collision hold must be resolved before T confirmation')
        if self.metadata.get('trajectory_source', '').startswith('synthetic'):
            raise ValueError('synthetic trajectory cannot carry human T confirmation')
        simulation_time = float(simulation_time)
        validate_recording_time(self.records, simulation_time)
        elapsed_time = simulation_time if elapsed_time is None else float(elapsed_time)
        position = [float(value) for value in position]
        if not math.isfinite(simulation_time) or not math.isfinite(elapsed_time) or elapsed_time < 0 or len(position) != 3 or not all(
                math.isfinite(value) for value in position):
            raise ValueError('success mark requires finite simulation time and 3D position')
        previous_frame_id = self.records[-1]['frame_id'] if self.records else None
        if terminal_snapshot is not None:
            _validate_terminal_snapshot(terminal_snapshot, simulation_time, position)
            self.metadata['terminal_snapshot'] = copy.deepcopy(terminal_snapshot)
        self.metadata.update({
            'success': True,
            'success_time': elapsed_time,
            'success_simulation_time_s': simulation_time,
            'success_position': position,
            'success_marked_by': 'keyboard:T',
            'success_trajectory_frame_id': previous_frame_id,
            'task_result': {
                'status': 'success',
                'marked_by': 'keyboard:T',
                'simulation_time_s': simulation_time,
                'elapsed_time_s': elapsed_time,
                'drone_position_m': position,
                'nearest_trajectory_frame_id': previous_frame_id,
            },
        })
        return True

    def record_collision(self, simulation_time, detail):
        from scripts.recording_undo import append_event
        if self._closed or self.metadata.get('success') is not None:
            raise ValueError('Cannot edit a closed or confirmed run')
        validate_recording_time(self.records,float(simulation_time))
        if self.metadata.get('collision_pending'):return
        append_event(self.metadata,dict(type='collision',simulation_time_s=float(simulation_time),detail=detail))
        self.metadata['collision_pending']=True

    def undo_to(self, anchor_frame_id, simulation_time):
        from scripts.recording_undo import active_branch,append_event
        if self._closed or self.metadata.get('success') is not None or not self.metadata.get('collision_pending'):
            raise ValueError('Undo requires an unconfirmed collision hold')
        validate_recording_time(self.records,float(simulation_time))
        branch,_=active_branch(self.records,self.metadata)
        anchor=next((r for r in branch if r['frame_id']==anchor_frame_id),None)
        if anchor is None:raise ValueError('Undo anchor is not on the active branch')
        event=append_event(self.metadata,dict(type='undo',anchor_frame_id=anchor_frame_id,
            simulation_time_s=float(simulation_time),next_raw_frame_id=len(self.records),
            anchor_source_time_s=anchor.get('source_simulation_time_s',anchor['simulation_time'])))
        self.metadata['collision_pending']=False
        return event

    def flush(self) -> Path:
        """Write the current in-memory records without ending the recorder."""
        trajectory = self.run_dir / 'trajectory.jsonl'
        with trajectory.open('w', encoding='utf-8', newline='\n') as stream:
            for record in self.records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
        metadata = dict(self.metadata)
        if isinstance(metadata.get('task'), dict) and metadata['task']:
            if metadata.get('success') is True:
                metadata.setdefault('task_result', {'status': 'success'})
            else:
                metadata['success'] = None
                metadata['task_result'] = {'status': 'unjudged'}
        metadata.update({'record_count': len(self.records), 'trajectory_file': str(trajectory)})
        (self.run_dir / 'run_metadata.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        from scripts.collection_mileage import summarize_mileage
        mileage_report = summarize_mileage(self.records, metadata)
        if self.metadata.get('collection_edit_events'):
            from scripts.recording_undo import active_branch
            branch,_=active_branch(self.records,self.metadata)
            mileage_report['path_semantics']='raw attempts; includes discarded motion and recovery discontinuities'
            corrected=build_training_sequence(self.records,self.metadata) if self.metadata.get('terminal_snapshot') else branch
            # Corrected times differ from the untouched original T result; the branch is already cropped.
            corrected_metadata={**metadata,'task_result':{},'terminal_snapshot':None,'success':None}
            mileage_report['corrected_branch']=summarize_mileage(corrected,corrected_metadata)
        mileage_report['source_trajectory_sha256'] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        mileage_report['source_metadata_sha256'] = hashlib.sha256((self.run_dir/'run_metadata.json').read_bytes()).hexdigest()
        (self.run_dir/'mileage_quality.json').write_text(
            json.dumps(mileage_report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        if self.metadata.get('terminal_snapshot') is not None:
            sequence = build_training_sequence(self.records, self.metadata)
            (self.run_dir / 'training_trajectory.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in sequence),
                encoding='utf-8')
            (self.run_dir / 'training_sequence_metadata.json').write_text(json.dumps({
                'source_trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                'source_run_metadata_sha256': hashlib.sha256((self.run_dir/'run_metadata.json').read_bytes()).hexdigest(),
                'qualification': 'MANUAL_CONFIRMED', 'automatic_task_acceptance': 'NOT_EVALUATED',
                'terminal_semantics': 'human T event converted to STOP; raw recording unchanged',
                'record_count': len(sequence),
                'command_semantics': 'sampled command at source time; intervening input changes not reconstructed',
                **({'collection_edit_schema':self.metadata['collection_edit_schema'],
                    'collection_edit_events':self.metadata['collection_edit_events'],
                    'time_semantics':'corrected simulation_time; source_simulation_time_s is original clock'}
                   if self.metadata.get('collection_edit_events') else {}),
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        return trajectory

    def close(self) -> Path:
        # STOP ends this timeline epoch; normal shutdown can close again safely.
        if self._closed:
            return self.run_dir / 'trajectory.jsonl'
        trajectory = self.flush()
        self._closed = True
        return trajectory


def trajectory_provenance(path: str | Path, *, _seen=None) -> dict[str, Any]:
    """Copy task/label evidence from this recorded run, never from today's YAML."""
    path = Path(path).resolve()
    seen=set() if _seen is None else set(_seen)
    if path in seen or len(seen)>=8:
        raise ValueError('cyclic or excessive derived source chain')
    seen.add(path)
    run_path = path.parent / 'run_metadata.json'
    raw = run_path.read_bytes() if run_path.is_file() else None
    run = json.loads(raw.decode('utf-8')) if raw is not None else {}
    if not isinstance(run, dict):
        raise ValueError(f'{run_path}: expected a JSON object')
    task = copy.deepcopy(run.get('task')) or None
    if task is not None and not isinstance(task, dict):
        raise ValueError(f'{run_path}: task must be a mapping')
    result = copy.deepcopy(run.get('task_result'))
    if result is None:
        success = run.get('success', (task or {}).get('success'))
        result = {'status': 'success' if success is True else
                  'failed' if success is False else 'unjudged'}
        if success is True:
            result['marked_by'] = run.get('success_marked_by', 'legacy_recorded_label')
            result['simulation_time_s'] = run.get('success_simulation_time_s', run.get('success_time'))
    if not isinstance(result, dict) or result.get('status') not in ('success', 'unjudged', 'failed'):
        raise ValueError(f'{run_path}: invalid task_result')
    report = {
        'trajectory_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_run_id': path.parent.name,
        'trajectory_source': run.get('trajectory_source', 'legacy_unclassified'),
        'recording_scene_config_path': run.get('scene_config_path'),
        'recording_scene_config_sha256': run.get('scene_config_sha256'),
        'recording_stage_sha256': run.get('stage_sha256'),
        'recording_sensor_config_sha256': run.get('sensor_config_sha256'),
        'recording_thermal_config_sha256': run.get('thermal_config_sha256'),
        'thermal_input_snapshot': copy.deepcopy(run.get('thermal_input_snapshot')),
        'source_run_metadata_path': str(run_path) if raw is not None else None,
        'source_run_metadata_sha256': hashlib.sha256(raw).hexdigest() if raw is not None else None,
        'task': task,
        'task_result': result,
        'terminal_snapshot': copy.deepcopy(run.get('terminal_snapshot')),
        'task_metadata_status': 'recorded' if task else 'missing',
        'task_config_path': run.get('task_config_path'),
        'task_config_sha256': run.get('task_config_sha256'),
        'task_cli_overrides': copy.deepcopy(run.get('task_cli_overrides', {})),
    }
    if 'capture_contract' in run:
        report['capture_contract'] = copy.deepcopy(run['capture_contract'])
        report['sample_period_s'] = run['sample_period_s']
    for key in ('collection_edit_schema','collection_edit_events','collision_pending','collision_guard',
                'collection_branch_applied'):
        if key in run:report[key]=copy.deepcopy(run[key])
    if run.get('source_trajectory_path') and run.get('source_derivation_schema_version') == 1:
        parent=Path(run['source_trajectory_path']).resolve()
        parent_meta=parent.parent/'run_metadata.json'
        if (not parent.is_file() or not parent_meta.is_file()
                or hashlib.sha256(parent.read_bytes()).hexdigest()!=run.get('source_trajectory_sha256')
                or hashlib.sha256(parent_meta.read_bytes()).hexdigest()!=run.get('source_run_metadata_sha256')):
            raise ValueError('derived source recording changed or is missing')
        provenance=trajectory_provenance(parent,_seen=seen)
        if report['task']!=provenance['task'] or report['task_result']!=provenance['task_result']:
            raise ValueError('derived source task or manual result changed')
        for key in ('collection_edit_schema','collection_edit_events','collision_pending','collision_guard'):
            if report.get(key)!=provenance.get(key):
                raise ValueError('derived source collision edit history or protection contract changed')
        report['source_derivation']={'trajectory_path':str(parent),
            'trajectory_sha256':provenance['trajectory_sha256'],
            'metadata_sha256':provenance['source_run_metadata_sha256'],
            'source_run_id':provenance['source_run_id'],'not_independent_demonstration':True}
    return report


def read_trajectory(path: str | Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}: invalid JSON on line {line_number}') from exc
        if not isinstance(item, dict):
            raise ValueError(f'{path}: line {line_number} is not an object')
        result.append(item)
    if not result:
        raise ValueError(f'{path}: trajectory is empty')
    frame_ids = [item.get('frame_id') for item in result]
    if any(not isinstance(value, int) for value in frame_ids) or len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f'{path}: frame_id values must be unique integers')
    times = [float(item['simulation_time']) for item in result]
    if times != sorted(times):
        raise ValueError(f'{path}: simulation_time must be monotonic')
    return result
