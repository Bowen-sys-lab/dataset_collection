"""Isolated launch contract for the first, group-A factory BC demonstrations.

No Kit startup in --check-only. A finite smoke run is always labelled synthetic.
The existing preview owns recording; this entry verifies what it actually saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.scene_config import load_scene_config
from scripts.collection_capture_defaults import TRAJECTORY_PERIOD_S
from scripts.recording_quality import recording_quality
from scripts.manual_collision_undo import collision_undo_enabled, RECOVERY_CONTRACT
from scripts.factory_collection_config import resolve_selection, runtime_configuration

TASK_ID = 'model02_v7m_bc_pilot'
SPEEDS = {'max_horizontal_speed': 3.0, 'max_vertical_speed': 1.0,
          'max_yaw_speed': math.pi / 4}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_launch(run_id: str, *, smoke_frames: int = 0, target='tank', instruction_variant=1, start_group='A', collision_undo=None) -> dict:
    """Validate a fresh run and return the arguments and its provenance contract."""
    if not isinstance(run_id, str) or not re.fullmatch(r'factory-bc-(?:pilot|probe)-[A-Za-z0-9_-]+', run_id):
        raise ValueError('RunId must start with factory-bc-pilot- or factory-bc-probe- and contain only letters, digits, _ or -')
    if type(smoke_frames) is not int or not 0 <= smoke_frames <= 120:
        raise ValueError('smoke_frames must be an integer in 0..120')
    synthetic = smoke_frames > 0
    recovery = collision_undo_enabled(probe_control=synthetic, requested=collision_undo)
    if run_id.startswith('factory-bc-probe-') != synthetic:
        raise ValueError('Use factory-bc-probe-* only with --smoke-frames; manual runs use factory-bc-pilot-*')
    run_dir = ROOT / 'validation/trajectory_runs' / run_id
    output = ROOT / 'validation/factory-bc-pilot/previews' / run_id
    launch_path = ROOT / 'validation/factory-bc-pilot/launches' / (run_id + '.json')
    for path in (run_dir, output, launch_path):
        if path.exists():
            raise FileExistsError(f'Existing output will not be overwritten: {path}; choose a new RunId')

    config = load_scene_config(ROOT / 'configs/model02_v7m_bc_pilot_scene.yaml')
    selection=resolve_selection(target,instruction_variant,start_group)
    task=selection['task']
    runtime_dir=launch_path.parent/run_id
    if runtime_dir.exists(): raise FileExistsError(f'Existing runtime configuration: {runtime_dir}')
    config_path,config_bytes,task_path,task_bytes=runtime_configuration(selection,runtime_dir)
    if not config.task or config.task.get('task_id') != TASK_ID:
        raise ValueError('Pilot configuration does not load the approved pilot task')
    requirements = task['environment_evidence']['factory_inspection']['inspection_requirements']
    if (requirements['require_path_length'] is not False
            or requirements['hold_duration_s'] != 3.0
            or requirements['observation_distance_range_m'] != [6.0, 8.0]
            or requirements['maximum_observation_speed_m_s'] != 0.3):
        raise ValueError('Pilot observation requirements differ from the collection contract')
    args = ['--config', str(config_path), '--preview-mode', 'live',
            '--record-trajectory', '--run-id', run_id, '--no-save',
            '--speed-scale', '1', '--horizontal-speed', '3.0', '--vertical-speed', '1.0',
            '--trajectory-period', str(TRAJECTORY_PERIOD_S), '--preview-fps', '15', '--out', str(output)]
    args += ['--live-refresh','on-demand','--thermal-preview-fps','2',
             '--start-position',*[str(v) for v in task['start_position']],
             '--start-yaw-degrees',str(selection['yaw_degrees'])]
    args += ['--collision-undo' if recovery else '--no-collision-undo']
    if synthetic:
        args += ['--headless', '--probe-control', '--frames', str(smoke_frames)]
    return {'run_id': run_id, 'task_id': task['task_id'], 'start_group': start_group,
            'target':target,'instruction_variant':instruction_variant,'split':selection['split'],
            'expected_start_yaw_degrees':selection['yaw_degrees'],'task_snapshot':task,
            'runtime_config_text':config_bytes.decode('utf-8'),'runtime_task_text':task_bytes.decode('utf-8'),
            'trajectory_source': 'synthetic_control_probe' if synthetic else 'manual_driving',
            'controller_parameters': dict(SPEEDS), 'trajectory_period_s': TRAJECTORY_PERIOD_S,
            'collision_undo': recovery, 'collision_contract': RECOVERY_CONTRACT if recovery else 'disabled',
            'scene_config_path': str(config_path), 'scene_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'task_config_path': str(task_path), 'task_config_sha256': hashlib.sha256(task_bytes).hexdigest(),
            'stage_path': str(config.scene.stage), 'stage_sha256': _sha256(config.scene.stage),
            'expected_start_position_m': task['start_position'],
            'preview_resolution': [config.sensors.width, config.sensors.height],
            'run_dir': str(run_dir), 'preview_output': str(output),
            'launch_manifest_path': str(launch_path), 'preview_args': args,
            'preview_entrypoint': str(ROOT / 'scripts/start_crazyflie_dual_camera_view.py'),
            'task_acceptance': 'NOT_EVALUATED',
            'note': 'Launch contract only; T is human confirmation, not automatic task qualification.'}


def validate_recording(metadata: dict, launch: dict) -> dict:
    """Fail if the actual recorder did not use the requested task, speeds or start."""
    for field in ('scene_config_sha256', 'task_config_sha256', 'trajectory_source'):
        if metadata.get(field) != launch[field]:
            raise ValueError(f'Recorded {field} differs from pilot launch contract')
    if metadata.get('task', {}).get('task_id') != launch['task_id']:
        raise ValueError('Recorded task_id differs from pilot launch contract')
    actual = metadata.get('controller_parameters', {})
    for key, expected in launch['controller_parameters'].items():
        value = actual.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value - expected) > 1e-9:
            raise ValueError(f'Recorded controller parameter {key} must be {expected}, got {value}')
    start = metadata.get('start_position')
    if (not isinstance(start, list) or len(start) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in start)
            or math.dist(start, launch['expected_start_position_m']) > 0.25):
        raise ValueError('Recorded initial body position does not match group A (0.25 m tolerance)')
    count = metadata.get('record_count')
    if type(count) is not int or count <= 0:
        raise ValueError('Recording contains no trajectory samples')
    if 'expected_start_yaw_degrees' in launch:
        yaw=metadata.get('start_yaw_radians')
        if yaw is None or abs(math.atan2(math.sin(yaw-math.radians(launch['expected_start_yaw_degrees'])),
                                        math.cos(yaw-math.radians(launch['expected_start_yaw_degrees']))))>math.radians(1.):
            raise ValueError('Recorded start yaw differs from launch contract')
        if metadata.get('task') != launch['task_snapshot']:
            raise ValueError('Recorded target/instruction/start task snapshot differs from launch contract')
    return {'runtime_contract_verified': True, 'task_acceptance': 'NOT_EVALUATED',
            'trajectory_source': metadata['trajectory_source'], 'start_group': launch.get('start_group','A'),
            'record_count': count, 'actual_start_position_m': start,
            'actual_controller_parameters': actual,
            'recorded_task_result': metadata.get('task_result'),
            'note': 'Task completion and image visibility require separate replay review.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--target',choices=('tank','pipe'),default='tank')
    parser.add_argument('--instruction-variant',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--start-group',choices=('A','B','D','E','F'),default='A')
    parser.add_argument('--probe-ir-once',action='store_true',help='synthetic inspection in a finite smoke run only')
    parser.add_argument('--collision-undo', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke-frames', type=int, default=0,
                        help='finite headless synthetic probe, never a human demonstration')
    args = parser.parse_args(argv)
    try:
        launch = build_launch(args.run_id, smoke_frames=args.smoke_frames,target=args.target,
                              instruction_variant=args.instruction_variant,start_group=args.start_group, collision_undo=args.collision_undo)
        if args.probe_ir_once:
            if args.smoke_frames<=0:raise ValueError('--probe-ir-once requires a finite synthetic smoke run')
            launch['preview_args'].append('--probe-ir-once')
        if args.check_only:
            print(json.dumps(launch, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        path = Path(launch['launch_manifest_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        runtime_dir=Path(launch['scene_config_path']).parent
        runtime_dir.mkdir(exist_ok=False)
        Path(launch['scene_config_path']).write_bytes(launch['runtime_config_text'].encode('utf-8'))
        Path(launch['task_config_path']).write_bytes(launch['runtime_task_text'].encode('utf-8'))
        with path.open('x', encoding='utf-8') as stream:
            json.dump(launch, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(f'[factory-bc] {args.target} / description {args.instruction_variant} / group {args.start_group} | 3.0 / 1.0 m/s', flush=True)
        print('[factory-bc] Close after confirmation. T is a human mark; replay qualification is separate.', flush=True)
        # SimulationApp.close() normally fast-exits its entire process. Keep
        # provenance checks in this parent instead of placing them after close().
        code = subprocess.run([sys.executable, '-B', launch['preview_entrypoint'],
                               *launch['preview_args']], cwd=ROOT, check=False).returncode
        performance = json.loads((Path(launch['preview_output']) / 'performance.json').read_text(encoding='utf-8'))
        if code or performance.get('completion') not in ('completed', 'user_interrupted'):
            raise ValueError('Preview failed; inspect its log and performance.json before reusing this entry')
        if args.probe_ir_once:
            events=performance.get('inspection_events',[])
            if len(events)!=1 or events[0].get('status')!='COMPLETED' or not events[0].get('thermal_resources_released'):
                raise ValueError('Synthetic inspection did not complete and release its resources')
        run = Path(launch['run_dir'])
        metadata = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
        report = validate_recording(metadata, launch)
        from scripts.trajectory_recorder import read_trajectory
        quality=recording_quality(read_trajectory(run/'trajectory.jsonl'),
                                  sample_period_s=metadata.get('sample_period_s', .1), metadata=metadata)
        report['technical_quality']=quality
        if quality['record_count']!=metadata['record_count']: raise ValueError('Trajectory record count differs from metadata')
        (run/'technical_quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
        if quality['result']!='PASS': raise ValueError(f'Trajectory quality failed: {quality["issues"]}')
        if metadata.get('terminal_snapshot') is not None:
            if not (run/'training_trajectory.jsonl').is_file():raise ValueError('Missing terminal training export')
        if metadata.get('stage_sha256') != launch['stage_sha256']:
            raise ValueError('Recorded stage hash differs from pilot launch contract')
        (run / 'pilot_runtime_check.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'[factory-bc] Runtime parameters verified. Task qualification pending: {run}', flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'[factory-bc] Refused: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(main())
