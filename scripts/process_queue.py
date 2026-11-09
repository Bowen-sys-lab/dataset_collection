"""Serial, resumable wrapper around process.ps1; never edits source recordings."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from scripts.collection_capture_defaults import SAMPLE_PERIOD_S
from scripts.collection_shell import powershell_executable

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def save_queue(path, state):
    save(path, state)
    lines = [f"Queue: {Path(path).resolve()}", f"Status: {state['status']}",
             'Technical processing only; task success / training readiness remain separate.', '']
    for item in state['items']:
        last = item['attempts'][-1] if item['attempts'] else {}
        lines.append(f"{item['status']} | {item['run_id']} | frames={last.get('frames', '-')} "
                     f"| seconds={last.get('elapsed_s', '-')} | attempts={len(item['attempts'])}")
        for key in ('output', 'log'):
            if last.get(key):
                lines.append(f"  {key}: {last[key]}")
        if item.get('error'):
            lines.append(f"  error: {item['error']}")
    Path(path).with_suffix('.summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


@contextmanager
def exclusive(path):
    """OS releases the lock even if the worker is killed; no stale PID guessing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another queue worker holds {path}') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def preflight(run, interval):
    """CPU-only checks; existing replay remains responsible for full technical QC."""
    from scripts.episode_replay_source import recorded_backend, recorded_target_snapshot, select_source
    run = Path(run).resolve()
    metadata = read_json(run / 'run_metadata.json')
    evidence = metadata['task']['environment_evidence']
    episode = evidence.get('city_collection')
    scene = 'city' if episode else recorded_backend(metadata)
    if not episode:
        episode = evidence['collection_episode']
    if metadata.get('run_id') != run.name:
        raise ValueError('RunId differs from run_metadata.json')
    paths = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Recorded snapshot changed: {path}')
        paths[str(path)] = actual

    check(run / 'trajectory.jsonl'); check(run / 'run_metadata.json')
    for key in ('scene_config', 'stage'):
        check(metadata[key + '_path'], metadata[key + '_sha256'])
    if episode.get('source_snapshot_version'):
        asset_path = Path(episode['scene_assets_path'])
        check(asset_path, episode['scene_assets_sha256'])
        assets = read_json(asset_path)
        if not isinstance(assets, dict) or not assets:
            raise ValueError('Empty scene asset snapshot')
        for path, expected in assets.items():
            check(path, expected)
    for path, expected in metadata.get('thermal_input_snapshot', {}).get('file_sha256', {}).items():
        check(path, expected)
    if scene == 'forest':
        check(recorded_target_snapshot(metadata))
    rows = [json.loads(line) for line in (run / 'trajectory.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    # Validate the entire raw epoch, including samples after T, before terminal cropping.
    previous = -math.inf
    previous_id = -1
    for row in rows:
        stamp = float(row['simulation_time'])
        if not math.isfinite(stamp) or stamp <= previous or row['frame_id'] <= previous_id:
            raise ValueError('Raw trajectory times/frame IDs must strictly increase')
        previous, previous_id = stamp, row['frame_id']
        for key in ('recorded_body_world_matrix', 'rgb_camera_world_matrix', 'thermal_camera_world_matrix'):
            matrix = row.get(key)
            if (not isinstance(matrix, list) or len(matrix) != 4
                    or any(not isinstance(r, list) or len(r) != 4 for r in matrix)
                    or any(not math.isfinite(float(v)) for r in matrix for v in r)):
                raise ValueError(f'Missing or invalid {key}')
    if not metadata.get('terminal_snapshot'):
        raise ValueError('Human T confirmation / terminal snapshot missing')
    selected = select_source(rows, metadata, interval)
    return {'scene': scene, 'episode': episode.get('episode', episode.get('episode_id', episode.get('id'))),
            'raw_frames': len(rows), 'selected_frames': len(selected),
            'confirmation_time_s': metadata['task_result']['simulation_time_s'],
            'source_hashes': paths}


def busy_processes():
    # Check known project renderer/driver entrypoints, not every CPU-only kit.exe job.
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.Name -match '^(kit|python|pythonw|powershell|pwsh)(\\.exe)?$' "
               "-and $_.CommandLine -match '(?i)(start_crazyflie_dual_camera_view\\.py|replay_factory_inspection\\.py|"
               "replay_trajectory_highres\\.py|forest_collection\\.py|process_collection_episode|"
               "[\\\\/]record\\.ps1|[\\\\/]process\\.ps1)' } | "
               "Select-Object ProcessId,Name | ConvertTo-Json -Compress")
    result = subprocess.run([str(powershell_executable()), '-NoProfile', '-Command', command],
                            capture_output=True, text=True, check=True)
    value = json.loads(result.stdout.strip()) if result.stdout.strip() else []
    return value if isinstance(value, list) else [value]


def replay(run, output, log, interval):
    command = [str(powershell_executable()), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
               str(ROOT / 'scripts/process.ps1'), '-Run', str(run), '-Out', str(output),
               '-SampleInterval', str(interval)]
    with Path(log).open('wb') as stream:
        return subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode


def completion(output, run_id):
    output = Path(output)
    status_path = output / 'pipeline_status.json'
    status = read_json(status_path)
    if status.get('status') != 'COMPLETE' or status.get('run_id') != run_id:
        raise ValueError('Pipeline did not report COMPLETE for this RunId')
    replay_path = output / 'replay/replay_metadata.json'
    if read_json(replay_path).get('result') != 'PASS':
        raise ValueError('Replay final metadata missing or not PASS')
    index = output / 'dataset/episode.json'
    if not index.is_file():
        raise ValueError('Episode data index missing')
    if read_json(index).get('run_id') != run_id:
        raise ValueError('Episode index RunId mismatch')
    return {str(p): digest(p) for p in (status_path, replay_path, index)}


def create_queue(runs, path, interval=SAMPLE_PERIOD_S, *, checker=preflight):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('Sample interval must be finite and positive')
    path = Path(path).resolve()
    if path.exists():
        raise ValueError('Queue already exists; use -Resume')
    resolved = []
    for value in runs:
        candidate = Path(value)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = ROOT / 'validation/trajectory_runs' / candidate
        candidate = candidate.resolve()
        if candidate in resolved:
            raise ValueError(f'Duplicate recording: {candidate}')
        resolved.append(candidate)
    if not resolved:
        raise ValueError('Select at least one actual RunId or recording directory')
    state = {'schema_version': 1, 'queue_id': uuid.uuid4().hex, 'created_at': now(),
             'status': 'PENDING', 'sample_interval_s': interval, 'items': []}
    for run in resolved:
        item = {'run': str(run), 'run_id': run.name, 'status': 'PENDING', 'attempts': []}
        try:
            item['preflight'] = checker(run, interval)
        except Exception as error:
            item.update(status='INVALID', error=str(error))
        state['items'].append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_queue(path, state)
    return state


def execute(path, *, checker=preflight, runner=replay, busy=busy_processes, verify=completion):
    path = Path(path).resolve()
    with exclusive(path.with_suffix('.lock')), exclusive(ROOT / 'validation/process_queues/.worker.lock'):
        state = read_json(path)
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported queue schema')
        state.update(status='RUNNING', updated_at=now())
        save_queue(path, state)
        try:
            for item in state['items']:
                try:
                    current = checker(Path(item['run']), state['sample_interval_s'])
                    if item.get('preflight') and current != item['preflight']:
                        raise ValueError('Recording/configuration changed since queue creation; create a new queue')
                    item['preflight'] = current
                    if item['status'] == 'COMPLETE':
                        attempt = item['attempts'][-1]
                        if verify(attempt['output'], item['run_id']) != attempt['completion_hashes']:
                            raise ValueError('Completed result changed; inspect it before creating a new queue')
                        print(f"SKIP completed: {item['run_id']}", flush=True)
                        continue
                    if busy():
                        state.update(status='BLOCKED', error='Another project capture/replay is active; close it normally and resume')
                        save_queue(path, state)
                        return 2
                    # Every interrupted/failed attempt keeps its directory and log.
                    if item['attempts'] and item['attempts'][-1]['status'] == 'RUNNING':
                        item['attempts'][-1].update(status='INTERRUPTED', finished_at=now())
                    number = len(item['attempts']) + 1
                    output = ROOT / 'validation/episodes' / item['run_id'] / f"queue-{state['queue_id']}-{number:03d}"
                    log = path.parent / f"{path.stem}-{state['items'].index(item)+1:03d}-{number:03d}.log"
                    attempt = {'status': 'RUNNING', 'output': str(output), 'log': str(log), 'started_at': now()}
                    item['attempts'].append(attempt)
                    item.update(status='RUNNING'); item.pop('error', None)
                    save_queue(path, state)
                    print(f"START {item['run_id']} | log: {log}", flush=True)
                    started = time.monotonic()
                    try:
                        code = runner(Path(item['run']), output, log, state['sample_interval_s'])
                        attempt['exit_code'] = code
                        if code:
                            raise RuntimeError(f'process.ps1 exited {code}; see {log}')
                        attempt['completion_hashes'] = verify(output, item['run_id'])
                        if (output / 'replay/replay_metadata.json').is_file():
                            attempt['frames'] = read_json(output / 'replay/replay_metadata.json').get('frames')
                        item['status'] = attempt['status'] = 'COMPLETE'
                    except Exception as error:
                        item['status'] = attempt['status'] = 'FAILED'
                        item['error'] = attempt['error'] = str(error)
                    finally:
                        attempt.update(finished_at=now(), elapsed_s=time.monotonic()-started)
                except Exception as error:
                    item.update(status='INVALID', error=str(error))
                save_queue(path, state)
                print(f"{item['status']}: {item['run_id']} {item.get('error', '')}", flush=True)
            failed = any(item['status'] != 'COMPLETE' for item in state['items'])
            state['status'] = 'COMPLETED_WITH_ERRORS' if failed else 'COMPLETE'
            state.pop('error', None)
            save_queue(path, state)
            return 1 if failed else 0
        except KeyboardInterrupt:
            state.update(status='INTERRUPTED', updated_at=now())
            save_queue(path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+')
    parser.add_argument('--run-list', type=Path)
    parser.add_argument('--queue', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--interval', type=float)
    args = parser.parse_args()
    if args.resume:
        if not args.queue or args.runs or args.run_list or args.interval is not None or args.check_only:
            parser.error('Resume requires only --queue and --resume')
    elif not args.runs and not args.run_list:
        parser.error('Use actual --runs or --run-list; no automatic latest-recording selection')
    path = args.queue or ROOT / 'validation/process_queues' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    if not args.resume:
        runs = list(args.runs or [])
        if args.run_list:
            runs += [line.strip() for line in args.run_list.read_text(encoding='utf-8-sig').splitlines()
                     if line.strip() and not line.strip().startswith('#')]
        with exclusive(path.with_suffix('.lock')):
            state = create_queue(runs, path, SAMPLE_PERIOD_S if args.interval is None else args.interval)
        print(f'Queue: {path.resolve()}', flush=True)
        for item in state['items']:
            print(f"{item['status']}: {item['run_id']} {item.get('error', '')}", flush=True)
        if args.check_only:
            return int(any(item['status'] == 'INVALID' for item in state['items']))
    code = execute(path)
    print(f'Summary: {path.resolve()} (technical processing only; no automatic task-success judgement)', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
