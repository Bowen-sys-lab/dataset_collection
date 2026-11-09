"""Combine separate automated and human evidence; never promote incomplete runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_report(args):
    sync = read(args.sync)
    replay = read(args.replay)
    gui = read(args.gui)
    required_gui = ('forward', 'lateral', 'vertical', 'yaw', 'paired_updates',
                    'pause_resume', 'no_visible_stale_frames', 'camera_follow')
    gui_pass = gui.get('result') == 'PASS' and all(gui.get('checks', {}).get(k) is True for k in required_gui)
    errors = {}
    for name, path in (('sync', args.sync_log), ('replay', args.replay_log), ('gui', args.gui_log)):
        text = path.read_text(encoding='utf-8', errors='replace')
        errors[name] = [line for line in text.splitlines() if re.search(
            r'\[Error\]|Out of resource descriptors|corrupted input render|step skipped', line, re.I)]
    conditions = replay.get('experiment_conditions', {})
    files = conditions.get('files', {})
    hashes_present = all(files.get(k, {}).get('sha256') for k in
                         ('scene_config', 'thermal_config', 'sensor_config', 'usd', 'thermal_environment_config'))
    hashes_current = bool(hashes_present) and all(
        Path(v['path']).is_file() and hashlib.sha256(Path(v['path']).read_bytes()).hexdigest() == v['sha256']
        for v in files.values())
    probe_files = sync.get('experiment_conditions', {}).get('files', {})
    configs_match = bool(files) and files == probe_files
    probe_conditions = sync.get('experiment_conditions', {})
    runtime_keys = ('resolution', 'flat_field_enabled', 'sensor_seed', 'ao_aov_enabled', 'default_occlusion')
    runtime_match = all(key in conditions and key in probe_conditions and
                        conditions[key] == probe_conditions[key] for key in runtime_keys)
    sync_pass = (sync.get('result') == 'PASS' and
                 sync.get('strict_per_frame_pixel_sync_verified') is True and
                 sync.get('source_unchanged') is True and not sync.get('failed_frames') and
                 bool(sync.get('frames')) and all(f.get('pass') is True for f in sync['frames']))
    replay_pass = (replay.get('result') == 'PASS_FILE_AND_ARRAY_CHECKS' and not replay.get('failures') and
                   replay.get('source_usd_unchanged') is True and
                   replay.get('frames_validated', 0) > 0)
    gates = {'no_key_render_errors': not any(errors.values()),
             'strict_marker_replay_sync': bool(sync_pass),
             'normal_replay_arrays_and_files': bool(replay_pass),
             'gui_human_observation': bool(gui_pass),
             'experiment_hashes_present_current_and_matching': bool(hashes_current and configs_match),
             'capture_conditions_matching': runtime_match}
    ready = all(gates.values())
    report = {'benchmark_status': 'benchmark-ready' if ready else 'pilot/development data',
              'scope': 'Camera acquisition workflow for these exact configs; not instrument calibration or low-light qualification.',
              'gates': gates, 'blocking_items': [k for k, v in gates.items() if not v],
              'strict_per_frame_pixel_sync_verified': bool(sync_pass),
              'pixel_sync_scope': 'Separate marker diagnostic replay, not an assertion that unmarked dataset frames contain pixel witnesses.',
              'marker_probe_frames': len(sync.get('frames', [])),
              'normal_replay_frames': replay.get('frames_validated'),
              'experiment_conditions': conditions, 'gui': gui,
              'render_errors': errors,
              'evidence': {k: str(getattr(args, k).resolve()) for k in
                           ('sync', 'replay', 'sync_log', 'replay_log', 'gui_log', 'gui')}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sync', 'replay', 'sync-log', 'replay-log', 'gui-log', 'gui', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'benchmark_status': report['benchmark_status'], 'gates': report['gates']}, indent=2))


if __name__ == '__main__':
    main()
