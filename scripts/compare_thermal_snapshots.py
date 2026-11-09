"""Replay trusted local snapshots without Kit; exact comparisons, zero tolerance."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))

from scripts.thermal_snapshot import (load_snapshot, run_numerical, output_comparison,
                                     sensor_history, state_digest, compare_arrays)
from scripts.thermal_diagnostics import statistics


def compare_assembly_results(left, right):
    names = left['planes'].keys() | right['planes'].keys()
    planes = {name: compare_arrays(left['planes'].get(name), right['planes'].get(name))
              for name in sorted(names)}
    output = output_comparison(left['outputs'], right['outputs'])
    history = state_digest(left['history']) == state_digest(right['history'])
    return {'exact': bool(all(p['exact'] for p in planes.values()) and output['exact'] and history),
            'planes': planes, 'outputs': output, 'history_exact': history}


def compare_material_cache_sequence(paths):
    from scripts.thermal_snapshot import run_assembly_checkpoint
    from scripts.thermal_runtime import MaterialLookupCache
    cache = MaterialLookupCache()
    rows, snapshots = [], []
    for path in paths:
        snapshot = load_snapshot(path)
        snapshots.append(snapshot)
        checkpoint = snapshot['assembly_checkpoint']
        if snapshot['metadata'].get('reset_history') or snapshot['metadata'].get('config_changed'):
            cache.clear()
        off = run_assembly_checkpoint(checkpoint, cache_enabled=False)
        on = run_assembly_checkpoint(checkpoint, cache_enabled=True, cache=cache)
        comparison = compare_assembly_results(off, on)
        captured = output_comparison(off['outputs'], snapshot['outputs'])
        plane_names = off['planes'].keys() | snapshot['planes'].keys()
        captured_planes = {name: compare_arrays(off['planes'].get(name), snapshot['planes'].get(name))
                           for name in sorted(plane_names)}
        captured_history = state_digest(off['history']) == state_digest(snapshot['assembly_history_after'])
        rows.append({'path': str(path), 'case': snapshot['metadata']['case'],
                     'comparison': comparison, 'matches_captured': captured,
                     'captured_planes': captured_planes,
                     'captured_history_exact': captured_history,
                     'cache': cache.stats(),
                     'pass': (comparison['exact'] and captured['exact'] and captured_history
                              and all(plane['exact'] for plane in captured_planes.values()))})
    sensitivity = False
    if snapshots:
        checkpoint = deepcopy(snapshots[0]['assembly_checkpoint'])
        # Pick a visible material, deliberately rebind it to a different valid material.
        ids = checkpoint['raw_aovs'].instance_id
        from irsim_isaac.pipeline.material_ids import labels_from_payload
        labels = labels_from_payload(checkpoint['raw_aovs'].device_handles['instance'])
        visible = {labels.get(int(i)) for i in np.unique(ids) if int(i) != 0}
        attrs = checkpoint['attributes']
        from dataclasses import replace
        resolutions = list(attrs['resolutions'])
        for index, resolution in enumerate(resolutions):
            if resolution.path in visible:
                valid_ids = [i for i in range(1, len(attrs['config'].materials.names))
                             if i != resolution.material_id]
                resolutions[index] = replace(resolution, material_id=valid_ids[0])
                break
        else:
            raise ValueError('No visible material for mapping sensitivity check')
        attrs['resolutions'] = resolutions
        wrong = run_assembly_checkpoint(checkpoint, cache_enabled=True, cache=cache)
        reference = run_assembly_checkpoint(snapshots[0]['assembly_checkpoint'], cache_enabled=False)
        sensitivity = not compare_assembly_results(reference, wrong)['exact']
    scenarios = scenario_checks(snapshots)
    passed = bool(rows and all(r['pass'] for r in rows) and sensitivity and all(scenarios.values()))
    return {'result': 'PASS' if passed else 'FAIL', 'frames': rows,
            'scenario_checks': scenarios, 'wrong_mapping_detected': sensitivity,
            'scope': 'Full owned-AOV assembly + vendor numeric + project display; independent checkpoints per frame',
            'full_shutter_drift_chain_covered': all(s['config'].chain is not None for s in snapshots),
            'strict_sync_verified': False, 'exact_tolerance': 0.0}


def scenario_checks(snapshots):
    by_case = {}
    for snapshot in snapshots:
        by_case.setdefault(snapshot['metadata']['case'], []).append(snapshot)
    required = ('static', 'translation', 'rotation', 'occlusion', 'paused', 'resumed',
                'reset', 'instance_label_change', 'config_change', 'after_config_change')
    if any(case not in by_case for case in required):
        return {'all_required_cases_present': False}
    base = by_case['static'][0]
    matrix = lambda s: np.asarray(s['metadata']['camera_world_matrix'])
    from irsim_isaac.pipeline.material_ids import labels_to_paths
    def paths(snapshot):
        return labels_to_paths(snapshot['assembly_evidence']['labels'])
    def visible(s, path):
        ids = s['assembly_evidence']['instance_ids']
        return any(p == path and np.any(ids == ident) for ident, p in paths(s).items())
    reset_state = deepcopy(by_case['reset'][0]['state_before'])
    reset_state.t_s = base['state_before'].t_s
    changed = by_case['config_change'][0]
    return {
        'all_required_cases_present': True,
        'translation_pose_changed': bool(np.max(np.abs(matrix(by_case['translation'][0])[3, :3]-matrix(base)[3, :3])) > .01),
        'rotation_pose_changed': bool(np.max(np.abs(matrix(by_case['rotation'][0])[:3, :3]-matrix(base)[:3, :3])) > .01),
        'occluder_visible': visible(by_case['occlusion'][0], '/World/DiagnosticOccluder'),
        'new_instance_visible': visible(by_case['instance_label_change'][0], '/World/DiagnosticOccluderNew'),
        'paused_time_held': by_case['paused'][0]['state_before'].t_s == by_case['occlusion'][-1]['state_before'].t_s,
        'resumed_time_advanced': by_case['resumed'][0]['state_before'].t_s > by_case['paused'][0]['state_before'].t_s,
        'reset_restores_checkpoint': state_digest(reset_state) == state_digest(base['state_before']),
        'noise_seed_changed': (changed['config'].sensor_seed != base['config'].sensor_seed and
                               changed['config'].noise.sensor_seed != base['config'].noise.sensor_seed),
    }


def compare_sequence(paths, *, frame_function=None, require_scenarios=True):
    frames = []
    snapshots = []
    configs, states = [None, None], [None, None]
    for path in paths:
        snapshot = load_snapshot(path)
        snapshots.append(snapshot)
        meta = snapshot['metadata']
        reset = meta.get('reset_history', False) or states[0] is None
        for branch in range(2):
            if reset:
                configs[branch], states[branch] = deepcopy(snapshot['config']), deepcopy(snapshot['state_before'])
            elif meta.get('config_changed', False):
                configs[branch] = deepcopy(snapshot['config'])
        outputs, timings, histories = [], [], []
        for branch in range(2):
            started = time.perf_counter()
            result, configs[branch], states[branch] = run_numerical(
                snapshot, config=configs[branch], state=states[branch], frame_function=frame_function)
            timings.append(time.perf_counter()-started)
            outputs.append(result)
            histories.append(sensor_history(configs[branch], states[branch]))
        comparison = output_comparison(*outputs)
        captured = output_comparison(snapshot['outputs'], outputs[0])
        exact_history = state_digest(histories[0]) == state_digest(histories[1])
        capture_history = state_digest(histories[0]) == state_digest(snapshot['history_after'])
        evidence = snapshot.get('assembly_evidence', {})
        material = motion = None
        if evidence:
            from irsim_isaac.pipeline.material_ids import material_id_plane
            from scripts.thermal_runtime import material_id_plane_cached
            mapping_args = (evidence['instance_ids'], evidence['labels'], evidence['resolutions'])
            material = compare_arrays(material_id_plane(*mapping_args), material_id_plane_cached(*mapping_args))
            if evidence.get('motion_args') is not None:
                from irsim.optics.motion import image_plane_motion
                from scripts.live_motion_optimization import compact_motion
                motion = compare_arrays(image_plane_motion(*evidence['motion_args']),
                                        compact_motion(*evidence['motion_args']))
        frames.append({'snapshot': str(path), 'case': meta['case'], 'snapshot_id': meta['snapshot_id'],
            'serial_repeat': comparison, 'matches_captured_serial': captured,
            'branch_history_exact': exact_history, 'captured_history_exact': capture_history,
            'existing_material_cache_input_equivalence': material,
            'existing_compact_motion_input_equivalence': motion,
            'reference_s': timings[0], 'repeat_s': timings[1],
            'pass': bool(comparison['exact'] and captured['exact'] and exact_history and capture_history
                         and (material is None or material['exact']) and (motion is None or motion['exact']))})
    scenarios = scenario_checks(snapshots) if require_scenarios else {}
    sensitivity = {}
    if snapshots and require_scenarios:
        changed = deepcopy(snapshots[0])
        changed['planes']['temperature_k'] += np.float32(10.)
        changed_output, _, _ = run_numerical(changed, frame_function=frame_function)
        sensitivity['temperature_change_detected'] = not compare_arrays(
            snapshots[0]['outputs'].radiance, changed_output.radiance)['exact']
        changed_state = deepcopy(snapshots[0]['state_before'])
        changed_state.frame_index += 1
        sensitivity['history_change_detected'] = state_digest(sensor_history(
            snapshots[0]['config'], changed_state)) != state_digest(sensor_history(
            snapshots[0]['config'], snapshots[0]['state_before']))
    passed = frames and all(f['pass'] for f in frames) and all(scenarios.values()) and all(sensitivity.values())
    return {'result': 'PASS' if passed else 'FAIL',
            'scenario_checks': scenarios, 'intentional_change_sensitivity': sensitivity,
            'comparison_mode': 'serial_repeat + captured_serial + existing_mapping_motion',
            'new_optimized_numerical_implementation': False,
            'exact_tolerance': 0., 'frames': frames,
            'reference_s': statistics(f['reference_s'] for f in frames),
            'repeat_s': statistics(f['repeat_s'] for f in frames),
            'scope': 'Post-assembly numerical replay plus isolated mapping/motion input tests; not full assembly or live synchronization certification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--material-cache', action='store_true')
    parser.add_argument('--trusted-local-snapshots', action='store_true', required=True,
                        help='Only use this with snapshots created locally by this project; pickle can execute code')
    args = parser.parse_args()
    paths = sorted(args.snapshots.glob('frame_*.pkl'))
    if not paths:
        parser.error('No snapshot files')
    report = compare_material_cache_sequence(paths) if args.material_cache else compare_sequence(paths)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"THERMAL_EQUIVALENCE_{report['result']} frames={len(report['frames'])} {args.out}", flush=True)
    return 0 if report['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
