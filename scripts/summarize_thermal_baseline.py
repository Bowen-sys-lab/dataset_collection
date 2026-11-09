"""Summarize already measured diagnostic rounds; never infer sync or GUI acceptance."""
import argparse
import json
from pathlib import Path

from scripts.thermal_diagnostics import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads((p/'summary.json').read_text(encoding='utf-8')) for p in args.runs]
    groups, segments = {}, {}
    for run in runs:
        for row in run['baseline']['rounds']:
            groups.setdefault(row['condition'], []).extend(row['samples'])
            for name, values in row['segments'].items():
                segments.setdefault(name, []).append(values)
    conditions = {}
    for name, samples in groups.items():
        durations = [s['loop_s'] for s in samples]
        conditions[name] = {'samples': len(samples), 'processing_fps': 1/statistics(durations)['mean'],
                            'timings': {key: statistics(s[key] for s in samples) for key in samples[0]}}
    profile = {name: {'exclusive_wall_mean_s': statistics(
        row['exclusive_wall_s']['mean'] for row in rows)['mean'],
        'inclusive_wall_mean_s': statistics(row['inclusive_wall_s']['mean'] for row in rows)['mean'],
        'total_calls': sum(row['calls']['total'] for row in rows)} for name, rows in segments.items()}
    snapshot = json.loads((args.snapshots/'summary.json').read_text(encoding='utf-8'))
    comparison = json.loads((args.snapshots/'comparison.json').read_text(encoding='utf-8'))
    report = {'result': 'PASS' if all(r['result'] == 'PASS' for r in runs+[snapshot]) else 'FAIL',
              'scope': 'Stages 1 and 2 diagnostics only', 'data_usage': 'diagnostic-only',
              'strict_sync_verified': False, 'valid_new_pair_fps_verified': False,
              'gui_driving_acceptance_verified': False, 'conditions': conditions,
              'profile_segments': profile, 'comparison': comparison,
              'run_directories': [str(p.resolve()) for p in args.runs],
              'snapshot_directory': str(args.snapshots.resolve()),
              'experiment_conditions': runs[0]['experiment_conditions'],
              'source_unchanged': all(r['source_unchanged'] for r in runs+[snapshot]),
              'vendor_source_unchanged': all(r['vendor_source_unchanged'] for r in runs+[snapshot]),
              'external_viewport_count': [r['baseline'].get('enabled_viewport_count') for r in runs],
              'thread_cpu_clock_note': 'Windows thread_time is quantized at about 15.625 ms; short segment CPU times are coarse.',
              'profiler_note': 'Function cProfile includes warm-up; use unprofiled rounds for throughput.'}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'performance.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({name: row['processing_fps'] for name, row in conditions.items()}, indent=2))
    return 0 if report['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
