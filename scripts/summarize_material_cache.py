"""Accept material lookup cache only from six comparable, unprofiled processes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.thermal_diagnostics import statistics


ORDER = ('material_cache_off', 'material_cache_on', 'material_cache_on',
         'material_cache_off', 'material_cache_off', 'material_cache_on')
MATCH_KEYS = ('resolution', 'renderer', 'enabled_viewport_count', 'preview_rate_mode',
              'preconversion', 'shared_live_render_product', 'warmup_frames',
              'motion_sequence_sha256', 'motion_baseline')


def summarize(runs, equivalence, *, paths=(), equivalence_path=None):
    errors, rows = [], []
    if len(runs) != 6:
        errors.append('Exactly six independent process reports are required')
    if equivalence.get('result') != 'PASS':
        errors.append('Numeric equivalence did not PASS')
    first = runs[0] if runs else {}
    for index, run in enumerate(runs):
        baseline = run.get('baseline', {})
        candidates = baseline.get('rounds', [])
        if run.get('result') != 'PASS' or not run.get('source_unchanged') or not run.get('vendor_source_unchanged'):
            errors.append(f'Run {index}: integrity/result gate failed')
        before, after = run.get('file_hashes_before'), run.get('file_hashes_after')
        if not before or before != after or before != first.get('file_hashes_before'):
            errors.append(f'Run {index}: scene/config/source/vendor/controller hashes differ')
        if run.get('experiment_conditions') != first.get('experiment_conditions'):
            errors.append(f'Run {index}: experiment conditions differ')
        if any(key not in baseline or baseline[key] != first.get('baseline', {}).get(key) for key in MATCH_KEYS):
            errors.append(f'Run {index}: workload, renderer, viewport or motion differs')
        if baseline.get('resolution') != [160, 128] or baseline.get('renderer') != 'PathTracing' or baseline.get('enabled_viewport_count') != 1 or baseline.get('preview_rate_mode') != 'uncapped' or baseline.get('warmup_frames', 0) < 30:
            errors.append(f'Run {index}: required experiment settings missing')
        if len(candidates) != 1:
            errors.append(f'Run {index}: must contain exactly one throughput round')
            continue
        row = candidates[0]
        if index >= len(ORDER) or row.get('condition') != ORDER[index]:
            errors.append(f'Run {index}: incorrect condition/order')
        if row.get('profiled') is not False or row.get('segments') or row.get('function_hotspots'):
            errors.append(f'Run {index}: profiling was present or unspecified')
        samples = row.get('samples', [])
        if len(samples) != 600:
            errors.append(f'Run {index}: must have exactly 600 measured frames')
        try:
            timings = {key: statistics(s[key] for s in samples) for key in samples[0]}
            if not timings['loop_s']['mean'] or any(s['loop_s'] <= 0 for s in samples):
                raise ValueError('nonpositive duration')
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f'Run {index}: invalid timing samples')
            continue
        rows.append({'run': index, 'condition': row.get('condition'), 'timings': timings,
                     'processing_fps': 1/timings['loop_s']['mean'],
                     'cache_stats_before': row.get('cache_stats_before'),
                     'cache_stats_after': row.get('cache_stats_after'),
                     'cache_stats_delta': row.get('cache_stats_delta'),
                     'process_memory_before': row.get('process_memory_before'),
                     'process_memory_after': row.get('process_memory_after')})
    result = {'result': 'FAIL' if errors else 'PASS', 'retention': False, 'default_enabled': False,
              'errors': errors, 'rounds': rows, 'pairs': [], 'aggregate': {}, 'gates': {},
              'run_directories': [str(p) for p in paths], 'equivalence_report': str(equivalence_path) if equivalence_path else None,
              'strict_sync_verified': False, 'valid_new_pair_fps_verified': False,
              'human_input_latency_verified': False, 'gui_driving_acceptance_verified': False,
              'scope': 'Scene-processing throughput with deterministic commanded poses and identical UI publish',
              'caveats': ['Processing FPS does not prove fresh RGB/thermal pairs or human input latency.',
                          'Cache storage is an approximate Python object size; process working set includes unrelated allocations.',
                          'Inclusive profile stages overlap; only exclusive times avoid nested double counting.']}
    if errors:
        return result
    for pair in range(3):
        a, b = rows[pair*2:pair*2+2]
        off, on = (a, b) if a['condition'] == 'material_cache_off' else (b, a)
        off_loop, on_loop = off['timings']['loop_s'], on['timings']['loop_s']
        result['pairs'].append({'pair': pair+1, 'off_run': off['run'], 'on_run': on['run'],
            'off_loop_s': off_loop, 'on_loop_s': on_loop,
            'off_processing_fps': off['processing_fps'], 'on_processing_fps': on['processing_fps'],
            'mean_loop_reduction': 1-on_loop['mean']/off_loop['mean'],
            'p95_worse': on_loop['p95'] > off_loop['p95'], 'p99_worse': on_loop['p99'] > off_loop['p99']})
    grouped = {condition: [r for r in rows if r['condition'] == condition] for condition in set(ORDER)}
    means = {key: [r['timings']['loop_s']['mean'] for r in value] for key, value in grouped.items()}
    off_mean = sum(means['material_cache_off'])/3
    on_mean = sum(means['material_cache_on'])/3
    gain = off_mean-on_mean
    variation = sum(max(values)-min(values) for values in means.values())/2
    result['aggregate'] = {'off_mean_loop_s': off_mean, 'on_mean_loop_s': on_mean,
        'off_processing_fps': 1/off_mean, 'on_processing_fps': 1/on_mean,
        'mean_loop_reduction': gain/off_mean, 'mean_loop_gain_s': gain,
        'half_combined_round_mean_range_s': variation,
        'approximate_cache_bytes': [r['cache_stats_after'].get('approximate_bytes') if r['cache_stats_after'] else None for r in grouped['material_cache_on']],
        'timings': {key: {metric: statistics(sample[metric] for run in runs if run['baseline']['rounds'][0]['condition'] == key for sample in run['baseline']['rounds'][0]['samples'])
                          for metric in rows[0]['timings']} for key in grouped}}
    p95 = sum(p['p95_worse'] for p in result['pairs'])
    p99 = sum(p['p99_worse'] for p in result['pairs'])
    result['gates'] = {'all_pairs_improve': all(p['mean_loop_reduction'] > 0 for p in result['pairs']),
        'reduction_at_least_three_percent': gain/off_mean >= .03,
        'gain_exceeds_variation': gain > variation, 'p95_worse_pairs': p95, 'p99_worse_pairs': p99,
        'tails_not_consistently_worse': p95 < 2 and p99 < 2}
    result['retention'] = all(result['gates'][key] for key in ('all_pairs_improve', 'reduction_at_least_three_percent', 'gain_exceeds_variation', 'tails_not_consistently_worse'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=6, required=True)
    parser.add_argument('--equivalence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve() for p in args.runs]
    if len(set(paths)) != 6:
        parser.error('Run directories must be distinct independent processes')
    runs = [json.loads((p/'summary.json').read_text(encoding='utf-8')) for p in paths]
    equivalence = json.loads(args.equivalence.read_text(encoding='utf-8'))
    report = summarize(runs, equivalence, paths=paths, equivalence_path=args.equivalence.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'performance.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'result': report['result'], 'retention': report['retention'], 'gates': report['gates'], 'errors': report['errors']}, indent=2))
    return 0 if report['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
