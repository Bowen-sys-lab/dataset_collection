"""Combine measured preview performance without promoting live data to benchmark."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(root):
    runs = []
    for name in ('baseline-headless-160-v2', 'live-gui-160-v2', 'live-gui-320-v2',
                 'live-gui-640-v2', 'realtime-live-gui-160', 'live-gui-160-15hz'):
        path = root/name/'performance.json'
        report = read(path)
        text = (root/(name+'.log')).read_text(encoding='utf-8', errors='replace')
        errors = [line for line in text.splitlines() if re.search(
            r'\[Error\]|Out of resource descriptors|AmbientOcclusion.*(?:failed|error)', line, re.I)]
        runs.append({'name': name, 'report_path': str(path.resolve()),
            'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'resolution': report['resolution'], 'renderer': report['renderer'],
            'mode': report['preview_mode'], 'target_hz': report['preview_target_fps'],
            'frames': report['frames'], 'fps': report['paired_fps'],
            'p95_frame_ms': report['p95_frame_ms'], 'p99_frame_ms': report['p99_frame_ms'],
            'dropped_preview_frames': report['dropped_preview_frames'],
            'stage_timings_s': report['stage_timings_s'], 'app_update_s': report['app_update_s'],
            'control_metrics': report['control_metrics'], 'critical_render_errors': errors,
            'explicit_project_render_calls': report['actual_render_calls'],
            'source_usd_hash_matches': all(Path(f['path']).exists() and
                hashlib.sha256(Path(f['path']).read_bytes()).hexdigest() == f['sha256']
                for f in report['experiment_conditions']['files'].values()),
            'source_aov_old_frame_check': 'NOT_VERIFIED for live preview; global dispatcher count only',
            'vram': 'NOT_MEASURED', 'file_write_s': 0.,
            'gui_manual_observation': False})
    proofs = {}
    for name in ('realtime-marker-160', 'offline-marker-640'):
        path = root/name/'summary.json'
        proof = read(path)
        text = (root/(name+'.log')).read_text(encoding='utf-8', errors='replace')
        proofs[name] = {'path': str(path.resolve()), 'result': proof['result'],
            'frames': len(proof['frames']), 'renderer': proof['renderer'],
            'strict_per_frame_pixel_sync_verified': proof['strict_per_frame_pixel_sync_verified'],
            'source_unchanged': proof['source_unchanged'],
            'critical_errors': [s for s in text.splitlines() if '[Error]' in s or 'Out of resource descriptors' in s]}
    return {'benchmark_status': 'pilot/development data', 'runs': runs, 'proofs': proofs,
        'recommended_manual_preview': 'PathTracing, live, 160x128, target 15 Hz, external viewport disabled',
        'recommended_scientific_capture': 'PathTracing, frozen offline replay, 640x512',
        'async_api_audit': 'step_async exists, but wait_for_render explicitly disables timeline auto-update during wait; not used for independent live physics.',
        'control_response_scope': 'Controller source unchanged. No extra explicit-render stall in live mode; thermal calculation still occupies main thread. Not fully asynchronous or hard real-time.',
        'comparison_caveat': 'Historical baseline 4.94 FPS used GUI and explicit rendering. New matched headless baseline is 6.98 FPS; final live GUI is 14.04 FPS. Viewport policy differs; no isolated attribution to one change.',
        'limitations': ['Synthetic latency is not human end-to-end keyboard/display latency.',
            'Live source AOV ages and strict per-pair identities unavailable.',
            'Old lifecycle GUI render failure remains in acceptance history; new manual lifecycle check required.',
            'Four synthetic latency samples and 40-60 preview frames are pilot measurements, not long-run reliability evidence.',
            'High-resolution thermal computation still blocks the Kit main thread; do not drive at 640x512.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    for filename, data in (('performance_comparison.json', report),
        ('control_response.json', {'scope': report['control_response_scope'],
            'runs': [{k: r[k] for k in ('name', 'control_metrics')} for r in report['runs']],
            'human_end_to_end_verified': False}),
        ('renderer_comparison.json', {'proofs': report['proofs'],
            'default_changed': False, 'recommended': 'PathTracing',
            'live_strict_sync_verified': False})):
        (args.root/filename).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'benchmark_status': report['benchmark_status'],
        'fps': {r['name']: r['fps'] for r in report['runs']}}, indent=2))


if __name__ == '__main__':
    main()
