"""Aggregate measured diagnostics without treating processing FPS as fresh FPS."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def summarize_run(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    log = path.parent.parent / (path.parent.name + '.log')
    lines = log.read_text(encoding='utf-8', errors='replace').splitlines() if log.exists() else []
    frames = report.get('frames', [])
    def lag_counts(modality):
        return dict(Counter(str(f.get(modality+'_observed_lag_physics_steps')) for f in frames))
    errors = [line for line in lines if re.search(
        r'Out of resource descriptors|\[Error\].*(?:rtx|hydra|AmbientOcclusion)|'
        r'AmbientOcclusion.*(?:failed|error)', line, re.I)]
    return {
        'name': path.parent.name, 'report_path': str(path.resolve()),
        'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'result': report.get('result'), 'case': report.get('case'),
        'renderer': report.get('renderer'), 'resolution': report.get('resolution'),
        'external_viewport_enabled': report.get('external_viewport_enabled'),
        'thermal_enabled': report.get('thermal_enabled'),
        'image_upload_enabled': report.get('image_upload_enabled'),
        'shared_render_product': report.get('shared_render_product', False),
        'marker_motion': report.get('marker_motion'),
        'processing_fps': report.get('processing_fps'),
        'current_state_verified_pair_fps': report.get('valid_pair_fps'),
        'current_state_pairs': report.get('current_state_pairs'),
        'stale_or_unverified_pairs': report.get('stale_or_unverified_pairs'),
        'rgb_observed_lag_physics_steps_histogram': lag_counts('rgb'),
        'thermal_observed_lag_physics_steps_histogram': lag_counts('thermal'),
        'rgb_repeated_hash_count': sum(f.get('rgb_hash_repeated', False) for f in frames),
        'thermal_repeated_hash_count': sum(f.get('thermal_hash_repeated', False) for f in frames),
        'internal_aov_frame_identity': 'NOT_AVAILABLE; historical marker match is independent evidence',
        'frame_interval_s': report.get('frame_interval_s'),
        'loop_interval_s': report.get('loop_interval_s'),
        'app_update_s': report.get('app_update_s'),
        'timings_s': report.get('timings_s'),
        'synthetic_input_latency_s': report.get('input_latency_s'),
        'controller_metrics': report.get('controller_metrics'),
        'latency_scope': 'Synthetic input immediately before app.update; excludes OS/event queue and display. NOT human latency.',
        'dropped_preview_slots': report.get('dropped_preview_slots'),
        'max_pair_pose_error': max((f.get('pair_pose_max_error', 0.) for f in frames), default=None),
        'source_unchanged': report.get('source_unchanged'),
        'experiment_conditions': report.get('experiment_conditions'),
        'critical_render_errors': errors,
        'all_error_lines': [line for line in lines if '[Error]' in line],
        'failure': report.get('failure'),
        'realtime_driving_accepted': report.get('realtime_driving_accepted', False),
        'strict_per_frame_pixel_sync_verified': report.get('strict_per_frame_pixel_sync_verified', False),
        'offline_frame_count': len(frames) if 'offline' in path.parent.name else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preview-report', type=Path, action='append', default=[])
    args = parser.parse_args()
    runs = [summarize_run(p) for root in args.root for p in sorted(root.glob('*/summary.json'))]
    previews = [{
        'path': str(path.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'report': json.loads(path.read_text(encoding='utf-8'))
    } for path in args.preview_report]
    result = {'realtime_driving_accepted': False, 'benchmark_status': 'pilot/development data',
        'runs': runs,
        'actual_launcher_checks': previews,
        'conclusions': {
            'processing_fps_is_not_verified_fresh_pair_fps': True,
            'human_end_to_end_latency_verified': False,
            'ordinary_worker_thread_for_get_outputs': 'UNSAFE: reads Kit/USD and owns mutable sensor state; not used',
            'offline_mode_unchanged': True,
            'vram': 'NOT_MEASURED',
            'frame_freeze_detection': 'Continuous marker checks image content; updating delayed frames are not classified as permanently frozen',
            'hold_probe_freshness': 'INCONCLUSIVE when marker stays stationary; do not report as effective fresh FPS',
            'measurement_limitations': ['100 sampled frames per final run; not a long-run stability certificate',
                'Synthetic inputs are injected just before app.update, not at random times during blocking work',
                'First five timing samples excluded; first six marker PNG writes may still influence one interval',
                'Live pixel direction is not proven current-state-consistent when no current-state pair passes'],
        }}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'performance.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'runs': len(runs), 'report': str(args.out/'performance.json'),
                      'realtime_driving_accepted': False}))


if __name__ == '__main__':
    main()
