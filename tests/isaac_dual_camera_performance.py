"""Finite GUI performance and moving-camera regression; never saves the source stage."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    if '--all' in sys.argv:
        return benchmark_all()
    from scripts.start_crazyflie_dual_camera_view import open_preview
    from isaacsim import SimulationApp
    width, height = (320, 256) if '--medium' in sys.argv else (160, 128)
    app = SimulationApp({'headless': False, 'renderer': 'PathTracing',
                         'samples_per_pixel_per_frame': 1, 'multi_gpu': False})
    preview = controller = None
    scene = ROOT / 'lesson_crazyflie.usd'
    before = hashlib.sha256(scene.read_bytes()).hexdigest()
    out = ROOT / 'validation' / time.strftime(f'dual-performance-{width}-%Y%m%d-%H%M%S')
    out.mkdir()
    status = 1
    try:
        import omni.timeline
        preview, controller = open_preview(app, width=width, height=height)
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(180):
            app.update()
            if controller._ready:
                break
        assert controller._ready
        origin = controller.position
        yaw = controller.yaw
        stamps, geometries = [], set()
        deadline = time.perf_counter()+150
        for i in range(2200):
            t = timeline.get_current_time()
            controller.position = (origin[0]+.3*math.sin(t), origin[1], origin[2]+.1*math.sin(t*.5))
            controller.yaw = yaw + .15*math.sin(t*.7)
            app.update()
            rgb, thermal, meta = preview.capture(render=False)
            stamps.append(time.perf_counter())
            assert controller._running and controller._ready
            assert np.isfinite(thermal.apparent_t).all()
            assert np.isfinite(thermal.radiance).all()
            assert np.ptp(rgb[..., :3]) > 0
            assert np.ptp(thermal.display8[..., :3]) > 0
            assert meta['pair_pose_max_error'] < 1e-6
            if i % 100 == 0:
                depth = preview.thermal.last_frame.planes['distance_m']
                geometries.add(hashlib.sha256(depth.tobytes()).hexdigest())
            if i in (0, 100, 2199):
                preview.save_frame(out, i, rgb, thermal, meta)
            if i % 500 == 0:
                print(f'PERF_PROGRESS frames={i} simulation={t:.2f}', flush=True)
            if stamps[-1] > deadline:
                raise RuntimeError('Performance test exceeded 150 seconds')
        intervals = np.diff(stamps[30:])
        assert len(geometries) > 15, 'rendered geometry did not follow motion'
        assert meta['thermal_elapsed_s'] > 30, 'sky cache rollover not exercised'
        assert hashlib.sha256(scene.read_bytes()).hexdigest() == before
        result = {'result': 'PASS', 'resolution': [width, height], 'moving': True,
                  'frames': len(stamps), 'paired_fps': float(1/intervals.mean()),
                  'p95_frame_ms': float(np.percentile(intervals, 95)*1000),
                  'p99_frame_ms': float(np.percentile(intervals, 99)*1000),
                  'thermal_elapsed_s': meta['thermal_elapsed_s'],
                  'distinct_geometry_samples': len(geometries), 'source_unchanged': True,
                  'manual_keyboard_test': False, 'timings_s': preview.timing_summary()}
        (out/'performance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('DUAL_PERFORMANCE_PASS ' + json.dumps(result), flush=True)
        status = 0
    finally:
        if controller:
            controller.stop()
        if preview:
            preview.close()
        app.close(exit_code=status)


def benchmark_all():
    """Measure all three capture sizes, including optional file-write cost."""
    from scripts.start_crazyflie_dual_camera_view import open_preview
    from scripts.scene_config import DEFAULT_SCENE_CONFIG, load_scene_config
    from isaacsim import SimulationApp
    import omni.timeline
    config = load_scene_config(DEFAULT_SCENE_CONFIG)
    frames = int(sys.argv[sys.argv.index('--frames') + 1]) if '--frames' in sys.argv else 30
    run_id = time.strftime('%Y%m%d-%H%M%S')
    out_root = ROOT / 'validation' / 'performance' / run_id
    out_root.mkdir(parents=True, exist_ok=False)
    app = SimulationApp({'headless': False, 'renderer': 'PathTracing',
                         'samples_per_pixel_per_frame': 1, 'multi_gpu': False})
    results = []
    try:
        for width, height in ((160, 128), (320, 256), (640, 512)):
            preview = controller = None
            stamps = []
            try:
                preview, controller = open_preview(app, scene_config=config, width=width, height=height,
                                                   fast=True)
                timeline = omni.timeline.get_timeline_interface()
                timeline.play()
                for _ in range(240):
                    app.update()
                    if controller._ready:
                        break
                if not controller._ready:
                    raise RuntimeError(controller.diagnostics())
                output = out_root / f'{width}x{height}'
                output.mkdir()
                for index in range(frames):
                    app.update()
                    started = time.perf_counter()
                    rgb, thermal, meta = preview.capture(render=True)
                    capture_done = time.perf_counter()
                    preview.save_frame(output, index, rgb, thermal, meta)
                    finished = time.perf_counter()
                    meta['file_write_s'] = finished - capture_done
                    stamps.append({**meta, 'timings_s': dict(preview.last_timings),
                                   'total_wall_s': finished - started})
                total = np.array([item['total_wall_s'] for item in stamps])
                result = {
                    'resolution': [width, height], 'frames': frames,
                    'paired_fps': float(1 / total[2:].mean()) if len(total) > 2 else None,
                    'p95_frame_ms': float(np.percentile(total[2:], 95) * 1000) if len(total) > 2 else None,
                    'p99_frame_ms': float(np.percentile(total[2:], 99) * 1000) if len(total) > 2 else None,
                    'timings_s': {name: float(np.mean([item['timings_s'][name] for item in stamps]))
                                  for name in stamps[0].get('timings_s', {})},
                    'file_write_s_mean': float(np.mean([item.get('file_write_s', 0.0) for item in stamps])),
                    'strict_sync_verified': False, 'sync_status': 'strict_not_verified',
                    'source_stage_sha256': hashlib.sha256(config.scene.stage.read_bytes()).hexdigest(),
                }
                (output / 'performance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                results.append(result)
                print('[performance] ' + json.dumps(result), flush=True)
            finally:
                if controller:
                    controller.stop()
                if preview:
                    preview.close()
        report = {'run_id': run_id, 'results': results, 'automated_gui_runtime': True,
                  'manual_keyboard_test': False, 'gui_visual_test': False,
                  'strict_sync_verified': False,
                  'note': 'Explicit render was used, but RGB buffer identity is not yet independently verified.'}
        (out_root / 'performance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return 0
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(main() or 0)
