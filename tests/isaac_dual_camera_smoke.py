"""Real render/physics integration check; NOT a manual-keyboard-focus test."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/isaac-thermal-camera-simulation/src'))


def main():
    from scripts.start_crazyflie_dual_camera_view import open_preview
    from isaacsim import SimulationApp
    import numpy as np
    app = SimulationApp({'headless': '--gui' not in sys.argv, 'renderer': 'PathTracing',
                         'samples_per_pixel_per_frame': 1, 'multi_gpu': False})
    preview = controller = None
    path = ROOT / 'lesson_crazyflie.usd'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    out = ROOT / 'validation' / time.strftime('dual-camera-smoke-%Y%m%d-%H%M%S')
    out.mkdir()
    status = 1
    try:
        import omni.timeline
        import omni.usd
        from pxr import UsdGeom
        from scripts.start_crazyflie_controller import start_crazyflie
        # The bundled Boson preview configuration is 640x512.  Keep both
        # render products on that contract so the smoke test exercises the
        # actual thermal sensor rather than a mismatched display size.
        width, height = (320, 256) if '--small' in sys.argv else (640, 512)
        preview, controller = open_preview(app, stage_path=path, width=width, height=height)
        assert preview.window is not None and preview.window.visible, 'preview missing before Play'
        assert omni.timeline.get_timeline_interface().is_stopped(), 'startup must leave a clean stopped timeline'
        assert preview.thermal.optics.resolution == (width, height), 'preview is unexpectedly supersampled'
        rgb_camera = UsdGeom.Camera(preview.stage.GetPrimAtPath(preview.mount.rgb_path))
        ir_camera = UsdGeom.Camera(preview.stage.GetPrimAtPath(preview.mount.thermal_path))
        for field in ('FocalLength', 'HorizontalAperture', 'VerticalAperture'):
            assert abs(getattr(rgb_camera, 'Get'+field+'Attr')().Get() -
                       getattr(ir_camera, 'Get'+field+'Attr')().Get()) < 1e-5, field
        timeline = omni.timeline.get_timeline_interface()
        stage = omni.usd.get_context().get_stage()
        expected_cameras = {preview.mount.rgb_path, preview.mount.thermal_path}
        def assert_two_cameras():
            cameras = {str(p.GetPath()) for p in stage.Traverse()
                       if str(p.GetPath()).startswith('/World/Crazyflie/body/') and p.IsA(UsdGeom.Camera)}
            assert cameras == expected_cameras, cameras
        assert_two_cameras()
        timeline.play()
        for _ in range(180):
            app.update()
            if controller._ready:
                break
        assert controller._ready, controller.diagnostics()
        samples = []
        seen_frame_ids = set()
        first_rgb = None
        first_distance = None
        for i in range(5):
            if i == 1:
                controller.position = (controller.position[0]+.15, controller.position[1], controller.position[2]+.1)
                controller.yaw += .15
            for _ in range(3):
                app.update()
            before_capture = timeline.get_current_time()
            rgb, thermal, meta = preview.capture(render='--stream' not in sys.argv)
            required_meta = {
                'frame_id', 'simulation_time_before', 'simulation_time_after',
                'rgb_render_finished_wall_s', 'thermal_frame_time_s',
                'camera_world_matrix', 'thermal_camera_world_matrix',
                'pair_pose_max_error', 'simulation_capture_drift_s',
                'explicit_render_triggered', 'strict_sync_verified', 'resolution',
            }
            assert required_meta <= meta.keys(), sorted(required_meta - meta.keys())
            assert meta['frame_id'] not in seen_frame_ids
            seen_frame_ids.add(meta['frame_id'])
            if samples:
                assert meta['simulation_time_s'] >= samples[-1]['simulation_time_s']
            assert meta['explicit_render_triggered'] is ('--stream' not in sys.argv)
            assert meta['strict_sync_verified'] is False
            distance = preview.thermal.last_frame.planes['distance_m']
            if i == 0:
                first_rgb = rgb.copy()
                first_distance = distance.copy()
            if i == 1:
                assert not np.array_equal(first_rgb, rgb), 'RGB stayed stale after camera movement'
                assert not np.array_equal(first_distance, distance), 'thermal geometry stayed stale after camera movement'
            assert timeline.is_playing(), 'capture paused the controller'
            assert abs(timeline.get_current_time() - before_capture) < 1e-6
            assert abs(preview.thermal.last_frame_t_s - preview.thermal.scene.t0_s -
                       meta['thermal_elapsed_s']) < 1e-6, 'thermal model clock is out of sync'
            assert rgb.shape[:2] == (height,width), rgb.shape
            assert thermal.display8.shape[:2] == (height,width)
            assert np.ptp(rgb[...,:3]) > 0, 'black or constant RGB'
            assert np.ptp(thermal.display8[..., :3]) > 0, 'constant thermal display'
            assert np.isfinite(thermal.radiance).all()
            assert np.isfinite(thermal.apparent_t).all()
            assert meta['pair_pose_max_error'] < 1e-6, meta
            assert meta['simulation_capture_drift_s'] < 1e-6, meta
            samples.append(meta)
            assert_two_cameras()
            preview.save_frame(out, i, rgb, thermal, meta)
        assert np.max(np.abs(np.array(samples[1]['camera_world_matrix'])-
                             np.array(samples[0]['camera_world_matrix']))) > .05
        assert samples[-1]['thermal_elapsed_s'] > samples[0]['thermal_elapsed_s']
        timeline.pause()
        app.update()
        frozen_time = timeline.get_current_time()
        preview.capture()
        assert not timeline.is_playing(), 'capture resumed a paused timeline'
        assert abs(timeline.get_current_time()-frozen_time) < 1e-6
        timeline.stop()
        app.update()
        timeline.play()
        for _ in range(180):
            app.update()
            if controller._ready:
                break
        assert controller._ready, controller.diagnostics()
        _, _, replay = preview.capture()
        assert replay['epoch'] == 1, replay
        assert replay['thermal_elapsed_s'] >= samples[-1]['thermal_elapsed_s']
        controller.stop()
        controller = None
        rgb_path = preview.mount.rgb_path
        product_paths = [preview.rgb_product.path, preview.thermal._render_product.path]
        preview.close()
        preview.close()
        assert stage.GetPrimAtPath(rgb_path), 'existing RGB camera was removed by cleanup'
        for product_path in product_paths:
            assert not stage.GetPrimAtPath(product_path), f'render product leaked: {product_path}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        report = {'result':'PASS', 'samples':samples, 'replay':replay, 'source_sha256':before,
                  'manual_keyboard_test':False, 'gui_visual_test':False,
                  'source_unchanged':True, 'runtime_cleanup':True,
                  'active_body_cameras': sorted(expected_cameras)}
        (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('DUAL_CAMERA_SMOKE_PASS ' + str(out), flush=True)
        status = 0
    except Exception:
        import traceback
        (out/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
    finally:
        if controller:
            controller.stop()
        if preview:
            preview.close()
        app.close(exit_code=status)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
