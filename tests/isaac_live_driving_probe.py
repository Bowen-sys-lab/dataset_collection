"""Live main-thread driving/marker diagnostic; never benchmark data."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
sys.path.insert(0, str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/school_vln_scene.yaml')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=('A', 'B', 'C', 'D'), required=True)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--marker', action='store_true')
    parser.add_argument('--preview-hz', type=float, default=15.)
    parser.add_argument('--skip-preconversion', action='store_true', help='avoid live UI conversion that is repeated at publish')
    parser.add_argument('--renderer', choices=('PathTracing', 'RealTimePathTracing'), default='PathTracing')
    parser.add_argument('--marker-motion', choices=('continuous', 'jump', 'hold'), default='continuous')
    parser.add_argument('--shared-render-product', action='store_true')
    parser.add_argument('--disable-material-cache', action='store_true')
    parser.add_argument('--enable-material-cache', action='store_true')
    parser.add_argument('--compact-motion', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import yaml
    from scripts.scene_config import load_scene_config
    from scripts.preview_scheduler import PreviewScheduler
    from scripts.preview_metrics import ControlMetrics, render_dispatcher_state
    from scripts.live_frame_validation import LiveFrameValidator, latency_summary, driving_gates
    config = load_scene_config(args.config)
    thermal_on = args.case != 'C'
    upload_on = args.case != 'D'
    viewport_on = args.case != 'A'
    before = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
    report = {'case': args.case, 'marker_probe': args.marker, 'external_viewport_enabled': viewport_on,
        'thermal_enabled': thermal_on, 'image_upload_enabled': upload_on,
        'strict_sync_verified': False, 'preview_sync_status': 'live_preview_not_strict',
        'benchmark_status': 'pilot/development data', 'frames': [], 'result': 'INCOMPLETE',
        'source_usd_sha256_before': before, 'skip_preconversion': args.skip_preconversion,
        'human_end_to_end_verified': False, 'marker_motion': args.marker_motion,
        'renderer': args.renderer, 'resolution': [160, 128],
        'preview_rate_hz': args.preview_hz,
        'preview_rate_mode': 'uncapped' if args.preview_hz == 0 else 'capped'}
    report['shared_render_product'] = args.shared_render_product
    app = preview = controller = metrics = None
    callback = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': False, 'renderer': args.renderer, 'multi_gpu': False,
                             'samples_per_pixel_per_frame': 1})
        import omni.usd
        import omni.timeline
        from pxr import Sdf, UsdGeom
        context = omni.usd.get_context()
        context.open_stage(str(config.scene.stage))
        for _ in range(180):
            app.update()
            if not any(context.get_stage_loading_status()):
                break
        stage = context.get_stage()
        thermal_path = config.sensors.thermal_config
        marker = backdrop = None
        if args.marker:
            from isaac_dual_camera_sync import sheet, matrix, set_world
            probe_layer = Sdf.Layer.CreateAnonymous('live_marker_probe.usda')
            stage.GetSessionLayer().subLayerPaths.insert(0, probe_layer.identifier)
            stage.SetEditTarget(probe_layer)
            marker = sheet(stage, '/World/LiveSyncMarker', .6, .45, (0., 1., 0.))
            backdrop = sheet(stage, '/World/LiveSyncBackdrop', 20., 20., (.01, .01, .01))
            camera = stage.GetPrimAtPath(config.drone.camera_path)
            def place(prim, xyz):
                rigid = matrix(camera).copy()
                rigid[:3, :3] /= np.linalg.norm(rigid[:3, :3], axis=1)[:, None]
                local = np.eye(4)
                local[3, :3] = xyz
                set_world(prim, local @ rigid)
            place(marker, (0, 0, -4))
            place(backdrop, (0, 0, -5))
            data = yaml.safe_load(thermal_path.read_text(encoding='utf-8'))
            data['targets'].update(live_hot={'mode': 'fixed', 'temperature_c': 80.},
                                   live_cold={'mode': 'fixed', 'temperature_c': 10.})
            data['bindings'] += [
                {'name': 'live_marker', 'prim_path': '/World/LiveSyncMarker', 'include_children': True,
                 'target': 'live_hot', 'material': 'concrete'},
                {'name': 'live_backdrop', 'prim_path': '/World/LiveSyncBackdrop', 'include_children': True,
                 'target': 'live_cold', 'material': 'concrete'}]
            thermal_path = args.out/'probe_thermal.yaml'
            thermal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
        from scripts.start_crazyflie_dual_camera_view import open_preview
        preview, controller = open_preview(app, scene_config=config, width=160, height=128,
            thermal_config_path=thermal_path, renderer=args.renderer,
            disable_external_viewport=not viewport_on,
            shared_live_render_product=args.shared_render_product,
            material_cache_enabled=args.enable_material_cache and not args.disable_material_cache)
        preview.thermal.compact_motion_enabled = args.compact_motion
        report['material_cache_enabled'] = args.enable_material_cache and not args.disable_material_cache
        report['compact_motion_enabled'] = args.compact_motion
        preview.skip_live_preconversion = args.skip_preconversion
        report['experiment_conditions'] = preview.metadata['experiment_conditions']
        if not thermal_on:
            # Stop thermal AOV rendering too, not just CPU calculation. Do not remove scene geometry.
            preview.thermal._reader.detach()
            preview.thermal._render_product.hydra_texture.set_updates_enabled(False)
        metrics = ControlMetrics(controller)
        controller._has_control_focus = lambda: True
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        # Hold each marker position for several physics steps.  This gives a
        # renderer with a few-frame queue enough time to produce the current
        # state; the validator still rejects stale pairs.
        sequence = [-.9, .9, -.35, .6, -.65, .25, .75, -.8]
        marker_tick = [0]
        marker_hold = [0]
        marker_history = []
        if args.marker:
            from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
            def move_marker(dt, context):
                x = (-.9 + .2*(marker_tick[0] % 10) if args.marker_motion == 'continuous'
                     else sequence[marker_tick[0] % len(sequence)])
                place(marker, (x, 0, -4))
                place(backdrop, (0, 0, -5))
                from scripts.sync_validation_support import expected_uv
                c = UsdGeom.Camera(camera)
                marker_history.append({'physics_step': len(marker_history),
                    'wall_s': time.perf_counter(),
                    'simulation_time_s': float(timeline.get_current_time()),
                    'camera_world_matrix': matrix(camera).tolist(),
                    'expected_uv': expected_uv(matrix(marker)[3, :3], matrix(camera), 160, 128,
                        c.GetFocalLengthAttr().Get(), c.GetHorizontalApertureAttr().Get())})
                marker_hold[0] += 1
                if marker_hold[0] >= (12 if args.marker_motion == 'hold' else 1):
                    marker_hold[0] = 0
                    marker_tick[0] += 1
            callback = SimulationManager.register_callback(move_marker,
                event=SimulationEvent.PHYSICS_POST_STEP, order=200)
        scheduler = PreviewScheduler(args.preview_hz, 1.)
        validator = LiveFrameValidator()
        capture_times = []
        fresh_times = []
        update_times = []
        loop_times = []
        start = time.perf_counter()
        end_deadline = start + 90.
        last_probe = start
        probe_key = None
        probe_index = 0
        last_loop = start
        attempts = 0
        while app.is_running() and attempts < args.frames and time.perf_counter() < end_deadline:
            tick = time.perf_counter()
            loop_times.append(tick-last_loop)
            last_loop = tick
            if probe_key and tick-last_probe >= .35:
                controller.keys.release(probe_key)
                probe_key = None
            if (controller._ready and probe_key is None and tick-last_probe >= 1.5 and
                    max(abs(v) for v in controller.velocity) < 1e-6):
                probe_key = ('W', 'A', 'R', 'Q')[probe_index % 4]
                metrics.note_input('synthetic_before_app_update')
                controller.keys.press(probe_key)
                last_probe = tick
                probe_index += 1
            t0 = time.perf_counter()
            app.update()
            update_times.append(time.perf_counter()-t0)
            if not controller._ready or not scheduler.due(time.perf_counter()):
                continue
            t0 = time.perf_counter()
            source = render_dispatcher_state()
            if not scheduler.begin(t0, source['compute_count']):
                continue
            thermal = None
            if thermal_on:
                rgb, thermal, meta = preview.capture(render=False, publish=False)
                timings = dict(preview.last_timings)
            else:
                rgb_start = time.perf_counter()
                rgb = np.asarray(preview.rgb_annotator.get_data())
                timings = {'thermal_get_outputs_s': 0., 'rgb_read_s': time.perf_counter()-rgb_start,
                           'render_wait_s': 0., 'ui_upload_s': 0.}
                meta = {'simulation_time_s': float(timeline.get_current_time())}
            sample = {'attempt_id': attempts, 'simulation_time_s': float(timeline.get_current_time()),
                'capture_started_wall_s': t0, 'source_dispatcher': source,
                'rgb_camera_world_matrix': np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(
                    preview.mount.rgb_path)).ComputeLocalToWorldTransform(0)).tolist(),
                'thermal_camera_world_matrix': np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(
                    preview.mount.thermal_path)).ComputeLocalToWorldTransform(0)).tolist()}
            sample['pair_pose_max_error'] = float(np.max(np.abs(
                np.asarray(sample['rgb_camera_world_matrix'])-np.asarray(sample['thermal_camera_world_matrix']))))
            valid = True
            if args.marker:
                from scripts.sync_validation_support import expected_uv
                rgb_cam = UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.rgb_path))
                expected = expected_uv(matrix(marker)[3, :3], matrix(camera), 160, 128,
                    rgb_cam.GetFocalLengthAttr().Get(), rgb_cam.GetHorizontalApertureAttr().Get())
                geometry_mask = None
                if thermal_on:
                    from irsim_isaac.pipeline.material_ids import labels_to_paths
                    frame = preview.thermal.last_frame
                    ids = [i for i, p in labels_to_paths(frame.labels).items() if str(p) == '/World/LiveSyncMarker']
                    geometry_mask = np.isin(frame.instance_id, ids)
                sample.update(validator.inspect(rgb, thermal, geometry_mask, expected),
                    marker_tick=marker_tick[0], marker_world_matrix=matrix(marker).tolist(),
                    thermal_solver_time_s=preview.thermal.last_frame_t_s if thermal_on else None)
                # Historical marker positions independently expose a queued old
                # render. This is diagnostic evidence, never an internal AOV ID.
                for modality in ('rgb', 'thermal'):
                    uv = sample.get(modality+'_marker_uv')
                    matches = [h for h in marker_history[-4:] if uv is not None and
                        np.linalg.norm(np.asarray(uv)-h['expected_uv']) <= 2.]
                    matched = matches[-1] if matches else None
                    sample[modality+'_marker_matched_physics_step'] = None if matched is None else matched['physics_step']
                    sample[modality+'_observed_lag_physics_steps'] = (None if matched is None else
                        marker_history[-1]['physics_step']-matched['physics_step'])
                    sample[modality+'_matched_marker_simulation_time_s'] = None if matched is None else matched['simulation_time_s']
                from scripts.live_frame_validation import match_pair_history
                sample.update(match_pair_history(sample['rgb_marker_uv'], sample['thermal_marker_uv'],
                    sample['geometry_marker_uv'], marker_history[-4:], time.perf_counter()))
                sample['thermal_inputs_historical_state_verified'] = False
                sample['thermal_pose_time_note'] = ('Thermal solver still uses CPU request pose/time; '
                    'a common historical marker match alone does not validate thermal pose, temperature or sensor time')
                valid = sample['fresh_current_state_verified']
            else:
                from scripts.live_frame_validation import image_hash
                sample.update(rgb_sha256=image_hash(rgb), thermal_sha256=image_hash(thermal.display8) if thermal_on else None,
                    freshness_status='NOT_VERIFIED_without_moving_marker')
            frame_id = scheduler.finish(t0, time.perf_counter(), source['compute_count'], valid=valid)
            # Rejected frames never replace the displayed image and do not contribute to effective FPS.
            if frame_id is not None:
                if args.marker:
                    fresh_times.append(time.perf_counter())
                if upload_on:
                    upload_start = time.perf_counter()
                    from scripts.dual_camera_support import ui_rgba_array
                    for provider, pixels in ((preview.rgb_provider, rgb),
                        (preview.thermal_provider, thermal.display8 if thermal_on else None)):
                        if pixels is not None:
                            data, size = ui_rgba_array(pixels)
                            provider.set_data_array(data, size, strict=True)
                    timings['ui_upload_s'] = time.perf_counter()-upload_start
            capture_times.append(time.perf_counter())
            sample.update(timings_s=timings, published_frame_id=frame_id,
                          capture_total_s=time.perf_counter()-t0)
            report['frames'].append(sample)
            if args.marker and attempts < 6:
                from PIL import Image
                Image.fromarray(rgb).save(args.out/f'attempt_{attempts:03d}_rgb.png')
                if thermal_on:
                    Image.fromarray(thermal.display8).save(args.out/f'attempt_{attempts:03d}_thermal.png')
            attempts += 1
        gaps = np.diff(capture_times[5:]).tolist()
        controls = metrics.report()
        latency = [s['seconds'] for s in controls['input_latency_samples']]
        report.update(result='MEASURED', attempts=attempts, duration_s=time.perf_counter()-start,
            processing_fps=1./np.mean(gaps) if gaps else None,
            valid_pair_fps=((len(fresh_times)-1)/(capture_times[-1]-capture_times[0]) if len(fresh_times)>1 else 0.) if args.marker else None,
            current_state_pairs=validator.accepted if args.marker else None,
            stale_or_unverified_pairs=validator.stale if args.marker else None,
            same_historical_marker_pairs=sum(s.get('same_historical_marker_state_verified', False)
                                            for s in report['frames']) if args.marker else None,
            marker_edit_to_read_s=latency_summary([s['marker_edit_to_read_ms']/1000.
                for s in report['frames'] if s.get('marker_edit_to_read_ms') is not None]),
            frame_interval_s=latency_summary(gaps), app_update_s=latency_summary(update_times[5:]),
            loop_interval_s=latency_summary(loop_times[5:]), controller_metrics=controls,
            input_latency_s=latency_summary(latency), dropped_preview_slots=scheduler.dropped,
            timings_s={k: latency_summary([r['timings_s'][k] for r in report['frames'][5:]])
                       for k in report['frames'][0]['timings_s']} if report['frames'] else {})
        gates = driving_gates(viewport_enabled=viewport_on, latencies=latency,
            loop_intervals=loop_times[5:], samples=report['frames'] if args.marker else [], has_thermal=thermal_on)
        report.update(acceptance_gates=gates, realtime_driving_accepted=all(gates.values()))
    except Exception:
        report.update(result='ERROR', failure=traceback.format_exc())
        (args.out/'failure.txt').write_text(report['failure'], encoding='utf-8')
        print(report['failure'], flush=True)
    finally:
        report['source_unchanged'] = before == hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
        (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        if callback is not None:
            SimulationManager.deregister_callback(callback)
        if metrics:
            metrics.close()
        if controller:
            controller.stop()
        if preview:
            preview.close()
        if app:
            app.close()
    return 1 if report['result'] == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
