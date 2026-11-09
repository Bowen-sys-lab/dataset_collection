"""Frozen-state synchronization probe; only anonymous USD edits, no controller.

PASS requires independently detected RGB marker, IR geometry and apparent
temperature to match each requested snapshot. Renderer internal IDs unavailable.
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def matrix(prim):
    import numpy as np
    from pxr import UsdGeom
    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0), dtype=float)


def set_world(prim, world):
    import numpy as np
    from pxr import Gf, UsdGeom
    local = world @ np.linalg.inv(matrix(prim.GetParent()))
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(local.tolist()))


def sheet(stage, path, width, height, color):
    from pxr import UsdGeom, UsdShade, Sdf
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(-width/2, -height/2, 0), (width/2, -height/2, 0),
                           (width/2, height/2, 0), (-width/2, height/2, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([color])
    material = UsdShade.Material.Define(stage, path+'/Material')
    shader = UsdShade.Shader.Define(stage, path+'/Material/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh.GetPrim()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/school_vln_scene.yaml')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--width', type=int, default=512, help='RGB width')
    parser.add_argument('--height', type=int, default=410, help='RGB height')
    parser.add_argument('--thermal-width', type=int, default=None)
    parser.add_argument('--thermal-height', type=int, default=None)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--renderer', choices=('PathTracing', 'RealTimePathTracing'), default='PathTracing')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--trajectory', type=Path,
                        help='restore actual recorded poses/times; outputs contain diagnostic markers, NOT dataset frames')
    parser.add_argument('--sample-interval', type=float, default=.2)
    args = parser.parse_args(argv)
    if args.frames < 4 or min(args.width, args.height) < 32:
        parser.error('at least four frames and RGB resolution >=32 required')
    args.thermal_width = args.thermal_width or max(32, args.width // 2)
    args.thermal_height = args.thermal_height or max(32, args.height // 2)
    if min(args.thermal_width, args.thermal_height) < 32:
        parser.error('thermal resolution must be >=32')
    import numpy as np
    import yaml
    from scripts.scene_config import load_scene_config
    from scripts.sync_validation_support import (centroid, compare_paired_masks, expected_uv,
        compare_recorded_pose, largest_component, match_marker_position, registered_overlay,
        rgb_marker_mask, unique_marker_positions)
    config = load_scene_config(args.config)
    records = None
    if args.trajectory:
        from scripts.trajectory_recorder import read_trajectory
        from scripts.replay_trajectory_highres import _sample
        records = _sample(read_trajectory(args.trajectory), args.sample_interval)[:args.frames]
        args.frames = len(records)
        if args.frames < 4:
            parser.error('trajectory probe requires at least four sampled records')
    try:
        positions = unique_marker_positions(args.frames, args.thermal_width, args.thermal_height)
    except ValueError as exc:
        parser.error(str(exc))
    out = (args.out or ROOT/'validation/dual-camera-sync'/time.strftime('%Y%m%d-%H%M%S')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
    report = {'result': 'INCOMPLETE', 'phase': 'starting', 'strict_sync_verified': False,
              'gui_visual_test': False, 'scene_config_path': str(config.source_path),
              'scene_config_sha256': config.sha256, 'source_usd_sha256_before': before,
              'resolution': [args.width, args.height],
              'rgb_resolution': [args.width, args.height],
              'thermal_resolution': [args.thermal_width, args.thermal_height],
              'frames': [], 'renderer': args.renderer,
              'scope': 'frozen restored states with independent marker pixels',
              'strict_per_frame_pixel_sync_verified': False,
              'data_usage': 'diagnostic-only moving-marker probe; not dataset frames',
              'trajectory': str(args.trajectory.resolve()) if args.trajectory else None,
              'trajectory_sha256': hashlib.sha256(args.trajectory.read_bytes()).hexdigest() if args.trajectory else None,
              'render_internal_frame_identity': 'NOT_AVAILABLE'}
    def checkpoint():
        write_json(out/'summary.json', report)
    checkpoint()
    app = preview = stage = layer = old_edit = None
    original_step = None
    status = 1
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': not args.gui, 'renderer': args.renderer,
                             'samples_per_pixel_per_frame': 1, 'multi_gpu': False})
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf
        context = omni.usd.get_context()
        report.update(simulation_app_started=True, usd_context_available=context is not None,
                      phase='loading_stage')
        checkpoint()
        if not context.open_stage(str(config.scene.stage)):
            raise RuntimeError('stage open failed')
        for _ in range(180):
            app.update()
            if not any(context.get_stage_loading_status()):
                break
        stage = context.get_stage()
        stage_meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        drone = stage.GetPrimAtPath(config.drone.prim_path)
        body = stage.GetPrimAtPath(config.drone.body_path)
        camera = stage.GetPrimAtPath(config.drone.camera_path)
        from scripts.sync_validation_support import diagnostic_target_path
        spot_path = diagnostic_target_path(config.task)
        spot = stage.GetPrimAtPath(spot_path)
        def geometry_count(prim):
            if not prim or not prim.IsValid():
                return 0
            return sum(p.IsA(UsdGeom.Gprim) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        requirements = {'drone_valid': bool(drone and drone.IsActive()),
                        'body_xform_loaded': bool(body and body.IsA(UsdGeom.Xform)),
                        'rgb_camera_valid': bool(camera and camera.IsA(UsdGeom.Camera)),
                        'drone_geometry_count': geometry_count(drone),
                        'spot_geometry_count': geometry_count(spot)}
        errors = [str(e) for e in stage.GetCompositionErrors()]
        thermal_prim = stage.GetPrimAtPath(config.drone.body_path+'/Sensors/ThermalCamera')
        report['preflight'] = {**requirements, 'composition_errors': errors,
            'spot_path': spot_path, 'spot_visible_in_sensor': 'NOT_VERIFIED',
            'world_matrix': matrix(stage.GetPrimAtPath('/World')).tolist(),
            'drone_world_matrix': matrix(drone).tolist() if drone else None,
            'saved_pair_pose_error': float(np.max(np.abs(matrix(camera)-matrix(thermal_prim))))
                if camera and thermal_prim else None}
        checkpoint()
        if not all(requirements.values()) or errors:
            raise RuntimeError('preflight failed: required assets/prim types missing; see summary.json')
        if args.preflight_only:
            report.update(result='PREFLIGHT_PASS', phase='complete')
            status = 0
            return status
        import faulthandler
        faulthandler.dump_traceback_later(180, repeat=True)
        old_edit = stage.GetEditTarget()
        layer = Sdf.Layer.CreateAnonymous('sync_probe.usda')
        stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
        stage.SetEditTarget(layer)
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
        marker_path, backdrop_path = '/World/SyncMarker', '/World/SyncBackdrop'
        if stage.GetPrimAtPath(marker_path) or stage.GetPrimAtPath(backdrop_path):
            raise RuntimeError('probe names already exist; refusing to overwrite')
        marker = sheet(stage, marker_path, .6, .45, (0., 1., 0.))
        backdrop = sheet(stage, backdrop_path, 20., 20., (.01, .01, .01))
        def place(prim, xyz):
            camera_rigid = matrix(camera).copy()
            camera_rigid[:3, :3] /= np.linalg.norm(camera_rigid[:3, :3], axis=1)[:, None]
            local = np.eye(4)
            local[3, :3] = xyz
            set_world(prim, local @ camera_rigid)
        place(marker, (0, 0, -4))
        place(backdrop, (0, 0, -5))
        thermal_yaml = yaml.safe_load(config.sensors.thermal_config.read_text(encoding='utf-8'))
        thermal_yaml['targets']['sync_hot'] = {'mode': 'fixed', 'temperature_c': 80.}
        thermal_yaml['targets']['sync_cold'] = {'mode': 'fixed', 'temperature_c': 10.}
        thermal_yaml['bindings'] += [
            {'name': 'sync_marker', 'prim_path': marker_path, 'include_children': True,
             'target': 'sync_hot', 'material': 'concrete'},
            {'name': 'sync_backdrop', 'prim_path': backdrop_path, 'include_children': True,
             'target': 'sync_cold', 'material': 'concrete'}]
        test_config = out/'probe_thermal.yaml'
        test_config.write_text(yaml.safe_dump(thermal_yaml, sort_keys=False), encoding='utf-8')
        from scripts.start_crazyflie_dual_camera_view import open_preview
        report['phase'] = 'building_preview'
        checkpoint()
        preview, _ = open_preview(app, scene_config=config, stage_path=config.scene.stage,
                                  width=args.width, height=args.height, fast=False,
                                  thermal_width=args.thermal_width, thermal_height=args.thermal_height,
                                  thermal_config_path=test_config, start_controller=False, show_ui=args.gui,
                                  renderer=args.renderer)
        from scripts.experiment_conditions import snapshot_conditions
        report['experiment_conditions'] = snapshot_conditions(
            stage, config, args.width, args.height,
            thermal_width=args.thermal_width, thermal_height=args.thermal_height)
        report['thermal_bindings'] = preview.thermal.thermal_config_report['binding_counts']
        report['probe_thermal_sha256'] = hashlib.sha256(test_config.read_bytes()).hexdigest()
        rgb_schema = UsdGeom.Camera(camera)
        ir_schema = UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.thermal_path))
        optics = {field: [float(getattr(rgb_schema, 'Get'+field+'Attr')().Get()),
                          float(getattr(ir_schema, 'Get'+field+'Attr')().Get())]
                  for field in ('FocalLength', 'HorizontalAperture', 'VerticalAperture')}
        report['paired_optics'] = optics
        optics_match = all(abs(a-b) < 1e-6 for a, b in optics.values())
        initial_pair_pose_error = float(np.max(np.abs(
            matrix(camera) - matrix(stage.GetPrimAtPath(preview.mount.thermal_path)))))
        if abs(args.width / args.height - args.thermal_width / args.thermal_height) > .005:
            raise RuntimeError('RGB and thermal output aspect ratios do not match')
        report['optical_registration'] = {
            'same_camera_world_pose': True,
            'initial_pair_pose_max_error': initial_pair_pose_error,
            'same_focal_length_and_apertures': optics_match,
            'pixel_scale_rgb_to_thermal': [args.width / args.thermal_width,
                                           args.height / args.thermal_height],
            'different_native_resolutions': [args.width, args.height] !=
                                             [args.thermal_width, args.thermal_height],
        }
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.set_auto_update(False)
        step_calls = []
        original_step = rep.orchestrator.step
        def counted_step(*a, **kw):
            step_calls.append({k: kw.get(k) for k in ('delta_time', 'rt_subframes', 'pause_timeline')})
            return original_step(*a, **kw)
        rep.orchestrator.step = counted_step
        original_drone = matrix(drone).copy()
        camera_relative_to_drone = matrix(camera) @ np.linalg.inv(original_drone)
        camera_schema = UsdGeom.Camera(camera)
        focal = camera_schema.GetFocalLengthAttr().Get()
        aperture = camera_schema.GetHorizontalApertureAttr().Get()
        thermal_schema = UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.thermal_path))
        thermal_focal = thermal_schema.GetFocalLengthAttr().Get()
        thermal_aperture = thermal_schema.GetHorizontalApertureAttr().Get()
        thermal_vertical_aperture = thermal_schema.GetVerticalApertureAttr().Get()
        report['phase'] = 'capturing'
        checkpoint()
        previous_rgb = previous_ir = None
        expected_rgb_positions = []
        expected_thermal_positions = []
        if records is not None:
            from scripts.replay_pose import FrozenPoseRestorer
            restorer = FrozenPoseRestorer(stage, config)
        for i in range(args.frames):
            restored = original_drone.copy()
            if records is not None:
                desired_body, _ = restorer.restore(records[i])
                restored = matrix(drone)
            elif i >= args.frames//2:
                yaw = Gf.Matrix4d(1)
                yaw.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 8*(i-args.frames//2+1)))
                restored[:3, :3] = original_drone[:3, :3] @ np.asarray(yaw)[:3, :3]
                restored[3, 1] += .1*(i-args.frames//2+1)
            if records is None:
                set_world(drone, restored)
            marker_u, marker_v = positions[i]
            thermal_fx = args.thermal_width * thermal_focal / thermal_aperture
            thermal_fy = args.thermal_height * thermal_focal / thermal_vertical_aperture
            marker_local = ((marker_u - args.thermal_width / 2) * 4 / thermal_fx,
                            (args.thermal_height / 2 - marker_v) * 4 / thermal_fy, -4)
            place(marker, marker_local)
            place(backdrop, (0, 0, -5))
            timestamp = float(records[i]['simulation_time']) if records else i*.2
            timeline.set_current_time(timestamp)
            timeline.commit()
            cam_before = matrix(camera).copy()
            follow_error = float(np.max(np.abs(cam_before - camera_relative_to_drone @ restored)))
            body_error = float(np.max(np.abs(matrix(restorer.body) - desired_body))) if records else 0.
            recorded_camera_pose_validation = (
                compare_recorded_pose(cam_before, np.asarray(records[i]['rgb_camera_world_matrix']),
                                      meters_per_unit=stage_meters_per_unit)
                if records and records[i].get('rgb_camera_world_matrix') else None)
            recorded_camera_error = (recorded_camera_pose_validation['matrix_max_abs_difference']
                                     if recorded_camera_pose_validation else 0.)
            marker_before = matrix(marker).copy()
            rgb_uv = expected_uv(marker_before[3, :3], cam_before, args.width, args.height, focal, aperture)
            thermal_camera_matrix = matrix(stage.GetPrimAtPath(preview.mount.thermal_path))
            thermal_uv = expected_uv(marker_before[3, :3], thermal_camera_matrix,
                                     args.thermal_width, args.thermal_height,
                                     thermal_focal, thermal_aperture)
            expected_rgb_positions.append(rgb_uv)
            expected_thermal_positions.append(thermal_uv)
            count_before = len(step_calls)
            rgb, thermal, meta = preview.capture(render=True, frozen_state=True)
            from irsim_isaac.pipeline.material_ids import labels_to_paths
            ir_frame = preview.thermal.last_frame
            paths = labels_to_paths(ir_frame.labels)
            ids = [key for key, path in paths.items() if str(path) == marker_path]
            geometry_mask = np.isin(ir_frame.instance_id, ids)
            color = np.asarray(rgb)[..., :3].astype(float)
            rgb_mask = rgb_marker_mask(rgb)
            output_mask = largest_component(np.isfinite(thermal.apparent_t) & (thermal.apparent_t > 330.))
            geometry_check = compare_paired_masks(rgb_mask, geometry_mask, rgb_uv, thermal_uv)
            output_check = compare_paired_masks(rgb_mask, output_mask, rgb_uv, thermal_uv)
            rgb_history_match = match_marker_position(centroid(rgb_mask), expected_rgb_positions,
                                                      pixel_tolerance=6.)
            geometry_history_match = match_marker_position(centroid(geometry_mask), expected_thermal_positions)
            output_history_match = match_marker_position(centroid(output_mask), expected_thermal_positions)
            history_matches_this_frame = all(
                value['matched_frame_index'] == i and value['candidate_count'] == 1
                for value in (rgb_history_match, geometry_history_match, output_history_match))
            duplicate_rgb = previous_rgb is not None and np.array_equal(previous_rgb, rgb_mask)
            duplicate_ir = previous_ir is not None and np.array_equal(previous_ir, geometry_mask)
            calls = step_calls[count_before:]
            check = {'fresh_marker_state': bool(geometry_check['pass'] and output_check['pass'] and
                                                history_matches_this_frame and not duplicate_rgb and not duplicate_ir),
                     'explicit_render_once': len(calls) == 1 and calls[0]['delta_time'] == 0. and meta['explicit_render_completed'],
                     'pair_pose_match': meta['pair_pose_max_error'] < 1e-6 and initial_pair_pose_error < 1e-6,
                     'camera_follows_restored_drone': follow_error < 1e-6,
                     'recorded_body_restored': body_error < 1e-4,
                     'recorded_camera_restored': (recorded_camera_pose_validation is None or
                                                  recorded_camera_pose_validation['within_tolerance']),
                     'paired_optics_match': optics_match,
                     'requested_timestamp_match': max(abs(meta['simulation_time_before']-timestamp),
                                                      abs(meta['simulation_time_after']-timestamp)) < 1e-6,
                     'marker_pose_frozen': float(np.max(np.abs(matrix(marker)-marker_before))) < 1e-8,
                     'simulation_frozen': abs(meta['simulation_capture_drift_s']) < 1e-8,
                     'thermal_clock_match': abs(meta['thermal_clock_drift_s']) < 1e-6,
                     'stream_dimensions_match': rgb.shape[:2] == (args.height, args.width) and
                                                thermal.display8.shape[:2] ==
                                                (args.thermal_height, args.thermal_width),
                     'pose_frozen': float(np.max(np.abs(matrix(camera)-cam_before))) < 1e-8}
            check['dimensions_match'] = check['stream_dimensions_match']
            sample = {**meta, 'strict_sync_verified': bool(all(check.values())),
                      'strict_per_frame_pixel_sync_verified': bool(all(check.values())),
                      'trajectory_frame_id': records[i]['frame_id'] if records else None,
                      'restored_body_pose_max_error': body_error,
                      'recorded_camera_pose_max_error': recorded_camera_error,
                      'recorded_camera_pose_validation': recorded_camera_pose_validation,
                      'stage_meters_per_unit': stage_meters_per_unit,
                      'expected_marker_uv': rgb_uv, 'expected_rgb_marker_uv': rgb_uv,
                      'expected_thermal_marker_uv': thermal_uv, 'marker_world_matrix': marker_before.tolist(),
                      'drone_world_matrix': restored.tolist(), 'snapshot_time_s': timestamp,
                      'camera_follow_max_error': follow_error,
                      'simulation_interval_s': None if i == 0 else timestamp-report['frames'][-1]['snapshot_time_s'],
                      'geometry_pixel_check': geometry_check, 'thermal_output_pixel_check': output_check,
                      'rgb_history_match': rgb_history_match,
                      'thermal_geometry_history_match': geometry_history_match,
                      'thermal_output_history_match': output_history_match,
                      'marker_same_frame_verified': history_matches_this_frame,
                      'marker_detection_thresholds': {'rgb_green_core_min_uint8': 128,
                                                       'thermal_apparent_t_min_k': 330.},
                      'rgb_read_status': 'OK', 'thermal_read_status': 'OK',
                      'duplicate_rgb_mask': duplicate_rgb, 'duplicate_ir_mask': duplicate_ir,
                      'render_calls': calls, 'checks': check, 'pass': bool(all(check.values())),
                      'sync_status': 'PASS_RESTORED_STATE' if all(check.values()) else 'FAIL_RESTORED_STATE'}
            preview.save_frame(out, i, rgb, thermal, sample)
            from PIL import Image
            overlay = registered_overlay(color, rgb_mask, geometry_mask)
            Image.fromarray(overlay).save(out/f'frame_{i:06d}_overlay.png')
            report['frames'].append(sample)
            checkpoint()
            previous_rgb, previous_ir = rgb_mask.copy(), geometry_mask.copy()
        passed = all(s['pass'] for s in report['frames'])
        order_ok = [s['frame_id'] for s in report['frames']] == list(range(args.frames))
        times_ok = all(b['simulation_time_before'] > a['simulation_time_before']
                       for a, b in zip(report['frames'], report['frames'][1:]))
        passed = passed and order_ok and times_ok
        report.update(result='PASS' if passed else 'FAIL', automated_restored_state_sync_pass=passed,
                      strict_sync_verified=passed, strict_per_frame_pixel_sync_verified=passed, phase='complete',
                      marker_history_verified=passed,
                      marker_history_max_age_frames=0 if passed else None,
                      failed_frames=[s['frame_id'] for s in report['frames'] if not s['pass']],
                      spot_visible_in_sensor='NOT_VERIFIED',
                      spot_thermal_model=report['experiment_conditions']['spot'])
        report.update(frame_sequence_complete=order_ok, snapshot_times_monotonic=times_ok)
        status = 0 if passed else 1
    except Exception:
        failure = traceback.format_exc()
        (out/'failure.txt').write_text(failure, encoding='utf-8')
        print(failure, flush=True)
        report.update(result='FAIL', error=failure)
    finally:
        if 'faulthandler' in locals():
            faulthandler.cancel_dump_traceback_later()
        if original_step is not None:
            rep.orchestrator.step = original_step
        if preview:
            preview.close()
        if stage and layer:
            stage.SetEditTarget(old_edit)
            stage.GetSessionLayer().subLayerPaths.remove(layer.identifier)
        after = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
        report.update(source_usd_sha256_after=after, source_unchanged=after == before)
        if before != after:
            report.update(result='FAIL', strict_sync_verified=False, strict_per_frame_pixel_sync_verified=False)
            status = 1
        checkpoint()
        print('DUAL_CAMERA_SYNC_' + report['result'] + ' ' + str(out), flush=True)
        if app:
            app.close()
    return status


if __name__ == '__main__':
    raise SystemExit(main())
