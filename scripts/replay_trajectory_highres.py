"""Replay a recorded flight and capture high-resolution RGB/thermal pairs.

This is deliberately offline: it restores each recorded pose, triggers one
explicit zero-time render, and writes a self-contained frame bundle.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/isaac-thermal-camera-simulation/src'))

from scripts.collection_capture_defaults import (
    RGB_SIZE, THERMAL_SIZE, SAMPLE_PERIOD_S, replay_capture_contract,
)
from scripts.scene_config import DEFAULT_SCENE_CONFIG, load_scene_config
from scripts.trajectory_recorder import read_trajectory, trajectory_provenance
from scripts.sync_validation_metadata import build_sync_validation_metadata

TONEMAP_LOCK_SETTINGS = {
    '/rtx/post/tonemap/op': 6,
    '/rtx/post/tonemap/filmIso': 100.0,
    '/rtx/post/tonemap/cameraShutter': 50.0,
    '/rtx/post/tonemap/fNumber': 5.0,
    '/rtx/post/histogram/enabled': False,
}


def _positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('must be a positive integer') from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rgb-width', type=_positive_int, default=RGB_SIZE[0])
    parser.add_argument('--rgb-height', type=_positive_int, default=RGB_SIZE[1])
    parser.add_argument('--thermal-width', type=_positive_int, default=THERMAL_SIZE[0])
    parser.add_argument('--thermal-height', type=_positive_int, default=THERMAL_SIZE[1])
    parser.add_argument('--width', type=_positive_int, default=None,
                        help='legacy override: set both RGB and thermal widths')
    parser.add_argument('--height', type=_positive_int, default=None,
                        help='legacy override: set both RGB and thermal heights')
    parser.add_argument('--sample-interval', type=float, default=SAMPLE_PERIOD_S)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='optional frame cap for diagnostics; 0 replays the full trajectory')
    parser.add_argument('--flat-field', choices=('config', 'on', 'off'), default='config',
                        help='display-only flat-field override; config uses the scene YAML')
    parser.add_argument('--samples-per-pixel', type=_positive_int, default=1,
                        help='PathTracing quality for this replay; normal replay default is 1')
    parser.add_argument('--rt-subframes', type=_positive_int, default=4,
                        help='render subframes per frozen capture (default 4). Lowering it '
                             'shortens each frame but risks the Hydra drain lagging the pose '
                             'edit, so verify frames still differ before using fewer.')
    parser.add_argument('--lock-current-tonemap', action='store_true',
                        help='diagnostic only: explicitly set recorded current exposure/tonemap values')
    return parser


def resolve_capture_resolutions(args):
    """Return RGB and thermal (width, height), preserving the source aspect ratio."""
    if args.width is not None or args.height is not None:
        # Historical CLI behavior used 640x512 when only one legacy dimension
        # was supplied. Keep that contract and apply it to both streams.
        width = 640 if args.width is None else args.width
        height = 512 if args.height is None else args.height
        rgb, thermal = (width, height), (width, height)
    else:
        rgb = (args.rgb_width, args.rgb_height)
        thermal = (args.thermal_width, args.thermal_height)
    if any(value <= 0 for value in (*rgb, *thermal)):
        raise ValueError('RGB and thermal dimensions must be positive')
    rgb_aspect = rgb[0] / rgb[1]
    thermal_aspect = thermal[0] / thermal[1]
    if abs(rgb_aspect - thermal_aspect) / max(rgb_aspect, thermal_aspect) > 0.005:
        raise ValueError('RGB and thermal aspect ratios must match to preserve paired camera framing')
    return rgb, thermal


def apply_tonemap_lock(settings):
    """Set and verify the renderer exposure/tonemap values recorded in the current replay."""
    applied = {}
    for key, expected in TONEMAP_LOCK_SETTINGS.items():
        settings.set(key, expected)
        actual = settings.get(key)
        if actual != expected:
            raise RuntimeError(
                f'Could not lock renderer setting {key}: expected {expected!r}, got {actual!r}'
            )
        applied[key] = actual
    return applied


def _sample(records, interval):
    if interval <= 0:
        return records
    selected, next_time = [], None
    for item in records:
        t = float(item['simulation_time'])
        if next_time is None or t >= next_time - 1e-9:
            selected.append(item)
            next_time = t + interval
    if selected[-1] is not records[-1]:
        selected.append(records[-1])
    return selected


def prepare_replay_input(trajectory, output, interval):
    """Keep dense raw states immutable; replay a sampled, traceable T sequence."""
    from scripts.episode_replay_source import prepare_source
    if not (Path(trajectory).parent/'run_metadata.json').is_file():
        return Path(trajectory)
    return prepare_source(trajectory, output, interval)


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_scene_config(args.config, include_task=False)
    provenance = trajectory_provenance(args.trajectory)
    if not math.isfinite(args.sample_interval) or args.sample_interval <= 0:
        raise ValueError('--sample-interval must be finite and positive')
    if args.max_frames < 0:
        raise ValueError('--max-frames must be zero or positive')
    if args.flat_field != 'config':
        sensors = replace(config.sensors, flat_field_enabled=(args.flat_field == 'on'))
        config = replace(config, sensors=sensors)
    (rgb_width, rgb_height), (thermal_width, thermal_height) = resolve_capture_resolutions(args)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        source_hash = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
        recorded_stage_hash = provenance.get('recording_stage_sha256')
        if recorded_stage_hash is not None and recorded_stage_hash != source_hash:
            raise RuntimeError(
                f'The recorded source USD differs from the replay USD: '
                f'recorded={recorded_stage_hash}, replay={source_hash}. '
                'Use the original scene version for this trajectory.')
    except Exception:
        (args.out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    try:
        args.trajectory = prepare_replay_input(args.trajectory, args.out/'derived_input', args.sample_interval)
        provenance = trajectory_provenance(args.trajectory)
        records = _sample(read_trajectory(args.trajectory), args.sample_interval)
        if args.max_frames:
            records = records[:args.max_frames]
    except Exception:
        (args.out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'PathTracing',
                         'samples_per_pixel_per_frame': args.samples_per_pixel,
                         'multi_gpu': False})
    preview = controller = None
    tonemap_lock = None
    thermal_hash = hashlib.sha256(config.sensors.thermal_config.read_bytes()).hexdigest()
    started = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    try:
        if args.lock_current_tonemap:
            import carb
            tonemap_lock = apply_tonemap_lock(carb.settings.get_settings())
        from scripts.start_crazyflie_dual_camera_view import open_preview
        import omni.timeline
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        preview, controller = open_preview(
            app, scene_config=config, width=rgb_width, height=rgb_height,
            thermal_width=thermal_width, thermal_height=thermal_height,
            frozen_subframes=args.rt_subframes,
            fast=False, start_controller=False, show_ui=False)
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.set_auto_update(False)
        stage = preview.stage
        stage_meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        # Offline replay authors frozen transforms, not paused PhysX tensors.
        stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        from scripts.replay_pose import (
            MAX_POSE_ROTATION_ERROR_DEGREES,
            MAX_POSE_TRANSLATION_ERROR_M,
            FrozenPoseRestorer,
            compare_pose_matrices,
            pose_within_tolerance,
            world_matrix,
        )
        from scripts.experiment_conditions import snapshot_conditions
        restorer = FrozenPoseRestorer(stage, config)
        conditions = snapshot_conditions(stage, config, rgb_width, rgb_height,
                                         thermal_width=thermal_width,
                                         thermal_height=thermal_height)
        preview.metadata['experiment_conditions'] = conditions
        import carb
        settings = carb.settings.get_settings()
        actual_sampling = {key: settings.get(key) for key in (
            '/rtx/pathtracing/spp', '/rtx/pathtracing/totalSpp', '/rtx/pathtracing/clampSpp')}
        if any(value != args.samples_per_pixel for value in actual_sampling.values()):
            raise RuntimeError(f'PathTracing sampling was not applied: {actual_sampling}')
        if tonemap_lock is not None:
            for key, expected in tonemap_lock.items():
                if settings.get(key) != expected:
                    raise RuntimeError(f'Tonemap lock changed during scene initialization: {key}')
        rgb_render_options = {
            'renderer': 'PathTracing',
            'samples_per_pixel_per_frame': args.samples_per_pixel,
            'actual_sampling_settings': actual_sampling,
            'tonemap_lock_requested': args.lock_current_tonemap,
            'tonemap_settings_locked': tonemap_lock,
            'effective_tonemap_settings': conditions['low_light']['renderer_settings'],
        }
        preview.metadata['rgb_render_options'] = rgb_render_options
        app.update()
        frames = []
        for index, item in enumerate(records):
            timeline.set_current_time(float(item['simulation_time']))
            timeline.commit()
            desired, quat = restorer.restore(item)
            app.update()
            rgb, thermal, meta = preview.capture(render=True,
                                                  simulation_time=float(item['simulation_time']),
                                                  frozen_state=True)
            meta.update({
                'source_stage_sha256': source_hash,
                'scene_config_sha256': config.sha256,
                'thermal_config_sha256': thermal_hash,
                'sensor_config_sha256': hashlib.sha256(
                    config.sensors.sensor_config.read_bytes()).hexdigest(),
                'rgb_render_options': rgb_render_options,
            })
            sync_metadata = build_sync_validation_metadata(meta, ROOT)
            if meta['pair_pose_max_error'] > 1e-6 or abs(meta['simulation_capture_drift_s']) > 1e-8:
                raise RuntimeError(f'Replay frame {index}: camera/time contract failed')
            if not meta.get('thermal_coverage', {}).get('complete_for_this_view'):
                raise RuntimeError(f'Replay frame {index}: unmapped thermal geometry; {meta.get("thermal_coverage")}')
            body_matrix = world_matrix(restorer.body)
            body_error = float(np.max(np.abs(body_matrix - desired)))
            body_pose_errors = compare_pose_matrices(
                body_matrix, desired, meters_per_unit=stage_meters_per_unit)
            recorded_camera = item.get('rgb_camera_world_matrix')
            camera_pose_errors = (None if recorded_camera is None else compare_pose_matrices(
                np.asarray(meta['camera_world_matrix']), np.asarray(recorded_camera),
                meters_per_unit=stage_meters_per_unit))
            camera_error = (None if camera_pose_errors is None else
                            camera_pose_errors['matrix_max_abs_difference'])
            if not pose_within_tolerance(body_pose_errors) or (
                    camera_pose_errors is not None and not pose_within_tolerance(camera_pose_errors)):
                raise RuntimeError(
                    f'Replay frame {index}: restored pose mismatch '
                    f'(limits: translation<={MAX_POSE_TRANSLATION_ERROR_M:g}m, '
                    f'rotation<={MAX_POSE_ROTATION_ERROR_DEGREES:g}deg); '
                    f'body={body_pose_errors}, camera={camera_pose_errors}')
            meta.update({
                'frame_id': index,
                'trajectory_frame_id': item['frame_id'],
                'task_id': (provenance['task'] or {}).get('task_id'),
                'task_result_status': provenance['task_result']['status'],
                'trajectory_sha256': provenance['trajectory_sha256'],
                'trajectory_index': index,
                'replay_simulation_time': float(item['simulation_time']),
                'replay_source': str(args.trajectory.resolve()),
                'capture_mode': 'offline_frozen_state_render',
                'source_stage_sha256': source_hash,
                'resolution': [rgb_width, rgb_height],
                'rgb_resolution': [rgb_width, rgb_height],
                'thermal_resolution': [thermal_width, thermal_height],
                'position': [float(value) for value in desired[3, :3]],
                'orientation': list(quat),
                'controller_target_pose': item.get('controller_target_pose'),
                'source_trajectory_pose': {
                    key: item[key] for key in ('position', 'orientation') if key in item
                },
                'replay_pose_source': restorer.last_pose_source,
                'restored_body_pose_max_error': body_error,
                'recorded_camera_pose_max_error': camera_error,
                'pose_restore_validation': {
                    'stage_meters_per_unit': stage_meters_per_unit,
                    'translation_limit_m': MAX_POSE_TRANSLATION_ERROR_M,
                    'rotation_limit_degrees': MAX_POSE_ROTATION_ERROR_DEGREES,
                    'body': body_pose_errors,
                    'camera': camera_pose_errors,
                },
                'pose_restore_mode': 'session_layer_frozen_usd_transform',
                'flat_field_override': args.flat_field,
                'flat_field_enabled': bool(config.sensors.flat_field_enabled),
                'display_only_correction': True,
                'raw_dn_unchanged': True,
                'radiance_unchanged': True,
                'apparent_temperature_unchanged': True,
                'rgb_render_options': rgb_render_options,
                **sync_metadata,
            })
            preview.save_frame(args.out, index, rgb, thermal, meta)
            frames.append(meta)
            if index % 10 == 0:
                print(f'[replay] {index + 1}/{len(records)} t={item["simulation_time"]:.3f}s', flush=True)
        report = {
            'result': 'PASS', 'source_stage_sha256': source_hash,
            'scene_config_path': str(config.source_path), 'scene_config_sha256': config.sha256,
            'thermal_config_sha256': thermal_hash,
            'trajectory': str(args.trajectory.resolve()), 'frames': len(frames),
            'trajectory_provenance': provenance,
            'capture_contract': provenance.get('capture_contract'),
            'replay_capture_contract': replay_capture_contract(
                args.sample_interval, rgb_size=(rgb_width, rgb_height),
                thermal_size=(thermal_width, thermal_height)),
            'task': provenance['task'], 'task_result': provenance['task_result'],
            'resolution': [rgb_width, rgb_height],
            'rgb_resolution': [rgb_width, rgb_height],
            'thermal_resolution': [thermal_width, thermal_height],
            'sample_interval_s': args.sample_interval,
            'requested_flat_field_mode': args.flat_field,
            'flat_field_enabled': bool(config.sensors.flat_field_enabled),
            'display_only_correction': True,
            'raw_dn_unchanged': True,
            'radiance_unchanged': True,
            'apparent_temperature_unchanged': True,
            'sensor_config_path': str(config.sensors.sensor_config),
            'thermal_config_path': str(config.sensors.thermal_config),
            'sensor_config_sha256': hashlib.sha256(config.sensors.sensor_config.read_bytes()).hexdigest(),
            'rgb_render_options': rgb_render_options,
            'display_profile': (None if preview.display_profile is None else
                                preview.display_profile.metadata()),
            'started_local': started, 'strict_sync_verified': False,
            'strict_per_frame_pixel_sync_verified': False,
            'data_usage': 'pilot/development data',
            'experiment_conditions': conditions,
            'frame_metadata': preview.frame_metadata_summary(args.out),
            'note': 'The run is labeled validated only when each frame passes its runtime capture contract '
                    'and the independent pipeline validation evidence is available.',
        }
        sync_metadata = build_sync_validation_metadata(frames[-1], ROOT)
        report.update({key: sync_metadata[key] for key in (
            'sync_basis', 'sync_status', 'sync_validation_status', 'sync_validation_scope',
            'sync_validation_evidence', 'sync_note')})
        report['sync_scope_failures'] = sync_metadata.get('sync_scope_failures', [])
        report['strict_sync_verified'] = all(frame.get('strict_sync_verified') is True for frame in frames)
        report['strict_per_frame_pixel_sync_verified'] = all(
            frame.get('strict_per_frame_pixel_sync_verified') is True for frame in frames)
        check_names = sync_metadata['sync_validation_checks'].keys()
        report['sync_validation_checks_summary'] = {
            name: all(frame.get('sync_validation_checks', {}).get(name) is True for frame in frames)
            for name in check_names
        }
        if hashlib.sha256(config.scene.stage.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError('Source stage changed during replay')
        if trajectory_provenance(args.trajectory) != provenance:
            raise RuntimeError('Trajectory or recorded task changed during replay')
        (args.out / 'replay_metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('[replay] PASS ' + str(args.out), flush=True)
        return 0
    except Exception:
        (args.out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        if controller:
            controller.stop()
        if preview:
            preview.close()
        app.close()


if __name__ == '__main__':
    raise SystemExit(main())
