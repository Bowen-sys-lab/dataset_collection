"""Three-round serial-chain measurements and owned diagnostic snapshots, never dataset frames."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
sys.path.insert(0, str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def conditions_for_run(condition, *, profile_only=False):
    # Never disable and re-enable AOV products in one Kit process: cached GPU
    # annotator handles can become invalid on this build.
    if profile_only:
        return [condition+'_profiled']
    return [condition, 'full_thermal_profiled'] if condition == 'full_thermal' else [condition]


def commanded_pose(origin, index):
    """Small bounded diagnostic motion, indexed by captures rather than wall time."""
    return {'position': [origin['position'][0]+.2*math.sin(index*.037),
                         origin['position'][1]+.15*math.sin(index*.023),
                         origin['position'][2]+.05*math.sin(index*.031)],
            'yaw': origin['yaw']+.05*math.sin(index*.029)}


def apply_commanded_pose(controller, pose):
    controller.keys.pressed.clear()
    controller.position = tuple(pose['position'])
    controller.yaw = pose['yaw']
    controller.velocity = (0., 0., 0., 0.)
    controller.roll = controller.pitch = controller.roll_rate = controller.pitch_rate = 0.


def process_memory():
    """Current process working set/private usage via Windows API; no dependencies."""
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)]+[
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    try:
        kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info = Counters()
        info.cb = ctypes.sizeof(info)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
            return None
        return {'working_set_bytes': int(info.WorkingSetSize), 'private_usage_bytes': int(info.PrivateUsage)}
    except (AttributeError, OSError):
        return None


def experiment_file_hashes(config):
    files = {name: path for name, path in {
        'scene': config.scene.stage, 'scene_config': config.source_path,
        'thermal_config': config.sensors.thermal_config, 'sensor_config': config.sensors.sensor_config,
        'thermal_environment_config': config.scene.thermal_scene}.items()}
    for directory in ('scripts', 'third_party/isaac-thermal-camera-simulation/src'):
        files.update({str(path.relative_to(ROOT)).replace('\\', '/'): path for path in (ROOT/directory).rglob('*.py')})
    files['tests/isaac_thermal_baseline.py'] = Path(__file__)
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}


def finalize_integrity(report, *, source_unchanged, vendor_unchanged):
    report['source_unchanged'] = bool(source_unchanged)
    report['vendor_source_unchanged'] = bool(vendor_unchanged)
    if not (source_unchanged and vendor_unchanged):
        report['result'] = 'FAIL'


def noise_with_seed(noise, sensor_seed):
    from dataclasses import replace
    from irsim.noise.three_d import FixedPattern, Sigmas7
    seed = int(sensor_seed)
    fixed = FixedPattern.generate(noise.shape, Sigmas7.from_ratios(1., noise.ratios), seed)
    return replace(noise, sensor_seed=seed, unit_fixed=fixed)


def benchmark(app, preview, controller, args):
    import numpy as np
    import omni.timeline
    from scripts.thermal_diagnostics import (StageTimings, thermal_specs, statistics,
        summarize_segments, profile_material_mapping_scope, temperature_counts)
    from scripts.preview_metrics import ControlMetrics, configure_live_viewports
    from scripts.dual_camera_support import ui_rgba_array
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    metrics = ControlMetrics(controller)
    product = preview.thermal._render_product.hydra_texture
    original_enabled = product.updates_enabled if hasattr(product, 'updates_enabled') else True
    cache_experiment = args.condition.startswith('material_cache_')
    report = {'rounds': [], 'resolution': [160, 128], 'renderer': 'PathTracing',
              'warmup_frames': args.warmup, 'external_viewport_enabled': True,
              'preview_rate_mode': 'uncapped', 'strict_sync_verified': False,
              'valid_new_pair_fps_verified': False, 'human_input_latency_verified': False,
              'gui_driving_acceptance_verified': False,
              'preview_sync_status': 'live_preview_not_strict', 'data_usage': 'diagnostic-only'}
    report['enabled_viewport_count'] = configure_live_viewports(preview.cleanup, enabled=True)
    if report['enabled_viewport_count'] == 0:
        raise RuntimeError('External viewport required for baseline was not found')
    if cache_experiment and report['enabled_viewport_count'] != 1:
        raise RuntimeError('Material-cache experiment requires exactly one external viewport')
    preview.skip_live_preconversion = True
    report['preconversion'] = False
    report['shared_live_render_product'] = bool(preview.metadata.get('shared_live_render_product', False))
    report['motion_mode'] = 'commanded_capture_sinusoid_v1' if cache_experiment else 'baseline_controller_hold'
    try:
        if cache_experiment:
            # Allow normal controller callbacks to initialize their physics state.
            ready_deadline = time.monotonic()+30
            while not controller._ready:
                app.update()
                if not controller._running or time.monotonic() > ready_deadline:
                    raise RuntimeError('Controller did not initialize for material-cache benchmark')
            origin = {'position': [float(v) for v in controller.position], 'yaw': float(controller.yaw)}
            report['motion_baseline'] = origin
            report['motion_baseline_actual_state'] = controller.state_snapshot()
            sequence = [commanded_pose(origin, i) for i in range(args.profile_frames if args.profile_only else args.frames)]
            report['commanded_motion_sequence'] = sequence
            report['motion_sequence_sha256'] = hashlib.sha256(json.dumps(sequence, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for round_id in range(args.rounds):
            for condition in conditions_for_run(args.condition, profile_only=args.profile_only):
                product.set_updates_enabled(condition != 'rgb_only_aovs_disabled')
                full = condition.startswith(('full_thermal', 'material_cache_'))
                profiled = condition.endswith('_profiled')
                timer = StageTimings()
                profiler = None
                if profiled:
                    import cProfile
                    profiler = cProfile.Profile()
                    preview.profiler = profiler
                samples, segments = [], []
                actual_poses = []
                cache_before = cache_after = memory_before = memory_after = None
                counts_before = {}
                measured = args.profile_frames if profiled else args.frames
                with (temperature_counts(preview.thermal) if cache_experiment else nullcontext({})) as counts:
                    for index in range(args.warmup + measured):
                        if index == args.warmup and cache_experiment:
                            # Keep the warmed lookup; sample memory/counters outside frame timers.
                            cache_before = preview.thermal.material_cache_stats()
                            memory_before = process_memory()
                            counts_before = dict(counts)
                        start, cpu = time.perf_counter(), time.thread_time()
                        if cache_experiment:
                            motion_index = index if index < args.warmup else index-args.warmup
                            apply_commanded_pose(controller, commanded_pose(origin, motion_index))
                        update_start = time.perf_counter()
                        app.update()
                        update_done = time.perf_counter()
                        if not controller._ready:
                            raise RuntimeError('Controller not ready after warm-up')
                        timer.reset()
                        if full:
                            with (timer.wrap(thermal_specs(preview.thermal)) if profiled else nullcontext()), \
                                    (profile_material_mapping_scope(timer) if profiled else nullcontext()):
                                rgb, thermal, meta = preview.capture(render=False, publish=False)
                            thermal_done = time.perf_counter()
                            ui_start = time.perf_counter()
                            preview.publish_pair(rgb, thermal, {
                                **meta, 'frame_id': index,
                                'controller_sim_time': timeline.get_current_time(),
                                'preview_processing_ms': (thermal_done-update_done)*1000,
                                'dropped_preview_frames': 0})
                            rgb_read = preview.last_timings['rgb_read_s']
                            thermal_outputs = preview.last_timings['thermal_get_outputs_s']
                        else:
                            rgb = np.asarray(preview.rgb_annotator.get_data())
                            thermal_done = time.perf_counter()
                            rgb_read = thermal_done-update_done
                            thermal_outputs = 0.
                            ui_start = time.perf_counter()
                            data, size = ui_rgba_array(rgb)
                            preview.rgb_provider.set_data_array(data, size, strict=True)
                        finished = time.perf_counter()
                        cpu_elapsed = time.thread_time()-cpu
                        if index >= args.warmup:
                            samples.append({'app_update_s': update_done-update_start,
                                'thermal_and_rgb_s': thermal_done-update_done,
                                'capture_total_s': finished-update_done,
                                'thermal_get_outputs_s': thermal_outputs,
                                'rgb_read_s': rgb_read, 'ui_upload_s': finished-ui_start,
                                'loop_s': finished-start, 'main_thread_cpu_s': cpu_elapsed})
                            if cache_experiment:
                                actual_poses.append({'position': [float(v) for v in controller.position],
                                    'yaw': float(controller.yaw), 'camera_world_matrix': meta['camera_world_matrix']})
                            if profiled:
                                segments.append(deepcopy(timer.values))
                    if cache_experiment:
                        cache_after = preview.thermal.material_cache_stats()
                        memory_after = process_memory()
                        counts_after = dict(counts)
                row = {'round': round_id, 'condition': condition,
                       'profiled': profiled,
                       'samples': samples, 'processing_fps': 1/np.mean([s['loop_s'] for s in samples]),
                       'timings': {key: statistics(s[key] for s in samples) for key in samples[0]},
                       'segments': summarize_segments(segments)}
                report['rounds'].append(row)
                if cache_experiment:
                    row.update(cache_stats_before=cache_before, cache_stats_after=cache_after,
                        cache_stats_delta={key: cache_after[key]-cache_before[key] for key in ('hits', 'builds', 'evictions')},
                        process_memory_before=memory_before, process_memory_after=memory_after,
                        temperature_update_counts={key: counts_after[key]-counts_before[key] for key in counts_after},
                        actual_capture_poses=actual_poses,
                        actual_pose_sequence_sha256=hashlib.sha256(json.dumps(actual_poses, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
                if profiled:
                    import pstats
                    profiler.dump_stats(str(args.out/f'round_{round_id}_thermal.prof'))
                    stats = pstats.Stats(profiler)
                    row['function_hotspots'] = [
                        {'file': key[0], 'line': key[1], 'function': key[2],
                         'calls': value[1], 'self_s': value[2], 'cumulative_s': value[3]}
                        for key, value in sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)[:30]]
                    preview.profiler = None
                write_json(args.out/'baseline.json', report)
                print(f"[baseline] round={round_id} {condition} fps={row['processing_fps']:.2f}", flush=True)
        report['controller_metrics'] = metrics.report()
        report['controller_metrics']['measurement_scope'] = 'Controller initialization, warmup and measured captures; separate from throughput frame timings'
        report['profile_timing_note'] = 'Inclusive groups overlap; do not add them. Exclusive groups subtract timed descendants.'
        report['temperature_counter_note'] = 'Exact method request/target tick counts, not a claim of equal thermal solver histories across wall-clock-dependent live runs'
        report['thread_cpu_clock_note'] = 'Windows thread_time can be quantized; short frame and stage CPU durations are coarse'
        report['result'] = 'MEASURED'
        return report
    finally:
        product.set_updates_enabled(original_enabled)
        metrics.close()


def snapshots(app, preview, controller, config, args):
    import omni.timeline
    from scripts.thermal_snapshot import anonymous_session_layer
    if controller is not None:
        controller.stop()
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_auto_update(False)
    # PhysX restores its simulation-start transforms on the next app tick.
    # Drain that restoration before switching USD edit targets.
    app.update()
    with anonymous_session_layer(preview.stage):
        return _snapshots_in_layer(app, preview, None, config, args)


def _snapshots_in_layer(app, preview, controller, config, args):
    import numpy as np
    import omni.timeline
    from pxr import Sdf
    from scripts.thermal_snapshot import (capture_numerical_boundary, save_snapshot,
                                         copy_config, owned_copy)
    from scripts.thermal_imaging import display_from_outputs
    from scripts.replay_pose import world_matrix
    from isaac_dual_camera_sync import sheet, set_world
    from irsim_isaac.pipeline.material_ids import labels_from_payload
    camera = preview.thermal
    if controller is not None:
        controller.stop()
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.set_auto_update(False)
    stage = preview.stage
    # Timeline is stopped, so frozen renders do not need physics edits. Disabling
    # individual links of an existing articulation is unsupported by PhysX.
    original_drone = world_matrix(stage.GetPrimAtPath(config.drone.prim_path)).copy()
    source_camera = stage.GetPrimAtPath(preview.mount.rgb_path)
    # Occlusion is created only after performance measurement, not in the baseline scene.
    blocker = sheet(stage, '/World/DiagnosticOccluder', 2., 1.5, (0., 1., 0.))
    blocker.CreateAttribute('visibility', Sdf.ValueTypeNames.Token).Set('invisible')
    from types import SimpleNamespace
    target = next(iter(camera.bridge.prim_to_target.values()))
    camera.bridge.prim_to_target['/World/DiagnosticOccluder'] = target
    camera.resolutions = list(camera.resolutions) + [SimpleNamespace(
        path='/World/DiagnosticOccluder', material_id=camera.config.materials.id_for('concrete'))]
    baseline_state = owned_copy(camera.state)
    baseline_config = copy_config(camera.config, camera.state.t_s)
    baseline_motion = owned_copy(camera._motion)
    directory = args.out/'snapshots'
    directory.mkdir(exist_ok=False)
    captured_paths, raw_holder = [], {}
    reader_original = camera._reader.read
    def read_record():
        raw = reader_original()
        raw_holder.clear()
        raw_holder.update({name: np.array(getattr(raw, name), copy=True)
                           for name in ('distance_m', 'normal', 'position', 'instance_id')})
        raw_holder['labels'] = deepcopy(labels_from_payload(raw.device_handles.get('instance')))
        if args.assembly_snapshots:
            from scripts.thermal_snapshot import assembly_checkpoint
            raw_holder['assembly_checkpoint'] = assembly_checkpoint(camera, raw)
        return raw
    camera._reader.read = read_record
    base_translation = original_drone[3, :3].copy()
    reports = []
    try:
        cases = ['static', 'static', 'translation', 'translation', 'rotation', 'rotation',
                 'occlusion', 'occlusion', 'paused', 'resumed', 'reset',
                 'instance_label_change', 'config_change', 'after_config_change']
        for index, case in enumerate(cases):
            metadata = {'case': case, 'snapshot_id': index, 'reset_history': index == 0 or case == 'reset',
                'config_changed': case == 'config_change',
                'simulation_time_s': index*.1, 'seed': config.sensors.sensor_seed,
                'scene_config_path': str(config.source_path), 'scene_config_sha256': config.sha256,
                'experiment_conditions': preview.metadata['experiment_conditions'],
                'source_time_scope': 'Frozen diagnostic snapshot, not live image age'}
            metadata['lifecycle_test_scope'] = 'Frozen numeric time hold/resume/reset, NOT GUI or controller lifecycle'
            if case == 'reset':
                camera.state = owned_copy(baseline_state)
                camera.config = owned_copy(baseline_config)
                camera._motion = owned_copy(baseline_motion)
            if case == 'config_change':
                seed = camera.config.sensor_seed+1
                noise = noise_with_seed(camera.config.noise, seed)
                camera.config = replace(camera.config, sensor_seed=seed, noise=noise)
            if case == 'paused':
                metadata['simulation_time_s'] = (index-1)*.1
            timeline.set_current_time(metadata['simulation_time_s'])
            timeline.commit()
            world = original_drone.copy()
            if case in ('translation', 'rotation'):
                world[3, :3] = base_translation + np.array([.2*(index-1), 0., 0.])
            if case == 'rotation':
                theta = np.deg2rad(8*(index-3))
                rotation = np.array([[np.cos(theta), np.sin(theta), 0.],
                                     [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
                world[:3, :3] = world[:3, :3] @ rotation
            set_world(stage.GetPrimAtPath(config.drone.prim_path), world)
            visible = case in ('occlusion', 'paused', 'resumed', 'instance_label_change')
            blocker.GetAttribute('visibility').Set('inherited' if visible else 'invisible')
            if case == 'instance_label_change':
                # New geometry with a new path tests the next instance-label payload.
                blocker.GetAttribute('visibility').Set('invisible')
                blocker = sheet(stage, '/World/DiagnosticOccluderNew', 2., 1.5, (0., 1., 0.))
                camera.bridge.prim_to_target['/World/DiagnosticOccluderNew'] = target
                camera.resolutions = list(camera.resolutions) + [SimpleNamespace(
                    path='/World/DiagnosticOccluderNew', material_id=camera.config.materials.id_for('concrete'))]
            rigid = world_matrix(source_camera)
            rigid[:3, :3] /= np.linalg.norm(rigid[:3, :3], axis=1)[:, None]
            offset = np.eye(4)
            offset[3, 2] = -4.
            set_world(blocker, offset @ rigid)
            motion_before = owned_copy(camera._motion)
            with capture_numerical_boundary(camera, metadata) as snapshot:
                rgb, _, capture_meta = preview.capture(render=True, frozen_state=True,
                    simulation_time=metadata['simulation_time_s'], publish=False)
            snapshot['outputs'] = owned_copy(display_from_outputs(camera.config, snapshot['raw_outputs']))
            snapshot['rgb'] = np.array(rgb, copy=True)
            snapshot['metadata']['camera_world_matrix'] = capture_meta['camera_world_matrix']
            snapshot['metadata']['previous_motion_history'] = vars(motion_before)
            snapshot['metadata']['current_motion_history'] = owned_copy(vars(camera._motion))
            snapshot['metadata']['sensor_chain_attached'] = camera.config.chain is not None
            ids, labels = raw_holder['instance_id'], raw_holder['labels']
            motion_args = None
            if camera._motion.ready:
                previous, current = camera._motion.transform_stacks(labels)
                motion_args = owned_copy((raw_holder['position'], ids.astype(np.int64), previous, current,
                    camera._motion._prev_camera, camera._motion._cur_camera,
                    camera.optics.intrinsics, camera.sensor.sensor.optics.distortion))
            snapshot['assembly_evidence'] = {'raw_aovs': owned_copy(raw_holder),
                'instance_ids': owned_copy(ids), 'labels': owned_copy(labels),
                'resolutions': owned_copy(tuple(camera.resolutions)), 'motion_args': motion_args}
            if args.assembly_snapshots:
                from scripts.thermal_snapshot import camera_history
                snapshot['assembly_checkpoint'] = raw_holder.pop('assembly_checkpoint')
                snapshot['assembly_history_after'] = camera_history(camera)
            snapshot['metadata']['owned_aov_arrays'] = {
                name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': int(value.nbytes)}
                for name, value in raw_holder.items() if isinstance(value, np.ndarray)}
            path = directory/f'frame_{index:06d}.pkl'
            save_snapshot(path, snapshot)
            captured_paths.append(path)
            reports.append({'snapshot_id': index, 'case': case, 'path': str(path),
                'rgb_camera_matrix': capture_meta['camera_world_matrix'],
                'label_hash': hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest(),
                'sensor_chain_attached': camera.config.chain is not None})
            print(f'[snapshot] {case} {path.name}', flush=True)
        from scripts.compare_thermal_snapshots import compare_sequence
        comparison = compare_sequence(captured_paths)
        if args.assembly_snapshots:
            from scripts.compare_thermal_snapshots import compare_material_cache_sequence
            comparison = compare_material_cache_sequence(captured_paths)
        write_json(args.out/'comparison.json', comparison)
        return {'result': comparison['result'], 'snapshots': reports,
                'numerical_chain_exact': comparison['result'] == 'PASS'}
    finally:
        camera._reader.read = reader_original


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/school_vln_scene.yaml')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--profile-frames', type=int, default=30)
    parser.add_argument('--profile-only', action='store_true', help='Dedicated profiling pass; no throughput round')
    parser.add_argument('--phase', choices=('baseline', 'snapshots', 'both'), default='both')
    parser.add_argument('--assembly-snapshots', action='store_true')
    parser.add_argument('--condition', choices=('full_thermal', 'cpu_bypass_aovs_retained',
                                               'rgb_only_aovs_disabled', 'material_cache_off',
                                               'material_cache_on'), default='full_thermal')
    args = parser.parse_args(argv)
    if args.condition == 'rgb_only_aovs_disabled' and args.phase != 'baseline':
        parser.error('Disabled AOV condition requires --phase baseline in a dedicated process')
    if min(args.rounds, args.warmup, args.frames, args.profile_frames) < 1:
        parser.error('Counts must be positive')
    if args.profile_only and (args.phase != 'baseline' or args.condition not in ('full_thermal', 'material_cache_off', 'material_cache_on')):
        parser.error('--profile-only requires baseline phase and a full thermal condition')
    if args.condition.startswith('material_cache_'):
        if args.phase != 'baseline' or args.rounds != 1 or args.warmup < 30:
            parser.error('Material-cache condition requires --phase baseline --rounds 1 --warmup >=30')
        if not args.profile_only and args.frames != 600:
            parser.error('Unprofiled material-cache process requires exactly --frames 600')
    return args


def main():
    args = parse_arguments()
    args.out.mkdir(parents=True, exist_ok=False)
    from scripts.scene_config import load_scene_config
    config = load_scene_config(args.config)
    before = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
    source_files = list((ROOT/'third_party/isaac-thermal-camera-simulation/src').rglob('*.py'))
    def source_digest():
        return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(source_files))).hexdigest()
    vendor_before = source_digest()
    hashes_before = experiment_file_hashes(config)
    report = {'result': 'INCOMPLETE', 'stage_sha256_before': before,
              'vendor_src_sha256_before': vendor_before, 'data_usage': 'diagnostic-only',
              'strict_sync_verified': False, 'phase': args.phase,
              'process_id': os.getpid(), 'command_arguments': sys.argv[1:],
              'file_hashes_before': hashes_before}
    app = preview = controller = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': False, 'renderer': 'PathTracing', 'multi_gpu': False,
                             'samples_per_pixel_per_frame': 1})
        from scripts.start_crazyflie_dual_camera_view import open_preview
        preview, controller = open_preview(app, scene_config=config, width=160, height=128,
            material_cache_enabled=args.condition == 'material_cache_on')
        report['experiment_conditions'] = preview.metadata['experiment_conditions']
        if args.phase in ('baseline', 'both'):
            report['baseline'] = benchmark(app, preview, controller, args)
            write_json(args.out/'baseline.json', report['baseline'])
        if args.phase in ('snapshots', 'both'):
            report['snapshots'] = snapshots(app, preview, controller, config, args)
            controller = None
        report['result'] = 'PASS' if report.get('snapshots', {}).get('result', 'PASS') == 'PASS' else 'FAIL'
    except Exception:
        report['result'] = 'FAIL'
        report['failure'] = traceback.format_exc()
        (args.out/'failure.txt').write_text(report['failure'], encoding='utf-8')
        print(report['failure'], flush=True)
    finally:
        report['file_hashes_after'] = experiment_file_hashes(config)
        finalize_integrity(report,
            source_unchanged=before == hashlib.sha256(config.scene.stage.read_bytes()).hexdigest(),
            vendor_unchanged=vendor_before == source_digest())
        if report['file_hashes_after'] != hashes_before:
            report['result'] = 'FAIL'
            report['project_config_controller_unchanged'] = False
        else:
            report['project_config_controller_unchanged'] = True
        write_json(args.out/'summary.json', report)
        if controller is not None:
            controller.stop()
        if preview is not None:
            preview.close()
        if app is not None:
            app.close()
    return 0 if report['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
