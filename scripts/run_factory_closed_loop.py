"""Bounded, collision-guarded AeroVLA/Isaac feedback trial using fresh front RGB."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/factory_autonomous_pilot.yaml')
    parser.add_argument('--model-python', type=Path, default=ROOT/'.venvs/aerovla-pilot/Scripts/python.exe')
    parser.add_argument('--max-steps', type=int, default=3)
    parser.add_argument('--max-sim-seconds', type=float, default=60.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'result': 'RUNNING', 'steps': [], 'input_mode': 'duplicate_front_diagnostic',
              'trajectory_source': 'model_kinematic_feedback_diagnostic', 'feedback_verified': False,
              'task_result': {'status': 'unjudged'}, 'physics_response_certified': False,
              'ground_truth_direction_assistance': False, 'rewind_enabled': False,
              'video_generation': {'status': 'SKIPPED', 'reason': 'manual_only', 'path': None},
              'limits': {'steps': args.max_steps, 'simulation_seconds': args.max_sim_seconds,
                         'inference_timeout_seconds': 180}}
    app = preview = worker = worker_log = None
    trajectory = []
    exchange = args.out/'model-worker'
    report_path = args.out/'report.json'

    def persist():
        temporary = report_path.with_suffix('.pending')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, report_path)

    def abort_requested():
        return (args.out/'abort.request').exists()

    try:
        report['configuration_sha256'] = sha(args.config)
        if (not 1 <= args.max_steps <= 20 or not math.isfinite(args.max_sim_seconds)
            or not 0 < args.max_sim_seconds <= 300):
            raise ValueError('Require 1..20 steps and 0..300 simulation seconds')
        import yaml
        import numpy as np
        from scripts.factory_closed_loop import run_closed_loop
        from scripts.model_worker_protocol import wait_json, write_json, exchange_request
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        run = ROOT/'validation/trajectory_runs'/config['source_run']
        metadata = json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
        with (run/'trajectory.jsonl').open(encoding='utf-8') as stream:
            initial = json.loads(next(stream))
        report.update(source_recording=str(run), source_metadata_sha256=sha(run/'run_metadata.json'),
                      source_trajectory_sha256=sha(run/'trajectory.jsonl'),
                      source_stage_sha256=metadata['stage_sha256'], instruction=metadata['task']['instruction'])
        from scripts.episode_replay_source import validate_recorded_scene_assets
        validate_recorded_scene_assets(metadata)
        source_config = Path(metadata['scene_config_path'])
        if sha(source_config) != metadata['scene_config_sha256']:
            raise ValueError('Recorded scene configuration changed')
        sources = ['run_factory_closed_loop', 'factory_closed_loop', 'aerovla_worker', 'model_worker_protocol',
                   'aerovla_offline_probe', 'aerovla_action_bridge', 'collision_scene', 'continuous_collision',
                   'drone_envelope', 'replay_pose', 'drone_control_math', 'evaluate_factory_inspection',
                   'collision_path_cursor']
        report['implementation_sha256'] = {f'scripts/{name}.py': sha(ROOT/'scripts'/f'{name}.py') for name in sources}
        if not args.model_python.is_file():
            raise ValueError('Independent model Python is unavailable')
        # Refuse to add a second simulation+model beside another GPU computation.
        # The operator can rerun after their replay completes; no process is stopped.
        gpu_processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                       capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        report['preflight_gpu_processes'] = gpu_processes
        if gpu_processes:
            raise RuntimeError('GPU is already in use by a compute process; finish its work before this trial')
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'width': 160, 'height': 128})
        import omni.replicator.core as rep
        import omni.timeline
        from PIL import Image
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.replay_pose import FrozenPoseRestorer, world_matrix, compare_pose_matrices
        from scripts.drone_control_math import root_orientation_wxyz_from_yaw
        from scripts.collision_scene import build_collision_scene, check_path
        from scripts.aerovla_action_bridge import plan_aerovla_action
        from scripts.drone_envelope import measure_drone_envelope
        from scripts.collision_path_cursor import CollisionPathCursor
        scene_config = load_scene_config(source_config)
        preview, _ = open_preview(app, scene_config=scene_config, start_controller=False, show_ui=False,
                                 width=160, height=128, fast=True, renderer='RayTracedLighting',
                                 disable_external_viewport=True, defer_thermal=True)
        restorer = FrozenPoseRestorer(preview.stage, scene_config)
        restorer.restore(initial)
        envelope = measure_drone_envelope(preview.stage, scene_config.drone.prim_path, scene_config.drone.body_path)
        report['composed_drone_envelope'] = envelope
        if envelope['rotation_envelope_radius_m'] >= config['body_sphere_radius_m']:
            raise ValueError('Configured collision sphere is smaller than the composed vehicle')
        scene = build_collision_scene(preview.stage, start=initial['position'], bounds=config['trial_bounds_m'],
                                      radius=config['body_sphere_radius_m'], drone_root=scene_config.drone.prim_path)
        report['collision_scene'] = scene.report
        if not scene.report['ready']:
            raise ValueError('Collision coverage or exterior start is not ready')
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        clock = float(initial['simulation_time'])
        timeline.set_current_time(clock)
        body_prim = preview.stage.GetPrimAtPath(scene_config.drone.body_path)
        camera_prim = preview.stage.GetPrimAtPath(preview.mount.rgb_path)
        camera_in_body = world_matrix(camera_prim) @ np.linalg.inv(world_matrix(body_prim))
        actual_history = [list(initial['position'])]
        actual_cursor = CollisionPathCursor(scene)
        current = {'yaw': float(initial['yaw']), 'position': list(initial['position'])}
        (args.out/'observations').mkdir()
        trajectory_path = args.out/'trajectory.jsonl'

        def observe(index):
            if abort_requested():
                raise KeyboardInterrupt()
            before = world_matrix(body_prim)
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=True, wait_for_render=True)
            image = preview.publish_rgb()
            if image is None or image.shape[:2] != (128, 160):
                raise ValueError('Fresh renderer RGB is unavailable')
            body, camera = world_matrix(body_prim), world_matrix(camera_prim)
            follow = compare_pose_matrices(camera, camera_in_body @ body)
            drift = compare_pose_matrices(body, before)
            if any(errors['translation_error_m'] > 1e-4 or errors['rotation_error_degrees'] > .01
                   for errors in (follow, drift)):
                raise ValueError('Body drifted or camera detached while rendering')
            if abs(float(timeline.get_current_time())-clock) > 1e-6:
                raise ValueError('Render advanced the paused simulation clock')
            path = args.out/'observations'/f'{index:03d}.png'
            if path.exists():
                raise FileExistsError(path)
            Image.fromarray(image[..., :3]).save(path)
            observation = {'index': index, 'simulation_time_s': clock,
                           'position': body[3, :3].tolist(), 'yaw': current['yaw'],
                           'body_world_matrix': body.tolist(), 'camera_world_matrix': camera.tolist(),
                           'camera_follow_errors': follow, 'body_render_drift': drift,
                           'rgb_path': str(path), 'rgb_sha256': sha(path)}
            write_json(path.with_suffix('.json'), observation)
            print(f'OBSERVATION {index} time={clock:.3f}', flush=True)
            return observation

        def infer(observation, instruction):
            response = exchange_request(exchange, observation, instruction, timeout_s=180,
                                       alive=lambda: worker.poll() is None, should_abort=abort_requested)
            print(f'MODEL {observation["index"]}: {response.get("raw_output", response.get("error"))}', flush=True)
            return response

        def execute(plan, index):
            nonlocal clock
            origin = clock
            first_row = len(trajectory)
            maximum = {'translation_error_m': 0., 'rotation_error_degrees': 0.}
            with trajectory_path.open('a', encoding='utf-8') as stream:
                for pose in plan['poses']:
                    if abort_requested():
                        raise KeyboardInterrupt()
                    desired, _ = restorer.restore({'position': pose['position'],
                        'orientation': root_orientation_wxyz_from_yaw(pose['yaw'])})
                    timeline.set_current_time(origin+pose['time_s'])
                    app.update()
                    actual = world_matrix(body_prim)
                    errors = compare_pose_matrices(actual, desired)
                    row = {'step': index, 'simulation_time_s': float(timeline.get_current_time()),
                           'position': actual[3, :3].tolist(), 'body_world_matrix': actual.tolist(),
                           'commanded_yaw': pose['yaw'], 'pose_errors': errors}
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    trajectory.append(row)
                    report['steps'][index]['partial_execution'] = {'last_pose': row,
                                                                   'collision_check': {'status': 'NOT_CHECKED'}}
                    if errors['translation_error_m'] > 1e-4 or errors['rotation_error_degrees'] > .01:
                        raise ValueError('Actual body did not follow the planned pose')
                    # Each actual segment is checked, anchored at the certified start.
                    clearance = actual_cursor.extend([actual_history[-1], row['position']])
                    actual_history.append(row['position'])
                    report['steps'][index]['partial_execution'] = {'last_pose': row, 'collision_check': clearance}
                    if clearance['status'] != 'CLEAR':
                        clock = row['simulation_time_s']
                        current.update(position=row['position'], yaw=pose['yaw'])
                        return {'simulation_time_s': clock, **current, 'max_pose_errors': maximum,
                                'actual_collision_check': clearance, 'trajectory_rows': len(trajectory)-first_row}
                    for key in maximum:
                        maximum[key] = max(maximum[key], errors[key])
            clock = origin+plan['poses'][-1]['time_s']
            current.update(position=actual_history[-1], yaw=plan['poses'][-1]['yaw'])
            return {'simulation_time_s': clock, **current, 'max_pose_errors': maximum,
                    'actual_collision_check': clearance, 'trajectory_rows': len(plan['poses'])}

        # Complete rendering warmup before loading the model into shared VRAM.
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=True, wait_for_render=True)
        env = os.environ.copy()
        for key in ['PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP']:
            env.pop(key, None)
        env['PYTHONNOUSERSITE'] = '1'
        worker_log = (args.out/'model-worker.log').open('w', encoding='utf-8')
        worker = subprocess.Popen([str(args.model_python), '-B', '-X', 'utf8', str(ROOT/'scripts/aerovla_worker.py'),
                         '--exchange', str(exchange), '--base', str(ROOT/'assets/models/aerovla-pilot/openvla-7b'),
                         '--adapter', str(ROOT/'assets/models/aerovla-pilot/aerovla/aero_vla'),
                         '--max-requests', str(args.max_steps)], cwd=ROOT, env=env,
                         stdout=worker_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        ready = wait_json(exchange/'ready.json', timeout_s=300, alive=lambda: worker.poll() is None,
                          should_abort=abort_requested)
        report['model_worker'] = ready
        if ready['status'] != 'PASS' or ready['model_load_count'] != 1:
            raise ValueError(f'Model worker failed to load: {ready.get("error")}')
        persist()
        run_closed_loop(report=report, observe=observe, infer=infer, execute=execute,
                        guard=lambda path: check_path(scene, path), planner=plan_aerovla_action,
                        instruction=metadata['task']['instruction'], motion=config['motion'],
                        max_steps=args.max_steps, max_sim_seconds=args.max_sim_seconds,
                        should_abort=abort_requested, persist=persist)
        validate_recorded_scene_assets(metadata)
        if report['source_metadata_sha256'] != sha(run/'run_metadata.json') or report['source_trajectory_sha256'] != sha(run/'trajectory.jsonl'):
            raise ValueError('Original source recording changed during trial')
    except KeyboardInterrupt:
        report.update(result='INCOMPLETE_OR_FAILED_FEEDBACK', termination={'reason': 'manual_abort'})
    except Exception as error:
        report.update(result='INCOMPLETE_OR_FAILED_FEEDBACK',
                      termination={'reason': 'system_error', 'detail': f'{type(error).__name__}: {error}'})
        (args.out/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        if worker is not None:
            if exchange.exists() and not (exchange/'stop.json').exists():
                from scripts.model_worker_protocol import write_json
                write_json(exchange/'stop.json', {})
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Only the process created by this invocation is terminated.
                worker.terminate()
                worker.wait(timeout=15)
            report['worker_exit_code'] = worker.returncode
        if worker_log is not None:
            worker_log.close()
        report['trajectory_rows'] = len(trajectory)
        persist()
        if preview is not None:
            preview.close()
        if app is not None:
            app.close()
    print(json.dumps({'result': report['result'], 'termination': report.get('termination'),
                      'steps': len(report['steps'])}), flush=True)
    return 0 if report['result'] == 'PASS_BOUNDED_KINEMATIC_FEEDBACK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
