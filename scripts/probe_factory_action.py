"""Validate known and recorded model actions in a guarded kinematic factory trial."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/factory_autonomous_pilot.yaml')
    parser.add_argument('--model-result', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'result': 'RUNNING', 'trajectory_source': 'kinematic_control_diagnostic',
              'autonomous_closed_loop': False, 'task_result': {'status': 'unjudged'}, 'cases': []}
    report['program_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    app = preview = None
    trajectory = []
    try:
        report['configuration_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        import yaml
        import numpy as np
        from isaacsim import SimulationApp
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        run = ROOT / 'validation/trajectory_runs' / config['source_run']
        metadata = json.loads((run / 'run_metadata.json').read_text(encoding='utf-8'))
        report['source_recording'] = str(run)
        report['source_metadata_sha256'] = hashlib.sha256((run / 'run_metadata.json').read_bytes()).hexdigest()
        report['source_stage_sha256'] = metadata['stage_sha256']
        initial = json.loads((run / 'trajectory.jsonl').read_text(encoding='utf-8').splitlines()[0])
        from scripts.episode_replay_source import validate_recorded_scene_assets
        validate_recorded_scene_assets(metadata)
        envelope = json.loads((args.config.parent / config['body_envelope_evidence']).read_text(encoding='utf-8'))
        if (envelope.get('result') != 'PASS' or len(envelope.get('rotor_bounds', [])) != 4
            or envelope['rotating_geometry_sphere_upper_bound_m'] >= config['body_sphere_radius_m']
            or hashlib.sha256(Path(envelope['source_stage']).read_bytes()).hexdigest() != envelope['stage_sha256']):
            raise ValueError('Drone envelope evidence is missing, changed or too large')
        cases = [('known_right_turn', {'forward_m': 0., 'down_m': 0., 'yaw_rad': .3}),
                 ('known_ascent', {'forward_m': 0., 'down_m': -.5, 'yaw_rad': 0.}),
                 ('known_forward', {'forward_m': .5, 'down_m': 0., 'yaw_rad': 0.})]
        if args.model_result:
            from scripts.aerovla_offline_probe import parse_action
            model = json.loads(args.model_result.read_text(encoding='utf-8'))
            action = parse_action(model['raw_output'].replace('</s>', ''))
            if (model.get('status') != 'PASS' or not model.get('inference_completed')
                or action != model['action'] or model['instruction'] != metadata['task']['instruction']
                or hashlib.sha256(Path(model['rgb']).read_bytes()).hexdigest() != model['rgb_sha256']):
                raise ValueError('Model result or its input provenance is invalid')
            rgb_path = Path(model['rgb']).resolve()
            if not rgb_path.is_relative_to((ROOT / 'validation/episodes' / config['source_run']).resolve()):
                raise ValueError('Model image does not belong to the selected source run')
            from scripts.model_observation import validate_first_observation
            observation = validate_first_observation(run, rgb_path)
            report['model_observation_evidence'] = observation['evidence']
            if action['official_termination']:
                raise ValueError('Model requested termination; no motion to verify')
            cases.append(('recorded_model_action', action))
            report['model_result_sha256'] = hashlib.sha256(args.model_result.read_bytes()).hexdigest()
            report['model_input_mode'] = model['input_mode']
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
        source_config = Path(metadata['scene_config_path'])
        if hashlib.sha256(source_config.read_bytes()).hexdigest() != metadata['scene_config_sha256']:
            raise ValueError('Recorded scene configuration changed')
        scene_config = load_scene_config(source_config)
        preview, _ = open_preview(app, scene_config=scene_config, start_controller=False, show_ui=False,
                                 width=160, height=128, fast=True, renderer='RayTracedLighting',
                                 disable_external_viewport=True, defer_thermal=True)
        restorer = FrozenPoseRestorer(preview.stage, scene_config)
        restorer.restore(initial)
        actual = world_matrix(preview.stage.GetPrimAtPath(scene_config.drone.body_path))
        composed_envelope = measure_drone_envelope(preview.stage, scene_config.drone.prim_path, scene_config.drone.body_path)
        report['composed_drone_envelope'] = composed_envelope
        if composed_envelope['rotation_envelope_radius_m'] >= config['body_sphere_radius_m']:
            raise ValueError('Composed flight-stage drone exceeds the configured collision sphere')
        camera_in_body = world_matrix(preview.stage.GetPrimAtPath(preview.mount.rgb_path)) @ np.linalg.inv(actual)
        scene = build_collision_scene(preview.stage, start=initial['position'], bounds=config['trial_bounds_m'],
                                      radius=config['body_sphere_radius_m'], drone_root=scene_config.drone.prim_path)
        report['collision_scene'] = scene.report
        if not scene.report['ready']:
            raise ValueError('Collision scene or exterior start is not ready')
        timeline = omni.timeline.get_timeline_interface()
        clock = float(initial['simulation_time'])
        previous_path = None
        for name, action in cases:
            if previous_path:
                reset = check_path(scene, previous_path + [initial['position']])
                if reset['status'] != 'CLEAR':
                    raise ValueError('Reset between independent diagnostic cases was not clear')
                restorer.restore(initial)
            plan = plan_aerovla_action(initial['position'], initial['yaw'], action, **config['motion'])
            planned_path = [row['position'] for row in plan['poses']]
            clearance = check_path(scene, planned_path)
            if clearance['status'] != 'CLEAR':
                raise ValueError(f'Planned diagnostic motion blocked: {clearance}')
            case = {'name': name, 'action': action, 'planned_collision_check': clearance,
                    'phases': plan['phases'], 'translation_executed': plan['translation_executed'],
                    'max_translation_error_m': 0., 'max_rotation_error_degrees': 0.}
            actual_path = []
            for row in plan['poses']:
                desired, _ = restorer.restore({'position': row['position'],
                    'orientation': root_orientation_wxyz_from_yaw(row['yaw'])})
                timeline.set_current_time(clock + row['time_s'])
                app.update()
                actual = world_matrix(preview.stage.GetPrimAtPath(scene_config.drone.body_path))
                errors = compare_pose_matrices(actual, desired)
                if errors['translation_error_m'] > 1e-4 or errors['rotation_error_degrees'] > .01:
                    raise ValueError(f'Actual body motion differs from plan: {errors}')
                case['max_translation_error_m'] = max(case['max_translation_error_m'], errors['translation_error_m'])
                case['max_rotation_error_degrees'] = max(case['max_rotation_error_degrees'], errors['rotation_error_degrees'])
                position = actual[3, :3].tolist()
                actual_path.append(position)
                trajectory.append({'case': name, 'time_s': clock + row['time_s'], 'position': position,
                                   'body_world_matrix': actual.tolist(), 'planned_yaw': row['yaw']})
            actual_clearance = check_path(scene, actual_path)
            if actual_clearance['status'] != 'CLEAR':
                raise ValueError(f'Actual diagnostic path failed collision check: {actual_clearance}')
            case['actual_collision_check'] = actual_clearance
            case['end_position_m'] = actual_path[-1]
            case['end_yaw_rad'] = plan['poses'][-1]['yaw']
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=1, delta_time=0., pause_timeline=True, wait_for_render=True)
            image = preview.publish_rgb()
            if image.shape[:2] != (128, 160):
                raise ValueError('Actual end view is unavailable')
            image_path = args.out / (name + '.png')
            Image.fromarray(image[..., :3]).save(image_path)
            rendered_body = world_matrix(preview.stage.GetPrimAtPath(scene_config.drone.body_path))
            rendered_camera = world_matrix(preview.stage.GetPrimAtPath(preview.mount.rgb_path))
            camera_errors = compare_pose_matrices(rendered_camera, camera_in_body @ rendered_body)
            if camera_errors['translation_error_m'] > 1e-4 or camera_errors['rotation_error_degrees'] > .01:
                raise ValueError('End-view camera did not follow the actual body')
            case.update(rgb_path=str(image_path.resolve()), rgb_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                        rendered_body_world_matrix=rendered_body.tolist(), rendered_camera_world_matrix=rendered_camera.tolist(),
                        camera_follow_errors=camera_errors, observation_time_s=float(timeline.get_current_time()))
            report['cases'].append(case)
            previous_path = actual_path
            clock += plan['poses'][-1]['time_s'] + config['motion']['step_s']
        validate_recorded_scene_assets(metadata)
        report['result'] = 'PASS_KINEMATIC_ACTION_CHECKS'
        report['single_recorded_model_action_executed'] = args.model_result is not None
        report['scope'] = 'Guarded pose-driven control and rendered endpoint checks; no physics-response or closed-loop task certification'
    except Exception as error:
        report['result'] = 'FAIL'
        report['error'] = f'{type(error).__name__}: {error}'
        (args.out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        (args.out / 'trajectory.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in trajectory), encoding='utf-8')
        (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if preview is not None:
            preview.close()
        if app is not None:
            app.close()
    print(json.dumps({'result': report['result'], 'cases': len(report['cases']), 'error': report.get('error')}), flush=True)
    return 0 if report['result'] == 'PASS_KINEMATIC_ACTION_CHECKS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
