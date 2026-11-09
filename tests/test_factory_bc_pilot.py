"""Pilot launch contracts: exercise real configuration and argument consumers."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.factory_bc_pilot import build_launch, main, validate_recording
from scripts.scene_config import load_scene_config
from scripts.start_crazyflie_dual_camera_view import preview_motion_config


class FactoryBCPilotTests(unittest.TestCase):
    def test_runtime_loads_separate_task_with_observation_not_mileage_gate(self):
        base = load_scene_config(ROOT / 'configs/model02_v7m_flight_scene.yaml')
        pilot = load_scene_config(ROOT / 'configs/model02_v7m_bc_pilot_scene.yaml')
        self.assertEqual(pilot.scene, base.scene)
        self.assertEqual(pilot.sensors, base.sensors)
        self.assertNotEqual(pilot.task['task_id'], base.task['task_id'])
        req = pilot.task['environment_evidence']['factory_inspection']['inspection_requirements']
        self.assertEqual(req['observation_distance_range_m'], [6.0, 8.0])
        self.assertEqual(req['hold_duration_s'], 3.0)
        self.assertFalse(req['require_path_length'])
        self.assertNotIn('环绕', pilot.task['instruction'])
        self.assertNotIn('撤离', pilot.task['instruction'])
        self.assertNotIn('/World', pilot.task['instruction'])

    def test_launch_passes_speeds_to_real_motion_configuration(self):
        launch = build_launch('factory-bc-pilot-contract-001')
        args = launch['preview_args']
        value = lambda flag: args[args.index(flag) + 1]
        motion = preview_motion_config(float(value('--speed-scale')),
                                       float(value('--horizontal-speed')),
                                       float(value('--vertical-speed')))
        self.assertEqual(motion.max_horizontal_speed, 3.0)
        self.assertEqual(motion.max_vertical_speed, 1.0)
        self.assertAlmostEqual(motion.max_yaw_speed, math.pi / 4)
        self.assertEqual(value('--trajectory-period'), '1.0')
        self.assertEqual(launch['trajectory_period_s'], 1.)
        self.assertEqual(value('--preview-fps'), '15')
        self.assertEqual(value('--thermal-preview-fps'), '2')
        self.assertIn('--record-trajectory', args)
        self.assertIn('--no-save', args)
        self.assertNotIn('--probe-control', args)
        self.assertNotIn('--headless', args)

    def test_smoke_cannot_be_mislabelled_as_manual_demonstration(self):
        with self.assertRaises(ValueError):
            build_launch('factory-bc-pilot-contract-001', smoke_frames=3)
        launch = build_launch('factory-bc-probe-contract-001', smoke_frames=3)
        self.assertIn('--probe-control', launch['preview_args'])
        self.assertIn('--headless', launch['preview_args'])
        self.assertEqual(launch['trajectory_source'], 'synthetic_control_probe')

    def test_rejects_bad_run_ids_before_launch(self):
        for bad in ('../outside', 'C:/outside', 'hello world', 'factory;cmd', '', 'CON'):
            with self.subTest(run_id=bad), self.assertRaises(ValueError):
                build_launch(bad)

    def test_refuses_existing_recording_without_changing_it(self):
        parent = ROOT / 'validation/trajectory_runs'
        with tempfile.TemporaryDirectory(prefix='factory-bc-pilot-test-', dir=parent) as folder:
            marker = Path(folder) / 'do-not-overwrite.txt'
            marker.write_text('original', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                build_launch(Path(folder).name)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'original')

    def test_check_only_reports_contract_without_creating_recording(self):
        run_id = 'factory-bc-pilot-checkonly-contract'
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(['--run-id', run_id, '--check-only'])
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['start_group'], 'A')
        self.assertEqual(report['controller_parameters']['max_horizontal_speed'], 3.0)
        self.assertFalse((ROOT / 'validation/trajectory_runs' / run_id).exists())

    def test_recorded_runtime_must_match_contract_and_cannot_claim_task_pass(self):
        launch = {'task_id': 'factory_bc_tank_hotspot_pilot_v1',
                  'scene_config_sha256': 'scene-hash', 'task_config_sha256': 'task-hash',
                  'trajectory_source': 'manual_driving',
                  'expected_start_position_m': [700.0, -845.0, 8.5],
                  'controller_parameters': {'max_horizontal_speed': 1.5,
                      'max_vertical_speed': 0.5, 'max_yaw_speed': math.pi / 4}}
        metadata = {'task': {'task_id': 'factory_bc_tank_hotspot_pilot_v1'},
                    'scene_config_sha256': 'scene-hash', 'task_config_sha256': 'task-hash',
                    'trajectory_source': 'manual_driving', 'start_position': [700.0, -845.0, 8.5],
                    'record_count': 4,
                    'controller_parameters': {'max_horizontal_speed': 1.5,
                        'max_vertical_speed': 0.5, 'max_yaw_speed': math.pi / 4},
                    'task_result': {'status': 'unjudged'}}
        result = validate_recording(metadata, launch)
        self.assertTrue(result['runtime_contract_verified'])
        self.assertEqual(result['task_acceptance'], 'NOT_EVALUATED')
        for field, wrong in (('max_horizontal_speed', 7.5), ('max_vertical_speed', 3.0)):
            bad = copy.deepcopy(metadata)
            bad['controller_parameters'][field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_recording(bad, launch)
        for field, wrong in (('start_position', [0, 0, 0]),
                             ('start_position', [float('nan'), -845, 8.5]),
                             ('trajectory_source', 'synthetic_control_probe'),
                             ('record_count', 0)):
            bad = copy.deepcopy(metadata)
            bad[field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_recording(bad, launch)

    def test_parent_checks_saved_metadata_even_when_preview_child_exits_process(self):
        self._exercise_exiting_child('completed', expected_exit=0)

    def test_parent_rejects_failed_preview_even_when_child_exit_code_is_zero(self):
        self._exercise_exiting_child('failed', expected_exit=2)

    def _exercise_exiting_child(self, completion, expected_exit):
        launch = build_launch('factory-bc-pilot-parent-contract')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, output = root / 'run', root / 'preview'
            entrypoint = root / 'exit_preview.py'
            launch.update(run_dir=str(run), preview_output=str(output),
                          launch_manifest_path=str(root / 'launch.json'),
                          preview_entrypoint=str(entrypoint),scene_config_path=str(root/'runtime/scene.yaml'),
                          task_config_path=str(root/'runtime/task.yaml'))
            metadata = {'task': launch['task_snapshot'],
                        'start_yaw_radians': math.radians(launch['expected_start_yaw_degrees']),
                        'scene_config_sha256': launch['scene_config_sha256'],
                        'task_config_sha256': launch['task_config_sha256'],
                        'stage_sha256': launch['stage_sha256'],
                        'trajectory_source': 'manual_driving', 'start_position': [700.0, -845.0, 8.5],
                        'record_count': 2, 'controller_parameters': {'max_horizontal_speed': 3.0,
                            'max_vertical_speed': 1.0, 'max_yaw_speed': math.pi / 4}}
            matrix=[[float(i==j) for j in range(4)] for i in range(4)]
            rows=[{'frame_id':i,'simulation_time':i*.1,'recorded_body_world_matrix':matrix,
                   'rgb_camera_world_matrix':matrix,'thermal_camera_world_matrix':matrix} for i in range(2)]
            entrypoint.write_text(
                'import os\nfrom pathlib import Path\n'
                f'run=Path({str(run)!r}); run.mkdir()\n'
                f'out=Path({str(output)!r}); out.mkdir()\n'
                f'(run/"run_metadata.json").write_text({json.dumps(metadata)!r}, encoding="utf-8")\n'
                f'(run/"trajectory.jsonl").write_text({"".join(json.dumps(r)+chr(10) for r in rows)!r}, encoding="utf-8")\n'
                f'(out/"performance.json").write_text({json.dumps({"completion": completion})!r}, encoding="utf-8")\n'
                'os._exit(0)\n', encoding='utf-8')
            # The old in-process path must fail RED without starting Kit. The
            # fixed path runs a real child that writes its files then os._exit().
            with patch('scripts.factory_bc_pilot.build_launch', return_value=launch), \
                    patch('scripts.start_crazyflie_dual_camera_view.main', side_effect=SystemExit(0)):
                self.assertEqual(main(['--run-id', 'factory-bc-pilot-parent-contract']), expected_exit)
            check = run / 'pilot_runtime_check.json'
            if expected_exit == 0:
                self.assertTrue(json.loads(check.read_text(encoding='utf-8'))['runtime_contract_verified'])
            else:
                self.assertFalse(check.exists())
if __name__ == '__main__':
    unittest.main()
