"""Factory evaluator contract tests; no Isaac/Kit process is required."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

try:
    evaluator = importlib.import_module('scripts.evaluate_factory_inspection')
except ModuleNotFoundError:
    evaluator = None


REQUIREMENTS = {
    'minimum_equipment_distance_m': 5.0,
    'observation_distance_range_m': [6.0, 8.0],
    'hold_duration_s': 6.0,
    'reference_length_range_m': [180.0, 220.0],
    'body_envelope_radius_m': 0.18,
    'maximum_sample_gap_s': 0.25,
    'maximum_observation_speed_m_s': 0.3,
}
TRAJECTORY_HASH = '1' * 64
CONFIG_HASH = '2' * 64
HOTSPOT = '/World/Equipment/Hotspot'


def task_snapshot():
    return {
        'task_id': 'factory-inspection', 'instruction': 'Find the hot component and inspect it.',
        'target_prim_path': HOTSPOT,
        'environment_evidence': {'factory_inspection': {
            'equipment_prim_paths': ['/World/Equipment'], 'hotspot_prim_path': HOTSPOT,
            'geometry_file': 'factory/equipment.npz', 'ground_world_z_m': 0.0,
            'reference_waypoints_m': [[7.18, -200.0, 1.0], [7.18, 0.0, 1.0]],
            'reference_path_length_m': 200.0, 'observation_position_m': [7.18, 0.0, 1.0],
            'inspection_requirements': copy.deepcopy(REQUIREMENTS),
        }},
    }


def records(duration=6.0, distance=7.0):
    # An independently known 200 m approach, then a hold sampled every 0.1 s.
    result = []
    times = [0.0, 0.1] + [round(0.2 + i / 10, 8) for i in range(round(duration * 10) + 1)]
    for frame_id, time in enumerate(times):
        matrix = np.eye(4)
        position = [distance + 0.18, -200.0 if frame_id == 0 else 0.0, 1.0]
        matrix[3, :3] = position
        result.append({'frame_id': frame_id, 'simulation_time': time, 'position': position,
                       'equipment_distance_m': distance, 'linear_velocity': [0.0, 0.0, 0.0],
                       'rgb_camera_world_matrix': matrix.tolist(),
                       'scene_config_sha256': CONFIG_HASH})
    return result


def run_metadata(source='manual_driving'):
    return {'trajectory_source': source, 'success': True, 'success_marked_by': 'keyboard:T',
            'task_result': {'status': 'success', 'marked_by': 'keyboard:T'},
            'scene_config_sha256': CONFIG_HASH}


def evidence(samples):
    return [{'frame_id': row['frame_id'], 'simulation_time': row['simulation_time'],
             'trajectory_sha256': TRAJECTORY_HASH, 'scene_config_sha256': CONFIG_HASH,
             'hotspot_prim_path': HOTSPOT, 'hotspot_visible': True, 'unobstructed': True,
             'camera_world_matrix': row['rgb_camera_world_matrix'], 'evidence_image': 'frame.png'}
            for row in samples]


class FactoryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evaluator, 'factory evaluator has not been implemented')

    def evaluate(self, samples=None, visual=None, metadata=None, task=None, **extra):
        samples = records() if samples is None else samples
        return evaluator.evaluate_samples(
            samples, REQUIREMENTS, task=task or task_snapshot(),
            run_metadata=metadata or run_metadata(),
            visual_evidence=evidence(samples) if visual is None else visual,
            trajectory_sha256=TRAJECTORY_HASH, scene_config_sha256=CONFIG_HASH,
            verified_image_paths={'frame.png'}, **extra)

    def test_exact_triangle_distance_accounts_for_edges_and_body_radius(self):
        distances = evaluator.point_to_mesh_distances(
            np.array([[0.25, 0.25, 5.08], [2.0, 0.0, 0.0]]),
            np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
            np.array([[0, 1, 2]]), body_radius_m=0.18)
        np.testing.assert_allclose(distances, [4.9, 0.82], atol=1e-12)

    def test_degenerate_triangle_uses_segment_distance(self):
        distances = evaluator.point_to_mesh_distances(
            [[0.5, 2., 0.]], [[0., 0., 0.], [1., 0., 0.], [1., 0., 0.]], [[0, 1, 2]])
        self.assertAlmostEqual(distances[0], 2.0)

    def test_six_seconds_with_manual_visual_evidence_passes(self):
        result = self.evaluate()
        self.assertEqual(result['result'], 'PASS')
        self.assertTrue(result['distance_and_hold_pass'])
        self.assertEqual(result['actual_path_length_m'], 200.0)
        self.assertEqual(result['reference_path_length_m'], 200.0)
        self.assertAlmostEqual(result['verified_observation_duration_s'], 6.0)

    def test_clearance_4_9_fails_even_if_marked_success(self):
        samples = records()
        samples[0]['equipment_distance_m'] = 4.9
        result = self.evaluate(samples)
        self.assertEqual(result['result'], 'FAIL')
        self.assertFalse(result['safety_pass'])

    def test_free_search_does_not_require_flown_or_reference_mileage(self):
        samples=records()
        for row in samples:row['position'][1]=0.
        task=task_snapshot()
        task['environment_evidence']['factory_inspection'].pop('reference_waypoints_m')
        requirements={**REQUIREMENTS,'require_path_length':False}
        result=evaluator.evaluate_samples(samples,requirements,task=task,
            run_metadata=run_metadata(),visual_evidence=evidence(samples),
            trajectory_sha256=TRAJECTORY_HASH,scene_config_sha256=CONFIG_HASH,
            verified_image_paths={'frame.png'})
        self.assertEqual(result['result'],'PASS')
        self.assertEqual(result['actual_path_length_m'],0.)
        self.assertFalse(result['path_length_required'])
        samples[0]['equipment_distance_m']=4.9
        result=evaluator.evaluate_samples(samples,requirements,task=task)
        self.assertEqual(result['result'],'FAIL')

    def test_mileage_switch_requires_actual_boolean(self):
        with self.assertRaises(ValueError):
            evaluator.evaluate_samples(records(),{**REQUIREMENTS,'require_path_length':'false'})

    def test_observation_distance_6_and_8_are_inclusive(self):
        for distance in (6.0, 8.0):
            with self.subTest(distance=distance):
                self.assertEqual(self.evaluate(records(distance=distance))['result'], 'PASS')

    def test_5_9_seconds_is_insufficient(self):
        self.assertEqual(self.evaluate(records(duration=5.9))['result'], 'FAIL')

    def test_leaving_range_resets_hold(self):
        samples = records()
        samples[32]['equipment_distance_m'] = 8.1
        result = self.evaluate(samples)
        self.assertEqual(result['result'], 'FAIL')
        self.assertLess(result['verified_observation_duration_s'], 3.2)

    def test_out_of_frame_or_occluded_resets_hold(self):
        for field in ('hotspot_visible', 'unobstructed'):
            with self.subTest(field=field):
                samples = records()
                visual = evidence(samples)
                visual[32][field] = False
                result = self.evaluate(samples, visual)
                self.assertEqual(result['result'], 'FAIL')
                self.assertLess(result['verified_observation_duration_s'], 3.2)

    def test_gap_above_maximum_resets_hold(self):
        samples = records()
        for row in samples[32:]:
            row['simulation_time'] += 0.2
        self.assertEqual(self.evaluate(samples)['result'], 'FAIL')

    def test_missing_trajectory_frame_resets_hold(self):
        samples = records()
        del samples[32]
        self.assertEqual(self.evaluate(samples)['result'], 'FAIL')

    def test_missing_visual_frame_resets_hold(self):
        samples = records()
        visual = evidence(samples)
        del visual[32]
        self.assertEqual(self.evaluate(samples, visual)['result'], 'UNJUDGED')

    def test_missing_visual_evidence_cannot_pass(self):
        result = self.evaluate(visual=[])
        self.assertEqual(result['result'], 'UNJUDGED')
        self.assertEqual(result['visual_evidence']['status'], 'UNJUDGED')

    def test_velocity_cannot_claim_hover_while_positions_move(self):
        samples = records()
        for row in samples[2:]:
            row['position'][1] = row['simulation_time'] * 0.4
        result = self.evaluate(samples)
        self.assertEqual(result['result'], 'FAIL')

    def test_recorded_velocity_over_limit_resets_hold(self):
        samples = records()
        samples[32]['linear_velocity'] = [0.31, 0., 0.]
        self.assertEqual(self.evaluate(samples)['result'], 'FAIL')

    def test_duplicate_or_nonfinite_time_and_nonfinite_pose_are_invalid(self):
        for value in (float('nan'), float('inf'), 3.1):
            with self.subTest(time=value):
                samples = records()
                samples[32]['simulation_time'] = value
                self.assertEqual(self.evaluate(samples)['result'], 'FAIL')
        samples = records()
        samples[32]['position'][0] = float('nan')
        self.assertEqual(self.evaluate(samples)['result'], 'FAIL')

    def test_actual_and_reference_length_are_independently_checked(self):
        samples = records()
        samples[0]['position'][1] = -179.
        self.assertEqual(self.evaluate(samples)['result'], 'FAIL')
        task = task_snapshot()
        inspection = task['environment_evidence']['factory_inspection']
        inspection['reference_waypoints_m'][0][1] = -221.
        inspection['reference_path_length_m'] = 221.
        self.assertEqual(self.evaluate(task=task)['result'], 'FAIL')

    def test_reference_declared_length_cannot_replace_actual_waypoint_length(self):
        task = task_snapshot()
        task['environment_evidence']['factory_inspection']['reference_waypoints_m'][0][1] = -50.
        result = self.evaluate(task=task)
        self.assertEqual(result['result'], 'FAIL')
        self.assertEqual(result['reference_path_length_m'], 50.)

    def test_synthetic_source_cannot_be_manual_even_with_forged_success(self):
        result = self.evaluate(metadata=run_metadata('synthetic_reference'))
        self.assertEqual(result['result'], 'UNJUDGED')
        self.assertFalse(result['manual_confirmation']['confirmed'])
        self.assertEqual(result['qualification'], 'development')

    def test_no_keyboard_mark_is_unjudged(self):
        metadata = run_metadata()
        metadata['task_result'] = {'status': 'unjudged'}
        metadata['success'] = None
        result = self.evaluate(metadata=metadata)
        self.assertEqual(result['result'], 'UNJUDGED')
        self.assertFalse(result['manual_confirmation']['confirmed'])

    def test_explicit_unjudged_result_cannot_reuse_stale_success_flag(self):
        metadata = run_metadata()
        metadata['task_result'] = {'status': 'unjudged'}
        result = self.evaluate(metadata=metadata)
        self.assertFalse(result['manual_confirmation']['confirmed'])
        self.assertEqual(result['result'], 'UNJUDGED')

    def test_explicit_manual_failure_is_retained(self):
        metadata = run_metadata()
        metadata['success'] = False
        metadata['task_result'] = {'status': 'failed', 'marked_by': 'keyboard:T'}
        result = self.evaluate(metadata=metadata)
        self.assertEqual(result['manual_confirmation']['status'], 'FAIL')
        self.assertEqual(result['result'], 'FAIL')

    def test_visual_hash_time_camera_and_target_must_match_record(self):
        for field, value in (
                ('trajectory_sha256', '9' * 64), ('scene_config_sha256', '9' * 64),
                ('simulation_time', 999.), ('hotspot_prim_path', '/World/Wrong'),
                ('camera_world_matrix', np.eye(4).tolist()), ('evidence_image', 'missing.png')):
            with self.subTest(field=field):
                samples = records()
                visual = evidence(samples)
                for row in visual:
                    row[field] = value
                self.assertEqual(self.evaluate(samples, visual)['result'], 'UNJUDGED')

    def test_duplicate_visual_rows_cannot_certify_hold(self):
        samples = records()
        visual = evidence(samples)
        visual.append(copy.deepcopy(visual[32]))
        self.assertEqual(self.evaluate(samples, visual)['result'], 'UNJUDGED')

    def test_unknown_mesh_clearance_stays_unjudged_even_with_visual_rows(self):
        samples = records()
        for row in samples:
            del row['equipment_distance_m']
        result = self.evaluate(samples)
        self.assertEqual(result['result'], 'UNJUDGED')
        self.assertEqual(result['safety_status'], 'UNJUDGED')

    def test_configurable_speed_limit_is_applied_to_observation(self):
        samples = records()
        for row in samples:
            row['linear_velocity'] = [0.27, 0., 0.]
        requirements = copy.deepcopy(REQUIREMENTS)
        requirements['maximum_observation_speed_m_s'] = 0.25
        result = evaluator.evaluate_samples(samples, requirements, task=task_snapshot(),
                                            run_metadata=run_metadata())
        self.assertEqual(result['result'], 'FAIL')

    def test_an_earlier_interrupted_hold_can_be_followed_by_valid_six_seconds(self):
        samples = records(duration=10.0)
        samples[32]['equipment_distance_m'] = 8.1
        result = self.evaluate(samples)
        self.assertEqual(result['result'], 'PASS')
        self.assertGreaterEqual(result['verified_observation_duration_s'], 6.)

    def test_exact_mesh_api_does_not_use_precomputed_distance(self):
        samples = records()
        for row in samples:
            row['position'][0] = 5.08  # True body clearance is 4.9, record falsely claims 7.
        result = evaluator.evaluate_records(
            samples, task_snapshot(), np.array([[0., -1000., -100.], [0., 1000., -100.],
                                                [0., 1000., 100.], [0., -1000., 100.]]),
            np.array([[0, 1, 2], [0, 2, 3]]), run_metadata=run_metadata())
        self.assertFalse(result['safety_pass'])
        self.assertEqual(result['result'], 'FAIL')


class FactoryEvaluationCLITests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evaluator, 'factory evaluator has not been implemented')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.stage = self.root / 'factory.usda'
        self.stage.write_text('#usda 1.0\n', encoding='utf-8')
        for name in ('thermal.yaml', 'sensor.yaml'):
            (self.root / name).write_text('{}', encoding='utf-8')
        self.task_file = self.root / 'task.yaml'
        self.task_file.write_text('task_id: changed-today\ninstruction: changed today', encoding='utf-8')
        self.config = self.root / 'scene.yaml'
        self.config.write_text(yaml.safe_dump({
            'schema_version': 1, 'scene': {'stage': 'factory.usda', 'thermal_scene': 'thermal.yaml'},
            'drone': {'prim_path': '/World/Drone', 'body_path': '/World/Drone/body',
                      'camera_path': '/World/Drone/body/Camera', 'minimum_height': 0.5},
            'sensors': {'width': 160, 'height': 128, 'sensor_config': 'sensor.yaml',
                        'thermal_config': 'thermal.yaml'}, 'task': {'file': 'task.yaml'},
        }), encoding='utf-8')
        self.config_hash = hashlib.sha256(self.config.read_bytes()).hexdigest()
        task = task_snapshot()
        task['environment_evidence']['factory_inspection']['geometry_file'] = 'equipment.npz'
        np.savez(self.root / 'equipment.npz',
                 equipment_vertices=[[0., -1000., -100.], [0., 1000., -100.],
                                     [0., 1000., 100.], [0., -1000., 100.]],
                 equipment_triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
        self.metadata = run_metadata()
        self.metadata.update({'task': task, 'scene_config_sha256': self.config_hash,
                              'stage_sha256': hashlib.sha256(self.stage.read_bytes()).hexdigest(),
                              'task_config_sha256': '3' * 64})
        samples = records()
        for row in samples:
            row['scene_config_sha256'] = self.config_hash
        self.samples = samples
        self.trajectory = self.run / 'trajectory.jsonl'
        self.trajectory.write_text(''.join(json.dumps(row) + '\n' for row in samples), encoding='utf-8')
        self.write_metadata()

    def write_metadata(self):
        (self.run / 'run_metadata.json').write_text(json.dumps(self.metadata), encoding='utf-8')

    def invoke(self, suffix='out', visual=None):
        args = ['--run', str(self.run), '--config', str(self.config), '--out', str(self.root / suffix),
                '--repository-root', str(self.root)]
        if visual:
            args += ['--visual-evidence', str(visual)]
        return evaluator.main(args)

    def test_cli_preserves_recorded_task_and_does_not_read_current_task(self):
        self.assertEqual(self.invoke(), 0)
        result = json.loads((self.root / 'out' / 'factory_evaluation.json').read_text(encoding='utf-8'))
        self.assertEqual(result['task_id'], 'factory-inspection')
        self.assertEqual(result['result'], 'UNJUDGED')
        self.assertEqual(result['task_provenance'], 'recorded_run_snapshot')

    def test_cli_existing_output_is_never_overwritten(self):
        (self.root / 'out').mkdir()
        sentinel = self.root / 'out' / 'keep.txt'
        sentinel.write_text('untouched', encoding='utf-8')
        self.assertNotEqual(self.invoke(), 0)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'untouched')

    def test_cli_missing_geometry_is_unjudged(self):
        (self.root / 'equipment.npz').unlink()
        self.assertEqual(self.invoke(), 0)
        report = json.loads((self.root / 'out' / 'factory_evaluation.json').read_text(encoding='utf-8'))
        self.assertEqual(report['result'], 'UNJUDGED')
        self.assertEqual(report['provenance']['geometry']['status'], 'UNJUDGED')

    def test_cli_geometry_path_cannot_escape_repository(self):
        self.metadata['task']['environment_evidence']['factory_inspection']['geometry_file'] = '../outside.npz'
        self.write_metadata()
        self.assertNotEqual(self.invoke(), 0)
        self.assertFalse((self.root / 'out').exists())

    def test_cli_refuses_altered_config_or_stage_and_existing_output(self):
        self.config.write_text(self.config.read_text(encoding='utf-8') + '\n# modified', encoding='utf-8')
        self.assertNotEqual(self.invoke(), 0)
        self.assertFalse((self.root / 'out').exists())
        self.metadata['scene_config_sha256'] = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.write_metadata()
        self.stage.write_text('# changed stage', encoding='utf-8')
        self.assertNotEqual(self.invoke('out2'), 0)
        self.assertFalse((self.root / 'out2').exists())

    def test_cli_checks_geometry_sha_and_trajectory_hash(self):
        inspection = self.metadata['task']['environment_evidence']['factory_inspection']
        inspection['geometry_sha256'] = '0' * 64
        self.write_metadata()
        self.assertNotEqual(self.invoke(), 0)
        del inspection['geometry_sha256']
        self.metadata['trajectory_sha256'] = '0' * 64
        self.write_metadata()
        self.assertNotEqual(self.invoke('out2'), 0)

    def test_cli_visual_file_requires_existing_image_and_matching_hash(self):
        image = self.root / 'frame.png'
        image.write_bytes(b'known-existing-evidence-image')
        visual = evidence(self.samples)
        trajectory_hash = hashlib.sha256(self.trajectory.read_bytes()).hexdigest()
        for row in visual:
            row['trajectory_sha256'] = trajectory_hash
            row['scene_config_sha256'] = self.config_hash
        path = self.root / 'visual.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in visual), encoding='utf-8')
        self.assertEqual(self.invoke(visual=path), 0)
        result = json.loads((self.root / 'out' / 'factory_evaluation.json').read_text(encoding='utf-8'))
        self.assertEqual(result['result'], 'PASS')
        image.unlink()
        self.assertEqual(self.invoke('out2', path), 0)
        result = json.loads((self.root / 'out2' / 'factory_evaluation.json').read_text(encoding='utf-8'))
        self.assertEqual(result['result'], 'UNJUDGED')


if __name__ == '__main__':
    unittest.main()
