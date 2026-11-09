"""Offline cadence stays independent of the dense manual state recording."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


class OfflineCaptureContractTests(unittest.TestCase):
    def test_manual_defaults_record_one_hz_and_version_the_capture_contract(self):
        from scripts.collection_capture_defaults import capture_contract
        contract = capture_contract()
        self.assertEqual(contract['trajectory_period_s'], 1.)
        self.assertEqual(contract['contract_id'], 'paired_offline_v3')
        self.assertEqual(contract['offline_sample_interval_s'], 1.)
        self.assertEqual(contract['rgb_resolution'], [256, 256])
        self.assertEqual(contract['thermal_resolution'], [256, 256])
        self.assertEqual(contract['action_schema'], 'factory_pose_action_v2')
        self.assertEqual(contract['schema_version'], 1)

    def test_replay_contract_records_square_outputs_separately(self):
        from scripts.collection_capture_defaults import replay_capture_contract
        contract = replay_capture_contract(1.0)
        self.assertEqual(contract['contract_id'], 'paired_offline_v3')
        self.assertEqual(contract['offline_sample_interval_s'], 1.0)
        self.assertEqual(contract['rgb_resolution'], [256, 256])
        self.assertEqual(contract['thermal_resolution'], [256, 256])
        with self.assertRaises(ValueError):
            replay_capture_contract(0)
        custom = replay_capture_contract(rgb_size=(640, 512), thermal_size=(640, 512))
        self.assertEqual(custom['rgb_resolution'], [640, 512])
        self.assertEqual(custom['contract_id'], 'paired_offline_custom')

    def test_one_hz_source_keeps_exact_off_grid_t_and_original_bytes(self):
        from scripts.replay_trajectory_highres import prepare_replay_input
        from scripts.trajectory_recorder import read_trajectory, trajectory_provenance
        from scripts.collection_capture_defaults import capture_contract
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root/'raw'; run.mkdir()
            rows = [{'frame_id': i, 'simulation_time': float(i), 'position': [0., 0., 0.]}
                    for i in range(4)]
            terminal = dict(rows[2], frame_id=-1, simulation_time=2.35)
            for key in ('recorded_body_world_matrix', 'rgb_camera_world_matrix', 'thermal_camera_world_matrix'):
                terminal[key] = np.eye(4).tolist()
            contract = capture_contract()
            meta = {'sample_period_s': 1., 'capture_contract': contract,
                    'trajectory_source': 'manual_driving', 'terminal_snapshot': terminal,
                    'task_result': {'status': 'success', 'marked_by': 'keyboard:T',
                                    'simulation_time_s': 2.35, 'drone_position_m': [0., 0., 0.]}}
            original = run/'trajectory.jsonl'
            original.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            (run/'run_metadata.json').write_text(json.dumps(meta), encoding='utf-8')
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
            derived = prepare_replay_input(original, root/'derived', 1.)
            selected = read_trajectory(derived)
            self.assertEqual([r['simulation_time'] for r in selected], [0., 1., 2., 2.35])
            self.assertEqual(selected[-1]['source_event_ref'], 'run_metadata.json#terminal_snapshot')
            self.assertIsNone(selected[-1]['source_frame_id'])
            self.assertEqual(sum(r.get('action_type') == 'STOP' for r in selected), 1)
            self.assertEqual(trajectory_provenance(derived)['capture_contract'], contract)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()})

    def test_v2_accepts_actual_one_second_span_without_relaxing_v1(self):
        from scripts.bc_action_protocol import parse_action, encode_action, plan_action
        text = 'POSE 3 0 0 0 1.016666667'
        action = parse_action(text, schema='factory_pose_action_v2')
        decoded = parse_action(encode_action(action, schema='factory_pose_action_v2'), schema='factory_pose_action_v2')
        self.assertEqual(decoded['duration_s'], 1.016666667)
        plan = plan_action([0., 0., 3.], 0., action, schema='factory_pose_action_v2')
        self.assertEqual(plan['action_schema'], 'factory_pose_action_v2')
        self.assertEqual(plan['poses'][-1]['position'], [3., 0., 3.])
        with self.assertRaises(ValueError):
            parse_action(text)
        for bad in ('POSE 0 0 0 0 2', 'POSE 4 0 0 0 1', 'POSE 0 0 0 0 nan'):
            with self.subTest(text=bad), self.assertRaises(ValueError):
                parse_action(bad, schema='factory_pose_action_v2')
        with self.assertRaises(ValueError):
            parse_action('STOP', schema='unknown')

    def test_pair_dimensions_scale_focal_lengths_and_principal_points(self):
        from scripts.dual_camera_support import scale_camera_optics_for_resolution
        from tests.test_dual_camera_support import _CameraOptics, _Intrinsics
        thermal = _CameraOptics(1., 1., 1., _Intrinsics(100., 100., 64., 51., 128, 102), 1)
        rgb = scale_camera_optics_for_resolution(thermal, 256, 205)
        self.assertEqual((rgb.intrinsics.width, rgb.intrinsics.height), (256, 205))
        self.assertAlmostEqual(rgb.intrinsics.fx_px, 200.)
        self.assertAlmostEqual(rgb.intrinsics.fy_px, 100.*205/102)
        self.assertEqual((rgb.intrinsics.cx_px, rgb.intrinsics.cy_px), (128., 102.5))


if __name__ == '__main__':
    unittest.main()
