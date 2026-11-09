"""Factory-only offline replay contracts; these tests never launch Isaac/Kit."""
import copy
import importlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

try:
    replay = importlib.import_module('scripts.replay_factory_inspection')
except ModuleNotFoundError:
    replay = None


class FactoryReplayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(replay, 'factory replay implementation is required')

    def test_route_sampling_keeps_every_contiguous_observation_record(self):
        records = [{'frame_id': i, 'simulation_time': i / 10,
                    'position': [0., 0., 3.], 'linear_velocity': [0., 0., 0.]}
                   for i in range(101)]
        distances = [20.] * 20 + [7.] * 61 + [20.] * 20
        selected, observation_ids = replay.select_records(records, distances, {}, 1.)
        self.assertEqual(observation_ids, set(range(20, 81)))
        self.assertEqual([r['frame_id'] for r in selected],
                         [0, 10] + list(range(20, 81)) + [90, 100])

    def test_source_gap_and_position_speed_cannot_certify_hold(self):
        rows = [{'frame_id': i, 'simulation_time': i * .3, 'position': [0., 0., 3.],
                 'linear_velocity': [0., 0., 0.]} for i in range(25)]
        with self.assertRaisesRegex(ValueError, 'hold'):
            replay.select_records(rows, [7.] * 25, {}, 1.)
        for i, row in enumerate(rows):
            row.update(simulation_time=i * .25, position=[float(i), 0., 3.])
        with self.assertRaisesRegex(ValueError, 'hold'):
            replay.select_records(rows, [7.] * 25, {}, 1.)

    def test_manual_preview_allows_short_fast_flight_and_preserves_T_frame(self):
        self.assertIn('manual_only',inspect.signature(replay.select_records).parameters)
        records=[{'frame_id':i,'simulation_time':i/10,'position':[i,0.,3.],
                  'linear_velocity':[10.,0.,0.]} for i in range(10)]
        selected,observations=replay.select_records(records,[40.]*10,{},.2,
            manual_only=True,confirmation_frame_id=7)
        self.assertEqual([r['frame_id'] for r in selected],[0,2,4,6,7,8,9])
        self.assertEqual(observations,set())
        with self.assertRaisesRegex(ValueError,'hold'):
            replay.select_records(records,[40.]*10,{},.2)

    def test_manual_preview_preserves_human_label_without_formal_acceptance(self):
        self.assertIn('manual_only',inspect.signature(replay.development_labels).parameters)
        provenance={'trajectory_source':'manual_driving',
                    'task_result':{'status':'success','marked_by':'keyboard:T','simulation_time_s':1.2}}
        labels=replay.development_labels(provenance,manual_only=True)
        self.assertEqual(labels['task_result'],provenance['task_result'])
        self.assertEqual(labels['confirmation_mode'],'manual_only')
        self.assertFalse(labels['automatic_observation_gate_applied'])
        self.assertFalse(labels['strict_sync_verified'])

    def test_manual_summary_does_not_invent_confirmation(self):
        report_function=getattr(replay,'manual_confirmation_report',None)
        self.assertIsNotNone(report_function,'manual confirmation report is missing')
        provenance={'trajectory_source':'manual_driving','task_result':{'status':'unjudged'}}
        self.assertEqual(report_function(provenance)['result'],'UNCONFIRMED')
        provenance['task_result']={'status':'success','marked_by':'keyboard:T'}
        report=report_function(provenance)
        self.assertEqual(report['result'],'MANUAL_CONFIRMED')
        self.assertFalse(report['automatic_rules_evaluated'])

    def test_manual_bundle_cannot_omit_available_recorded_T_frame(self):
        provenance={'trajectory_source':'manual_driving','task_result':{
            'status':'success','marked_by':'keyboard:T','nearest_trajectory_frame_id':7}}
        metadata={**replay.development_labels(provenance,manual_only=True),
                  'trajectory_provenance':provenance}
        bundle=SimpleNamespace(metadata=metadata,frames=[],trajectory_path=Path('unused'))
        with patch('scripts.generate_replay_video.load_replay',return_value=bundle), \
             patch('scripts.trajectory_recorder.read_trajectory',return_value=[{'frame_id':7}]):
            with self.assertRaisesRegex(ValueError,'manual confirmation frame'):
                replay.validate_bundle(Path('unused'))

    def test_provenance_never_promotes_synthetic_confirmation_or_sync(self):
        provenance = {'trajectory_source': 'synthetic_reference_route',
                      'task_result': {'status': 'unjudged'}}
        labels = replay.development_labels(provenance)
        self.assertEqual(labels['trajectory_source'], 'synthetic_reference_route')
        self.assertEqual(labels['task_result'], {'status': 'unjudged'})
        self.assertFalse(labels['manual_confirmation_created'])
        self.assertFalse(labels['strict_sync_verified'])
        self.assertEqual(labels['sync_validation_status'], 'not_verified')
        provenance['task_result'] = {'status': 'success', 'marked_by': 'keyboard:T'}
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            replay.development_labels(provenance)

    def test_capture_checks_reject_nonfinite_and_pose_timestamp_drift(self):
        record = {'simulation_time': 4., 'rgb_camera_world_matrix': np.eye(4).tolist()}
        meta = {'explicit_render_triggered': True, 'explicit_render_completed': True,
                'frozen_state_capture': True, 'pair_pose_max_error': 0.,
                'simulation_capture_drift_s': 0., 'simulation_time_before': 4.,
                'simulation_time_after': 4., 'camera_world_matrix': np.eye(4).tolist(),
                'thermal_camera_world_matrix': np.eye(4).tolist(),
                'thermal_coverage': {'complete_for_this_view': True,
                                     'apparent_t_nonfinite_pixels': 0}}
        replay.validate_capture(meta, record)
        for field, bad in [('pair_pose_max_error', float('nan')),
                           ('simulation_time_after', 4.01)]:
            changed = copy.deepcopy(meta); changed[field] = bad
            with self.assertRaises(ValueError):
                replay.validate_capture(changed, record)
        changed = copy.deepcopy(meta); changed['camera_world_matrix'][3][0] = .01
        with self.assertRaises(ValueError):
            replay.validate_capture(changed, record)

    def test_saved_arrays_and_images_are_checked_and_nonfinite_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); stem = root / 'frame_000000'
            rgb = np.zeros((205, 256, 4), dtype=np.uint8)
            thermal = np.zeros((102, 128), dtype=np.uint8)
            np.save(str(stem) + '_rgb.npy', rgb)
            np.save(str(stem) + '_display8.npy', thermal)
            np.save(str(stem) + '_radiance.npy', np.ones((102, 128)))
            np.save(str(stem) + '_apparent_t.npy', np.ones((102, 128)) * 300.)
            np.save(str(stem) + '_dn16.npy', np.zeros((102, 128), dtype=np.uint16))
            Image.fromarray(rgb).save(str(stem) + '_rgb.png')
            Image.fromarray(thermal).save(str(stem) + '_thermal.png')
            result = replay.validate_saved_frame(root, 0)
            self.assertEqual(result['nonfinite_values'], 0)
            broken = np.ones((102, 128)); broken[0, 0] = float('nan')
            np.save(str(stem) + '_radiance.npy', broken)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                replay.validate_saved_frame(root, 0)
            np.save(str(stem) + '_radiance.npy', np.ones((102, 128)))
            Path(str(stem) + '_thermal.png').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                replay.validate_saved_frame(root, 0)

    def test_hotspot_uses_exact_renderer_instance_path_and_visible_pixels(self):
        ids = np.array([[5, 5, 6], [0, 6, 6]])
        result = replay.hotspot_evidence(ids, {5: '/Hotspot', 6: '/Hotspot/Other'}, '/Hotspot', 2)
        self.assertEqual(result['hotspot_instance_ids'], [5])
        self.assertEqual(result['hotspot_visible_pixels'], 2)
        self.assertTrue(result['hotspot_visible'])
        self.assertFalse(replay.hotspot_evidence(ids, {6: '/Hotspot/Other'}, '/Hotspot', 1)['hotspot_visible'])

    def test_explicit_multi_zone_core_counts_union_without_including_neighbours(self):
        ids=np.array([[5,6,7],[6,7,8]])
        paths=['/Core/A','/Core/B']
        result=replay.hotspot_evidence(ids,{5:'/Core/A',6:'/Core/B',7:'/Core/B/Neighbour',8:'/Other'},'/Core/A',3,target_paths=paths)
        self.assertTrue(result['hotspot_visible'])
        self.assertEqual(result['hotspot_visible_pixels'],3)
        self.assertEqual(result['hotspot_instance_ids'],[5,6])
        self.assertEqual(result['hotspot_prim_paths'],paths)
        with self.assertRaises(ValueError):replay.hotspot_evidence(ids,{5:'/Core/A'},'/Core/A',1,target_paths=['/Unrelated'])


if __name__ == '__main__':
    unittest.main()
