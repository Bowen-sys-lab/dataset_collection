import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sample(t, frame=0):
    m = np.eye(4).tolist()
    return {'frame_id': frame, 'simulation_time': t, 'position': [0.,0.,0.],
            'orientation': [1.,0.,0.,0.], 'recorded_body_world_matrix': m,
            'rgb_camera_world_matrix': m, 'thermal_camera_world_matrix': m,
            'command': {'forward': 0., 'right': 0., 'vertical': 0., 'yaw': 0.}}


class SplitPreviewTests(unittest.TestCase):
    def test_duplicate_timestamp_preserves_last_measured_speed(self):
        from scripts.factory_pilot_hud import ObservationTimer
        timer=ObservationTimer()
        timer.update(0.,[0,0,0],7.,True)
        moving=timer.update(.1,[.1,0,0],7.,True)
        duplicate=timer.update(.1,[.1,0,0],7.,True)
        self.assertEqual(duplicate['speed_m_s'],moving['speed_m_s'])
        self.assertFalse(duplicate['slow_enough'])

    def test_split_publisher_shows_async_age_and_save_flag_is_rejected(self):
        from scripts.start_crazyflie_dual_camera_view import DualCameraPreview
        from scripts import preview_scheduler as module
        self.assertTrue(hasattr(module,'validate_split_options'))
        with self.assertRaises(ValueError): module.validate_split_options('split',True)
        class Provider:
            def set_data_array(self,*args,**kwargs):pass
        class Label: text='paired FPS'
        p=DualCameraPreview(None,None,None,None,None,width=160,height=128,cleanup=None,metadata={},show_ui=True)
        p.window=object();p.thermal_provider=Provider();p.status=Label()
        class Thermal:display8=np.zeros((128,160),dtype=np.uint8)
        p.publish_thermal(Thermal(),{'simulation_time_s':.5})
        self.assertIn('independent',p.status.text)
        self.assertIn('IR',p.status.text)
    def test_benchmark_warmup_is_removed_from_control_metrics(self):
        from scripts.preview_metrics import ControlMetrics
        self.assertTrue(hasattr(ControlMetrics, 'clear_measurements'))
        metrics = ControlMetrics.__new__(ControlMetrics)
        metrics.steps=100; metrics.step_wall_times=[1.,2.]; metrics.callback_times=[1.]
        metrics.input_latencies=[{'seconds':1}]; metrics.pending=('stale',)
        metrics.clear_measurements()
        self.assertEqual(metrics.steps,0)
        self.assertEqual(metrics.step_wall_times,[])
        self.assertIsNone(metrics.pending)
    def test_independent_rates_do_not_invoke_thermal_on_rgb_ticks(self):
        from scripts import preview_scheduler as module
        self.assertTrue(hasattr(module, 'SplitPreviewScheduler'), 'independent RGB/IR scheduler missing')
        scheduler = module.SplitPreviewScheduler(15., 2.)
        rgb = thermal = 0
        for i in range(120):
            due = scheduler.poll(i / 60.)
            rgb += due['rgb']; thermal += due['thermal']
        self.assertEqual(rgb, 30)
        self.assertEqual(thermal, 4)
        # A slow frame skips missed work; it must not create a catch-up burst.
        self.assertEqual(scheduler.poll(10.), {'rgb': True, 'thermal': True})
        self.assertEqual(scheduler.poll(10.), {'rgb': False, 'thermal': False})

    def test_pause_duplicate_timestamp_does_not_erase_hold(self):
        from scripts.factory_pilot_hud import ObservationTimer
        timer = ObservationTimer()
        for i in range(32):
            result = timer.update(i/10, [0,0,0], 7., True)
        for _ in range(4):
            result = timer.update(3.1, [0,0,0], 7., True)
        self.assertAlmostEqual(result['hold_s'], 3.)

    def test_rgb_refresh_never_calls_thermal(self):
        from scripts.start_crazyflie_dual_camera_view import DualCameraPreview
        class RGB:
            def get_data(self): return np.zeros((128,160,4), dtype=np.uint8)
        class IR:
            def get_outputs(self, **kwargs): raise AssertionError('RGB tick called thermal')
        p = DualCameraPreview(None,None,IR(),RGB(),None,width=160,height=128,
                              cleanup=None,metadata={},show_ui=False)
        self.assertTrue(hasattr(p, 'publish_rgb'), 'RGB-only publisher missing')
        self.assertEqual(p.publish_rgb().shape, (128,160,4))


class TerminalTests(unittest.TestCase):
    def test_terminal_replay_has_its_own_readable_source_and_original_lineage(self):
        import json
        from scripts import replay_factory_inspection as module
        self.assertTrue(hasattr(module,'write_terminal_replay_source'))
        from scripts.trajectory_recorder import trajectory_provenance,read_trajectory
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); original=root/'raw';original.mkdir()
            trajectory=original/'trajectory.jsonl'
            trajectory.write_text(''.join(json.dumps(r)+'\n' for r in [sample(0.,0),sample(.1,1),sample(.2,2)]))
            metadata={'trajectory_source':'manual_driving','terminal_snapshot':sample(.15,2),
                      'task_result':{'status':'success','marked_by':'keyboard:T','simulation_time_s':.15,'drone_position_m':[0.,0.,0.]}}
            (original/'run_metadata.json').write_text(json.dumps(metadata))
            provenance=trajectory_provenance(trajectory)
            records=module.manual_replay_records(read_trajectory(trajectory),provenance)
            derived=module.write_terminal_replay_source(root/'replay',trajectory,records)
            self.assertEqual(read_trajectory(derived)[-1]['simulation_time'],.15)
            self.assertNotEqual(trajectory_provenance(derived)['trajectory_sha256'],provenance['trajectory_sha256'])
            self.assertEqual(len(read_trajectory(trajectory)),3)
            self.assertEqual(json.loads((derived.parent/'run_metadata.json').read_text())['source_trajectory_sha256'],provenance['trajectory_sha256'])
    def test_manual_replay_uses_exact_terminal_event_and_excludes_later_flight(self):
        from scripts import replay_factory_inspection as module
        self.assertTrue(hasattr(module,'manual_replay_records'))
        terminal=sample(.15,9)
        provenance={'trajectory_source':'manual_driving','terminal_snapshot':terminal,
                    'task_result':{'status':'success','marked_by':'keyboard:T',
                                   'simulation_time_s':.15,'drone_position_m':[0.,0.,0.]}}
        rows=module.manual_replay_records([sample(0.,0),sample(.1,1),sample(.2,2)],provenance)
        self.assertEqual([r['simulation_time'] for r in rows],[0.,.1,.15])
        self.assertEqual(rows[-1]['action_type'],'STOP')
    def test_exact_T_snapshot_is_preserved_and_training_ends_there(self):
        from scripts.trajectory_recorder import TrajectoryRecorder
        from scripts import trajectory_recorder as module
        self.assertTrue(hasattr(module, 'build_training_sequence'), 'terminal training export missing')
        with tempfile.TemporaryDirectory() as folder:
            r = TrajectoryRecorder.create(folder, {'trajectory_source': 'manual_driving'})
            r.append(sample(0.,0)); r.append(sample(.1,1))
            terminal = sample(.15,2)
            self.assertTrue(r.mark_success(.15, [0,0,0], terminal_snapshot=terminal))
            r.append(sample(.2,2))
            raw = copy.deepcopy(r.records)
            sequence = module.build_training_sequence(r.records, r.metadata)
            self.assertEqual(r.records, raw)
            self.assertEqual([x['simulation_time'] for x in sequence], [0.,.1,.15])
            self.assertEqual(sequence[-1]['action_type'], 'STOP')
            self.assertIsNone(sequence[-1]['source_frame_id'])
            self.assertEqual(sequence[-1]['source_event_ref'],'run_metadata.json#terminal_snapshot')
            self.assertEqual(sequence[-1]['rgb_camera_world_matrix'], terminal['rgb_camera_world_matrix'])
            r.flush()
            sidecar=json.loads((r.run_dir/'training_sequence_metadata.json').read_text())
            self.assertEqual(sidecar['source_run_metadata_sha256'],hashlib.sha256((r.run_dir/'run_metadata.json').read_bytes()).hexdigest())
            self.assertFalse(r.mark_success(.2,[0,0,0],terminal_snapshot=sample(.2,3)))

    def test_no_export_without_manual_confirmation_or_complete_snapshot(self):
        from scripts import trajectory_recorder as module
        self.assertTrue(hasattr(module, 'build_training_sequence'))
        for metadata in ({}, {'trajectory_source':'synthetic_control_probe',
                             'task_result': {'status':'success','marked_by':'keyboard:T'},
                             'terminal_snapshot':sample(.1,1)},
                         {'trajectory_source':'manual_driving',
                          'task_result': {'status':'success','marked_by':'keyboard:T'}}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                module.build_training_sequence([sample(0.)], metadata)

    def test_synthetic_source_cannot_mark_human_success(self):
        from scripts.trajectory_recorder import TrajectoryRecorder
        with tempfile.TemporaryDirectory() as folder:
            r=TrajectoryRecorder.create(folder,{'trajectory_source':'synthetic_control_probe'})
            with self.assertRaises(ValueError): r.mark_success(0.,[0,0,0])


class CollectionSelectionTests(unittest.TestCase):
    def test_readiness_binds_sensor_thermal_environment_display_and_material_dependencies(self):
        from scripts import factory_collection_config as module
        self.assertTrue(hasattr(module,'runtime_dependency_hashes'))
        hashes=module.runtime_dependency_hashes()
        for path in ('configs/model02_v7m_thermal.yaml','configs/model02_v7m_night_environment.yaml',
                     'configs/model02_v7m_display_fixed.yaml',
                     'third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml'):
            self.assertIn(path,hashes)
        self.assertTrue(any(p.startswith('factory/model02_thermal_v7m/configuration/materials/') for p in hashes))
    def test_trajectory_quality_catches_gaps_and_corrupt_camera_transforms(self):
        from scripts import factory_bc_pilot as module
        self.assertTrue(hasattr(module,'recording_quality'))
        good=[sample(0.,0),sample(.1,1),sample(.2,2)]
        self.assertEqual(module.recording_quality(good)['result'],'PASS')
        bad=copy.deepcopy(good);bad[2]['simulation_time']=.5
        self.assertEqual(module.recording_quality(bad)['result'],'FAIL')
        bad=copy.deepcopy(good);bad[1]['rgb_camera_world_matrix']=[[float('nan')]]
        self.assertEqual(module.recording_quality(bad)['result'],'FAIL')
    def test_all_selections_record_exact_description_target_and_shared_spawn(self):
        from scripts import factory_bc_pilot as module
        self.assertTrue(hasattr(module, 'resolve_selection'), 'target/start selection not connected')
        for group in ('A','B','D','E','F'):
            tank=module.resolve_selection('tank',1,group,require_ready=False)
            pipe=module.resolve_selection('pipe',2,group,require_ready=False)
            self.assertEqual(tank['task']['start_position'],pipe['task']['start_position'])
            self.assertEqual(tank['yaw_degrees'],pipe['yaw_degrees'])
            self.assertNotEqual(tank['task']['target_prim_path'],pipe['task']['target_prim_path'])
            self.assertNotEqual(tank['task']['instruction'],pipe['task']['instruction'])
            self.assertEqual(tank['task']['environment_evidence']['bc_pilot']['start_group'],group)
        with self.assertRaises(ValueError): module.resolve_selection('pipe',4,'A',require_ready=False)
        with self.assertRaises(ValueError): module.resolve_selection('tank',1,'C',require_ready=False)

    def test_new_launch_persists_runtime_config_and_independent_rates(self):
        from scripts import factory_bc_pilot as module
        self.assertTrue(hasattr(module,'resolve_selection'))
        # Runtime readiness is intentionally required; tested after geometry audit generates evidence.
        launch=module.build_launch('factory-bc-pilot-two-target-contract',target='tank',instruction_variant=2,start_group='A')
        self.assertEqual(launch['instruction_variant'],2)
        self.assertEqual(launch['preview_args'][launch['preview_args'].index('--live-refresh')+1],'on-demand')
        self.assertIn('--start-position',launch['preview_args'])
        self.assertEqual(launch['task_snapshot']['instruction'],module.resolve_selection('tank',2,'A')['task']['instruction'])


if __name__ == '__main__': unittest.main()
