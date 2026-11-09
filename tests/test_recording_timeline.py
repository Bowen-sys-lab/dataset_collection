import copy
import tempfile
import unittest
import numpy as np
from scripts import trajectory_recorder as module


def row(t, frame):
    matrix = np.eye(4).tolist()
    return {'frame_id':frame, 'simulation_time':t, 'position':[0.,0.,0.],
            'recorded_body_world_matrix':matrix, 'rgb_camera_world_matrix':matrix,
            'thermal_camera_world_matrix':matrix}


class Timeline:
    def __init__(self):
        self.end = 100.; self.current = 0.; self.loop = True; self.commits = 0
    def get_current_time(self): return self.current
    def get_end_time(self): return self.end
    def is_looping(self): return self.loop
    def set_end_time(self, value): self.end = value
    def set_looping(self, value): self.loop = value
    def commit(self): self.commits += 1


class RecordingTimelineTests(unittest.TestCase):
    def test_frozen_capture_extends_end_before_replicator_step(self):
        timeline = Timeline(); timeline.current = 100.0166666667
        self.assertTrue(hasattr(module, 'prepare_frozen_capture_timeline'))
        module.prepare_frozen_capture_timeline(timeline, timeline.current)
        self.assertFalse(timeline.loop)
        self.assertGreaterEqual(timeline.end, 101.0166666667)
        self.assertEqual(timeline.current, 100.0166666667)

    def test_already_safe_frozen_range_needs_no_timeline_commit(self):
        timeline = Timeline(); timeline.end = 86400.; timeline.loop = False
        self.assertTrue(hasattr(module, 'prepare_frozen_capture_timeline'))
        module.prepare_frozen_capture_timeline(timeline, 10.)
        self.assertEqual(timeline.commits, 0)

    def test_recording_extends_runtime_range_and_disables_loop_without_rewinding(self):
        timeline = Timeline(); timeline.current = 98.5
        self.assertTrue(hasattr(module, 'configure_recording_timeline'))
        report = module.configure_recording_timeline(timeline)
        self.assertFalse(timeline.loop)
        self.assertGreaterEqual(timeline.end, 86498.5)
        self.assertEqual(timeline.current, 98.5)
        self.assertGreater(timeline.commits, 0)
        self.assertEqual(report['original_end_time_s'], 100.)
        self.assertEqual(report['recording_end_time_s'], timeline.end)

    def test_live_clock_rewind_fails_but_paused_equal_time_is_valid(self):
        self.assertTrue(hasattr(module, 'validate_recording_time'))
        records = [row(99.9, 999)]
        module.validate_recording_time(records, 99.9)
        module.validate_recording_time(records, 100.1)
        with self.assertRaisesRegex(ValueError, 'time.*backwards|timeline.*rewind'):
            module.validate_recording_time(records, .83)

    def test_T_cannot_be_written_in_an_earlier_clock_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = module.TrajectoryRecorder.create(directory, {'trajectory_source':'manual_driving'})
            recorder.append(row(99.9, 999))
            before = copy.deepcopy(recorder.metadata)
            with self.assertRaisesRegex(ValueError, 'time.*backwards|timeline.*rewind'):
                recorder.mark_success(.83, [0,0,0], terminal_snapshot=row(.83,-1))
            self.assertEqual(recorder.metadata, before)

    def test_corrupt_historical_T_anchor_cannot_export_teleporting_STOP(self):
        records = [row(.1,0), row(.2,1), row(99.9,999)]
        metadata = {'trajectory_source':'manual_driving', 'terminal_snapshot':row(.83,-1),
                    'task_result':{'status':'success','marked_by':'keyboard:T',
                                   'simulation_time_s':.83,'drone_position_m':[0.,0.,0.],
                                   'nearest_trajectory_frame_id':999}}
        with self.assertRaisesRegex(ValueError, 'confirmation.*time|terminal.*time'):
            module.build_training_sequence(records, metadata)

    def test_valid_confirmation_can_precede_later_raw_flight(self):
        records = [row(.1,0),row(.2,1),row(.3,2)]
        metadata = {'trajectory_source':'manual_driving', 'terminal_snapshot':row(.15,-1),
                    'task_result':{'status':'success','marked_by':'keyboard:T',
                                   'simulation_time_s':.15,'drone_position_m':[0.,0.,0.],
                                   'nearest_trajectory_frame_id':0}}
        selected = module.build_training_sequence(records,metadata)
        self.assertEqual([r['simulation_time'] for r in selected],[.1,.15])
        self.assertEqual(selected[-1]['action_type'],'STOP')

    def test_real_confirmation_after100_seconds_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = module.TrajectoryRecorder.create(directory, {'trajectory_source':'manual_driving'})
            recorder.append(row(99.9,999));recorder.append(row(100.,1000))
            self.assertTrue(recorder.mark_success(100.05,[0,0,0],terminal_snapshot=row(100.05,-1)))
            selected = module.build_training_sequence(recorder.records,recorder.metadata)
            self.assertEqual([r['simulation_time'] for r in selected],[99.9,100.,100.05])
            self.assertEqual(selected[-1]['action_type'],'STOP')
