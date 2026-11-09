import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.trajectory_recorder import TrajectoryRecorder, build_training_sequence, read_trajectory


def state(frame, stamp, x):
    matrix=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[x,0.,2.,1.]]
    return dict(frame_id=frame,simulation_time=stamp,position=[x,0.,2.],
                orientation=[1.,0.,0.,0.],yaw=0.,recorded_body_world_matrix=matrix,
                rgb_camera_world_matrix=matrix,thermal_camera_world_matrix=matrix)


class RecordingUndoTests(unittest.TestCase):
    def recorder(self, directory):
        return TrajectoryRecorder.create(directory,dict(trajectory_source='manual_driving',
            sample_period_s=1.,task={'task_id':'city-trial'}),run_id='trial')

    def test_undo_keeps_raw_records_but_removes_discarded_motion_and_time_gap(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            r=self.recorder(directory)
            for i in range(7):r.append(state(i,float(i),float(i)))
            original=copy.deepcopy(r.records)
            r.record_collision(6.2,{'object_prim_path':'/Wall'})
            r.undo_to(3,7.)
            r.append(state(7,8.,4.))
            r.append(state(8,9.,5.))
            r.mark_success(9.2,[5.,0.,2.],terminal_snapshot=state(-1,9.2,5.))
            rows=build_training_sequence(r.records,r.metadata)
            self.assertEqual(r.records[:7],original)
            self.assertEqual([x['simulation_time'] for x in rows],[0.,1.,2.,3.,4.,5.,5.199999999999999])
            self.assertEqual([x['position'][0] for x in rows],[0.,1.,2.,3.,4.,5.,5.])
            self.assertEqual(rows[-1]['source_simulation_time_s'],9.2)
            self.assertEqual(sum(x['action_type']=='STOP' for x in rows),1)
            r.close()
            self.assertEqual(len(read_trajectory(r.run_dir/'trajectory.jsonl')),9)
            self.assertEqual(len(read_trajectory(r.run_dir/'training_trajectory.jsonl')),7)

    def test_second_undo_can_return_before_first_anchor(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            r=self.recorder(directory)
            for i in range(7):r.append(state(i,float(i),float(i)))
            r.record_collision(6.2,{'object_prim_path':'/Wall'});r.undo_to(3,7.)
            r.append(state(7,8.,4.));r.append(state(8,9.,5.))
            r.record_collision(9.2,{'object_prim_path':'/Wall'});r.undo_to(1,10.)
            r.append(state(9,11.,2.))
            r.mark_success(11.2,[2.,0.,2.],terminal_snapshot=state(-1,11.2,2.))
            rows=build_training_sequence(r.records,r.metadata)
            self.assertEqual([x['position'][0] for x in rows],[0.,1.,2.,2.])
            self.assertAlmostEqual(rows[-1]['simulation_time'],2.2)
            self.assertEqual(rows[2]['source_frame_id'],9)

    def test_exact_t_immediately_after_undo_keeps_source_time_and_provenance(self):
        from scripts.trajectory_recorder import trajectory_provenance
        from scripts.episode_replay_source import prepare_source
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            r=self.recorder(directory)
            for i in range(7):r.append(state(i,float(i),float(i)))
            r.record_collision(6.2,{'object_prim_path':'/Wall'});r.undo_to(3,8.)
            r.mark_success(8.2,[3.,0.,2.],terminal_snapshot=state(-1,8.2,3.))
            original=r.close()
            derived=prepare_source(original,Path(directory)/'replay',1.)
            provenance=trajectory_provenance(derived)
            self.assertEqual(provenance.get('collection_edit_schema'),'collision_undo_v1')
            self.assertEqual(len(provenance['collection_edit_events']),2)
            rows=read_trajectory(derived)
            self.assertEqual([row['source_frame_id'] for row in rows],[0,1,2,3,None])
            self.assertAlmostEqual(rows[-1]['simulation_time'],3.2)
            self.assertEqual(rows[-1]['source_simulation_time_s'],8.2)
            report=json.loads((r.run_dir/'mileage_quality.json').read_text())
            self.assertEqual(report['path_semantics'],'raw attempts; includes discarded motion and recovery discontinuities')
            self.assertIn('corrected_branch',report)
            derived_meta_path=derived.parent/'run_metadata.json'
            changed=json.loads(derived_meta_path.read_text(encoding='utf-8'))
            changed['collection_edit_events'][1]['anchor_frame_id']=0
            derived_meta_path.write_text(json.dumps(changed),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'edit history'):
                trajectory_provenance(derived)

    def test_t_in_same_simulation_tick_as_undo_has_one_terminal_pose(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            r=self.recorder(directory)
            for i in range(7):r.append(state(i,float(i),float(i)))
            r.record_collision(6.2,{'object_prim_path':'/Wall'});r.undo_to(3,8.)
            r.mark_success(8.,[3.,0.,2.],terminal_snapshot=state(-1,8.,3.))
            rows=build_training_sequence(r.records,r.metadata)
            self.assertEqual([x['simulation_time'] for x in rows],[0.,1.,2.,3.])
            self.assertEqual(rows[-1]['action_type'],'STOP')

    def test_pending_collision_cannot_be_marked_success_and_stale_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            r=self.recorder(directory)
            for i in range(7):r.append(state(i,float(i),float(i)))
            r.record_collision(6.2,{'object_prim_path':'/Wall'})
            with self.assertRaises(ValueError):r.mark_success(6.3,[6.,0.,2.],terminal_snapshot=state(-1,6.3,6.))
            with self.assertRaises(ValueError):r.undo_to(99,7.)
            r.undo_to(2,7.)
            r.append(state(7,8.,3.))
            r.record_collision(8.2,{'object_prim_path':'/Wall'})
            with self.assertRaises(ValueError):r.undo_to(5,9.)


if __name__=='__main__':unittest.main()
