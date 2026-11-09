import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from scripts.trajectory_recorder import TrajectoryRecorder
from tests.test_recording_undo import state


class ManualCollisionUndoTests(unittest.TestCase):
    def test_sweep_ignores_drone_and_detects_crossed_thin_wall(self):
        from scripts.manual_collision_undo import SphereGuard
        class Query:
            def overlap_sphere(self, radius, position, callback, anyHit):
                callback(SimpleNamespace(collision='/Drone/body',rigid_body='/Drone'))
            def sweep_sphere_all(self, radius, position, direction, distance, callback, bothSides):
                callback(SimpleNamespace(collision='/Drone/body',rigid_body='/Drone'))
                callback(SimpleNamespace(collision='/Wall',rigid_body='',distance=.4))
        guard=SphereGuard(Query(),'/Drone',.2,vector=lambda xyz:tuple(xyz))
        self.assertEqual(guard((0,0,2),(2,0,2))['object_prim_path'],'/Wall')

    def test_undo_commits_only_after_successful_restore_and_delays_sampling(self):
        from scripts.manual_collision_undo import RecordingRecovery
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            r=TrajectoryRecorder.create(directory,{'trajectory_source':'manual_driving','sample_period_s':1.},run_id='trial')
            for i in range(7):
                row=state(i,float(i),float(i))
                row['controller_target_pose']={'position':[i,0,2],'roll':.1,'pitch':.2,'yaw':.3}
                r.append(row)
            clock=[6.2]
            recovery=RecordingRecovery(r,lambda:clock[0],lambda old,new:None)
            recovery.collision({'object_prim_path':'/Wall'})
            clock[0]=8.
            pose=recovery.prepare_undo()
            self.assertEqual(pose['position'],[3,0,2])
            self.assertTrue(r.metadata['collision_pending'])
            self.assertEqual(len(r.metadata['collection_edit_events']),1)
            recovery.commit_undo()
            self.assertFalse(r.metadata['collision_pending'])
            self.assertFalse(recovery.can_sample(8.5))
            self.assertTrue(recovery.can_sample(9.))


if __name__=='__main__':unittest.main()
