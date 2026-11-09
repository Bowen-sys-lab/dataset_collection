import unittest
from types import SimpleNamespace
from scripts.drone_keyboard_controller import DroneKeyboardController


class CollisionUndoControlTests(unittest.TestCase):
    def controller(self):
        c=DroneKeyboardController()
        c._running=c._ready=True
        c._timeline=SimpleNamespace(is_playing=lambda:True)
        c._has_control_focus=lambda:True
        c._runtime=SimpleNamespace(valid=True,ready=True,write_pose=lambda *a:None)
        return c

    def test_obstacle_latches_hold_clears_motion_and_blocks_confirmation(self):
        c=self.controller();c.keys.press('W');events=[]
        c.collision_guard=lambda old,new:{'object_prim_path':'/Wall'}
        c.collision_callback=lambda detail:events.append(detail)
        origin=c.position
        c._on_physics_pre(.1,None)
        self.assertTrue(c.collision_held)
        self.assertEqual(c.position,origin)
        self.assertEqual(c.velocity,(0.,0.,0.,0.))
        self.assertFalse(c.keys.pressed)
        c._on_physics_pre(.1,None)
        self.assertEqual(len(events),1)
        calls=[];c.task_success_callback=lambda:calls.append(1) or True
        self.assertFalse(c._mark_task_success());self.assertFalse(calls)
        c._home=((99.,99.,99.),0.,0.,0.);c.reset_motion()
        self.assertEqual(c.position,origin)

    def test_undo_restores_controller_pose_and_waits_for_fresh_input(self):
        c=self.controller();c.collision_held=True;c.keys.press('W')
        c.undo_callback=lambda:{'position':[4.,5.,6.],'roll':.1,'pitch':.2,'yaw':.3}
        c.undo_collision()
        self.assertFalse(c.collision_held)
        self.assertEqual(c.position,(4.,5.,6.))
        self.assertEqual((c.roll,c.pitch,c.yaw),(.1,.2,.3))
        self.assertFalse(c.keys.pressed)

    def test_failed_pose_restore_does_not_commit_undo(self):
        c=self.controller();c.collision_held=True;committed=[]
        c.undo_callback=lambda:{'position':[4.,5.,6.]}
        c.undo_committed_callback=lambda:committed.append(True)
        def fail(*args):raise RuntimeError('tensor restore failed')
        c._runtime.write_pose=fail
        c._fail=lambda message:None
        self.assertFalse(c.undo_collision())
        self.assertTrue(c.collision_held)
        self.assertEqual(committed,[])


if __name__=='__main__':unittest.main()
