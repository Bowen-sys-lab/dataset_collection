import math
import unittest
import numpy as np


def pose(time, xyz, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return {'simulation_time': time, 'position': xyz, 'yaw': yaw,
            'recorded_body_world_matrix': [[c,s,0,0],[-s,c,0,0],[0,0,1,0],xyz+[1]]}


class BCPoseTargetTests(unittest.TestCase):
    def test_yaw_local_future_target_preserves_left_up_and_relative_start_heading(self):
        from scripts.bc_pose_targets import future_pose_target
        first = pose(1., [0.,0.,3.], math.pi/2)
        later = pose(1.5, [-1.,2.,3.5], math.pi/2+.2)
        label = future_pose_target(first, later, initial_yaw=0.)
        self.assertAlmostEqual(label['forward_m'], 2.)
        self.assertAlmostEqual(label['left_m'], 1.)
        self.assertAlmostEqual(label['up_m'], .5)
        self.assertAlmostEqual(label['yaw_delta_rad'], .2)
        self.assertAlmostEqual(label['relative_start_yaw_rad'], math.pi/2)
        self.assertEqual(label['duration_s'], .5)

    def test_backward_is_not_clipped_or_quantized_as_forward_zero(self):
        from scripts.bc_pose_targets import future_pose_target
        label = future_pose_target(pose(0.,[0.,0.,3.],0.), pose(.5,[-2.,0.,3.],0.), initial_yaw=0.)
        self.assertEqual(label['forward_m'], -2.)
        self.assertEqual(label['motion_kind'], 'MOVE')

    def test_yaw_wrap_takes_short_turn_and_hover_is_not_stop(self):
        from scripts.bc_pose_targets import future_pose_target
        label = future_pose_target(pose(0.,[0.,0.,3.], math.radians(179)),
                                   pose(.5,[0.,0.,3.], math.radians(-179)), initial_yaw=0.)
        self.assertAlmostEqual(label['yaw_delta_rad'], math.radians(2))
        hold = future_pose_target(pose(0.,[0.,0.,3.],0.), pose(.5,[0.,0.,3.],0.), initial_yaw=0.)
        self.assertEqual(hold['motion_kind'], 'HOLD')
        self.assertFalse(hold['stop'])

    def test_bad_time_pose_scale_or_yaw_is_rejected(self):
        from scripts.bc_pose_targets import future_pose_target
        first = pose(1., [0.,0.,3.], 0.)
        for bad in [pose(1.,[0.,0.,3.],0.), pose(2.,[math.nan,0.,3.],0.)]:
            with self.assertRaises(ValueError):
                future_pose_target(first, bad, initial_yaw=0.)
        for field, value in [('yaw',1.), ('position',[2.,0.,3.])]:
            bad = pose(2.,[0.,0.,3.],0.)
            bad[field] = value
            with self.assertRaises(ValueError):
                future_pose_target(first, bad, initial_yaw=0.)
        bad = pose(2.,[0.,0.,3.],0.)
        bad['recorded_body_world_matrix'][0][0] = 2
        with self.assertRaises(ValueError):
            future_pose_target(first, bad, initial_yaw=0.)

    def test_future_endpoint_uses_recorded_time_and_clips_only_at_exact_terminal_event(self):
        from scripts.bc_pose_targets import select_future_endpoint
        states = [pose(t,[t,0.,3.],0.) for t in [0.,.1,.2,.3,.4,.5,.6]]
        states[-1].update(action_type='STOP',source_event_ref='run_metadata.json#terminal_snapshot')
        self.assertEqual(select_future_endpoint(states, 0, horizon_s=.5)['simulation_time'], .5)
        self.assertEqual(select_future_endpoint(states, 4, horizon_s=.5)['simulation_time'], .6)
        with self.assertRaises(ValueError):
            select_future_endpoint(states, 6, horizon_s=.5)


if __name__ == '__main__':
    unittest.main()
