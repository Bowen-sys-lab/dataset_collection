import math
import unittest
import numpy as np


class ActionBridgeTests(unittest.TestCase):
    def plan(self, forward=1., down=0., yaw=0., **kwargs):
        from scripts.aerovla_action_bridge import plan_aerovla_action
        return plan_aerovla_action([0.,0.,10.], kwargs.pop('start_yaw', 0.),
            {'forward_m': forward, 'down_m': down, 'yaw_rad': yaw}, **kwargs)

    def test_ned_down_becomes_negative_world_z(self):
        plan = self.plan(forward=0., down=1.)
        np.testing.assert_allclose(plan['poses'][-1]['position'], [0,0,9])

    def test_small_turn_moves_along_new_heading(self):
        plan = self.plan(forward=2., yaw=.2)
        np.testing.assert_allclose(plan['poses'][-1]['position'], [2*math.cos(.2),-2*math.sin(.2),10])
        self.assertAlmostEqual(plan['poses'][-1]['yaw'], -.2)
        self.assertTrue(plan['translation_executed'])

    def test_large_turn_reference_policy_suppresses_horizontal_translation(self):
        plan = self.plan(forward=3.1632653, yaw=-.92040816)
        np.testing.assert_allclose(plan['poses'][-1]['position'], [0,0,10])
        self.assertAlmostEqual(plan['poses'][-1]['yaw'], .92040816)
        self.assertFalse(plan['translation_executed'])

    def test_translation_threshold_is_strict(self):
        self.assertTrue(self.plan(yaw=.249)['translation_executed'])
        self.assertFalse(self.plan(yaw=.25)['translation_executed'])

    def test_heading_and_time_sampling(self):
        plan = self.plan(start_yaw=math.pi/2)
        np.testing.assert_allclose(plan['poses'][-1]['position'], [0,1,10], atol=1e-12)
        times = [p['time_s'] for p in plan['poses']]
        self.assertEqual(times[0], 0.)
        self.assertTrue(all(0 < b-a <= 1/60+1e-10 for a,b in zip(times,times[1:])))

    def test_invalid_or_unbounded_commands_fail(self):
        for values in [{'forward': -1}, {'forward': 6}, {'yaw': float('nan')},
                       {'down': 6}, {'step_s': 1e-300}, {'yaw_speed_rad_s': 0}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.plan(**values)


if __name__ == '__main__':
    unittest.main()
