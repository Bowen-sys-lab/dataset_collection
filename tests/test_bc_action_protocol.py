import math
import unittest


class BCActionProtocolTests(unittest.TestCase):
    def test_signed_pose_and_duration_round_trip_without_clipping(self):
        from scripts.bc_action_protocol import encode_action, parse_action
        action={'forward_m':-.5,'left_m':.25,'up_m':-.1,'yaw_delta_rad':.2,'duration_s':.5,'stop':False}
        decoded=parse_action(encode_action(action))
        for key,value in action.items():
            self.assertEqual(decoded[key],value)

    def test_hover_and_stop_are_distinct_even_when_all_motion_values_are_zero(self):
        from scripts.bc_action_protocol import parse_action
        hover=parse_action('POSE 0 0 0 0 0.5')
        self.assertFalse(hover['stop'])
        self.assertEqual(hover['motion_kind'],'HOLD')
        stop=parse_action('STOP</s>')
        self.assertTrue(stop['stop'])
        self.assertEqual(stop['duration_s'],0)

    def test_malformed_nonfinite_original_bins_and_unsafe_rates_are_errors(self):
        from scripts.bc_action_protocol import parse_action
        for text in ['62 49 16','POSE 0 0 0 0','POSE nan 0 0 0 .5','POSE 1e9 0 0 0 .5',
                     'POSE 0 0 0 0 0','POSE 0 0 0 0 1','POSE 2 0 0 0 .5','POSE 0 0 1 0 .5',
                     'POSE 0 0 0 1 .5','STOP now','STOP</s></s>','POSE 0 0 0 0 .5 extra']:
            with self.subTest(text=text),self.assertRaises(ValueError):
                parse_action(text)

    def test_encode_cannot_hide_movement_in_stop_or_boolean_numbers(self):
        from scripts.bc_action_protocol import encode_action
        base={'forward_m':0.,'left_m':0.,'up_m':0.,'yaw_delta_rad':0.,'duration_s':.5,'stop':False}
        with self.assertRaises(ValueError):
            encode_action(dict(base,forward_m=True))
        with self.assertRaises(ValueError):
            encode_action(dict(base,forward_m=.1,duration_s=0.,stop=True))

    def test_plan_uses_initial_yaw_for_forward_left_and_simultaneous_turn(self):
        from scripts.bc_action_protocol import parse_action, plan_action
        plan=plan_action([10.,20.,3.],math.pi/2,parse_action('POSE 1 .25 .1 .3 .5'))
        end=plan['poses'][-1]
        self.assertAlmostEqual(end['position'][0],9.75)
        self.assertAlmostEqual(end['position'][1],21.)
        self.assertAlmostEqual(end['position'][2],3.1)
        self.assertAlmostEqual(end['yaw'],math.pi/2+.3)
        self.assertEqual(end['time_s'],.5)
        self.assertTrue(plan['translation_executed'])

    def test_plan_keeps_hover_time_and_stop_has_no_motion_or_time_advance(self):
        from scripts.bc_action_protocol import parse_action, plan_action
        hover=plan_action([0.,0.,3.],0.,parse_action('POSE 0 0 0 0 .5'))
        self.assertGreater(len(hover['poses']),1)
        self.assertEqual(hover['poses'][-1]['time_s'],.5)
        stop=plan_action([0.,0.,3.],0.,parse_action('STOP'))
        self.assertEqual(len(stop['poses']),1)
        self.assertTrue(stop['stop_requested'])

    def test_unrepresentable_motion_or_excessive_sampling_is_rejected(self):
        from scripts.bc_action_protocol import parse_action,plan_action
        with self.assertRaises(ValueError):
            plan_action([1e300,0.,3.],0.,parse_action('POSE .5 0 0 0 .5'))
        with self.assertRaises(ValueError):
            plan_action([0.,0.,3.],0.,parse_action('POSE .5 0 0 0 .5'),step_s=1e-9)


if __name__=='__main__':
    unittest.main()
