import math
import unittest
from scripts import drone_control_math as m


class AttitudeTests(unittest.TestCase):
    def test_new_keys_and_opposites(self):
        for keys, expected in [(('I', 'J'), (-1, 1)), (('K', 'L'), (1, -1)),
                               (('I', 'K', 'J', 'L'), (0, 0))]:
            self.assertEqual(m.attitude_command_from_keys(keys), expected)

    def test_combined_rotation_and_round_trip(self):
        q = m.orientation_wxyz_from_rpy(math.pi/2, 0, math.pi/2)
        for got in q:
            self.assertAlmostEqual(got, 0.5)
        q = m.orientation_wxyz_from_rpy(.2, -.3, 1.1)
        self.assertAlmostEqual(sum(v*v for v in q), 1)
        for got, want in zip(m.rpy_from_orientation_wxyz(q), (.2, -.3, 1.1)):
            self.assertAlmostEqual(got, want)

    def test_tilt_limits_speed_acceleration_and_return(self):
        c = m.AttitudeConfig()
        angle = rate = 0.
        for _ in range(300):
            previous = rate
            angle, rate = m.step_tilt(angle, rate, 1, .01, c)
            self.assertLessEqual(abs(angle), math.radians(20) + 1e-9)
            self.assertLessEqual(abs(rate), c.max_rate + 1e-9)
            self.assertLessEqual(abs(rate-previous), c.acceleration*.01 + 1e-8)
        self.assertAlmostEqual(angle, math.radians(20))
        for _ in range(300):
            angle, rate = m.step_tilt(angle, rate, 0, .01, c)
            self.assertGreaterEqual(angle, 0)
        self.assertAlmostEqual(angle, 0)
        self.assertAlmostEqual(rate, 0)

    def test_time_step_equivalence(self):
        results = []
        for hz in (30, 60, 120):
            angle = rate = 0.
            for _ in range(hz):
                angle, rate = m.step_tilt(angle, rate, -1, 1/hz)
            results.append(angle)
        self.assertLess(max(results)-min(results), .001)

    def test_invalid_config(self):
        for kw in ({'max_angle': 0}, {'max_rate': -1}, {'acceleration': float('nan')}):
            with self.assertRaises(ValueError):
                m.AttitudeConfig(**kw)
