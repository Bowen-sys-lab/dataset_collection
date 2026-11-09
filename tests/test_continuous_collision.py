import unittest
import numpy as np


class SurfaceSweepTests(unittest.TestCase):
    def sweep(self, start, end, radius=.1):
        from scripts.continuous_collision import sweep_surface_segment
        plate = np.array([[[0, -2, -2], [0, 2, -2], [0, 2, 2]],
                          [[0, -2, -2], [0, 2, 2], [0, -2, 2]]], dtype=float)
        return sweep_surface_segment(start, end, plate, radius=radius)

    def test_normal_pass_is_clear(self):
        self.assertEqual(self.sweep([-1, 3, 0], [1, 3, 0])['status'], 'CLEAR_SURFACES')

    def test_wall_gap_is_clear(self):
        self.assertEqual(self.sweep([.11, -1, 0], [.11, 1, 0])['status'], 'CLEAR_SURFACES')

    def test_exact_grazing_is_contact(self):
        self.assertEqual(self.sweep([.1, -1, 0], [.1, 1, 0])['status'], 'CONTACT')

    def test_thin_plate_cannot_be_tunnelled_through(self):
        result = self.sweep([-10, 0, 0], [10, 0, 0])
        self.assertEqual(result['status'], 'CONTACT')
        lower, upper = result['contact_fraction_interval']
        self.assertLessEqual(lower, .495)
        self.assertGreaterEqual(upper + 1e-10, .495)
        self.assertLess(upper - lower, 1e-4)

    def test_stationary_contact_is_found(self):
        self.assertEqual(self.sweep([0, 0, 0], [0, 0, 0])['status'], 'CONTACT')

    def test_invalid_inputs_fail(self):
        for start, end, radius in [([float('nan'), 0, 0], [1, 0, 0], .1),
                                   ([0, 0], [1, 0, 0], .1), ([0, 0, 0], [1, 0, 0], -.1)]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                self.sweep(start, end, radius)

    def test_finite_overflow_cannot_prove_clearance(self):
        from scripts.continuous_collision import sweep_surface_segment
        enormous = np.array([[[0, -1e200, -1e200], [0, 1e200, -1e200], [0, 0, 1e200]]])
        with self.assertRaises(ValueError):
            sweep_surface_segment([0, 0, 0], [0, 0, 0], enormous, radius=.1)
        with self.assertRaises(ValueError):
            self.sweep([-1e200, 0, 0], [1e200, 0, 0])


if __name__ == '__main__':
    unittest.main()
