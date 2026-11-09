import unittest

from scripts.render_model02_weaklight import assess_views


class WeaklightRenderTests(unittest.TestCase):
    def views(self):
        return [{'view':name,'nonblack_fraction':.3,'saturated_fraction':.001,
                 'thermal_apparent_temperature_finite':True}
                for name in ('overview','building','tank')]

    def test_almost_black_tank_is_not_accepted(self):
        views = self.views()
        views[2]['nonblack_fraction'] = .0000027
        self.assertTrue(any('tank' in issue for issue in assess_views(views)))

    def test_visible_weaklight_views_are_accepted(self):
        self.assertEqual(assess_views(self.views()), [])

    def test_missing_view_or_nonfinite_thermal_cannot_pass(self):
        self.assertTrue(assess_views(self.views()[:-1]))
        views = self.views()
        views[0]['thermal_apparent_temperature_finite'] = False
        self.assertTrue(assess_views(views))


if __name__ == '__main__':
    unittest.main()
