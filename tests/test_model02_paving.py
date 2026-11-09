"""Physical paving partitions, continuous field and manual-search contracts."""
import unittest
import numpy as np
from scripts.blender.model02_geometry import polygon_area
try:
    from scripts import model02_paving as paving
except ImportError:
    paving = None


class PavingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(paving, 'paving partition implementation is missing')

    def test_tile_and_joint_partition_preserves_area_without_overlaps(self):
        poly = [np.array([x,y,0.,x,y,0.,0.,1.]) for x,y in
                [(-.1,-.1),(.9,-.1),(.9,.9),(-.1,.9)]]
        parts = paving.partition(poly)
        self.assertAlmostEqual(sum(polygon_area(p) for key,p in parts), 1., places=9)
        self.assertTrue(any(key[2]=='joint' for key,p in parts))
        self.assertTrue(any(key[2]=='tile' for key,p in parts))
        for key,p in parts:
            self.assertGreater(polygon_area(p), 1e-12)
            for v in p:
                np.testing.assert_allclose(v[3:5], v[:2], atol=1e-12)

    def test_each_brick_has_four_independent_quadrants(self):
        poly = [np.array([x,y,0.,0.,0.,0.,0.,1.]) for x,y in
                [(0.,0.),(.608,0.),(.608,.608),(0.,.608)]]
        keys = {key for key,p in paving.partition(poly)}
        self.assertEqual({key[3] for key in keys if key[2]=='tile'}, {0,1,2,3})

    def test_temperatures_have_small_explainable_differences(self):
        tiles=[paving.temperature(x,y,'tile') for x in range(630,735,5) for y in range(-865,-780,5)]
        joints=[paving.temperature(x,y,'joint') for x in range(630,735,5) for y in range(-865,-780,5)]
        self.assertTrue(21.7<=min(tiles)<max(tiles)<=22.3)
        self.assertTrue(21.5<=min(joints)<max(joints)<=22.)
        self.assertLess(abs(paving.temperature(700,-830,'tile')-paving.temperature(700.6,-830,'tile')), .2)
        self.assertEqual(paving.temperature(700,-830,'tile'),paving.temperature(700,-830,'tile'))

    def test_v4_runtime_weather_uses_the_authored_condition(self):
        from scripts import model02_v4_configuration as config
        compile_environment=getattr(config,'compile_night_environment',None)
        self.assertIsNotNone(compile_environment,'v4 runtime weather propagation is missing')
        template={'scene':{'name':'old','weather_file':'old.csv','start_utc':'old'}}
        condition={'air_temperature_c':25,'indoor_temperature_c':24,'wind_m_s':2,
                   'hours_after_sunset':3,'weather':'dry','equipment_state':'operating'}
        environment,text=compile_environment(template,condition,'D:/scene/v4_weather.csv')
        self.assertEqual(environment['scene']['weather_file'],'D:/scene/v4_weather.csv')
        self.assertEqual(environment['scene']['start_utc'],'2026-10-18T13:00:00Z')
        self.assertIn(',25.0,0.60,2.0,',text)
        self.assertEqual(template['scene']['weather_file'],'old.csv')


if __name__=='__main__':unittest.main()
