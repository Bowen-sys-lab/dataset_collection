"""Real USD/texture checks for time-dependent, exception-safe IR surrogates."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = None

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipIf(Usd is None,'USD DLL runtime must be enabled for real-scene checks')
class ForestUsdProxyTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('forest_test_proxy',ROOT/'forest_fire_scene_local/scripts/fire_thermal_proxy.py')
        self.proxy=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.proxy)
        self.stage=Usd.Stage.Open(str(ROOT/'forest_fire_scene_local/scene.usda'))
        self.path='/World/Site/Target_BurningBarrel_001/Flames/Tongue_00'

    def test_proxy_samples_current_time_and_restores_animated_source(self):
        points=self.stage.GetPrimAtPath(self.path).GetAttribute('points')
        original=np.asarray(points.Get(0)).copy();samples=points.GetNumTimeSamples()
        layers=list(self.stage.GetSessionLayer().subLayerPaths)
        silhouettes=[]
        for tc in (0,24):
            with self.proxy.thermal_flame_surfaces(self.stage,tc) as count:
                self.assertEqual(count,7)
                silhouettes.append(np.asarray(points.Get(tc)).copy())
                self.assertGreater(len(silhouettes[-1]),len(original))
            self.assertEqual(list(self.stage.GetSessionLayer().subLayerPaths),layers)
            self.assertEqual(points.GetNumTimeSamples(),samples)
            np.testing.assert_array_equal(np.asarray(points.Get(0)),original)
        self.assertTrue(silhouettes[0].shape!=silhouettes[1].shape or
                        not np.array_equal(silhouettes[0],silhouettes[1]))

    def test_render_exception_cannot_leave_the_proxy_installed(self):
        layers=list(self.stage.GetSessionLayer().subLayerPaths)
        points=self.stage.GetPrimAtPath(self.path).GetAttribute('points')
        original=np.asarray(points.Get(24)).copy()
        with self.assertRaisesRegex(RuntimeError,'render failed'):
            with self.proxy.thermal_flame_surfaces(self.stage,24):
                raise RuntimeError('render failed')
        self.assertEqual(list(self.stage.GetSessionLayer().subLayerPaths),layers)
        np.testing.assert_array_equal(np.asarray(points.Get(24)),original)


if __name__=='__main__':unittest.main()
