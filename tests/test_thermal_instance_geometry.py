"""Real USD traversal regression: instances must not lose infrared coverage."""
import unittest
from pxr import Usd, UsdGeom
from scripts.thermal_scene_config import renderable_paths


class InstanceGeometryTests(unittest.TestCase):
    def test_visible_proxy_surfaces_retained_hidden_surfaces_excluded(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/World')
        UsdGeom.Xform.Define(stage, '/Asset')
        UsdGeom.Cube.Define(stage, '/Asset/Surface')
        hidden = UsdGeom.Cube.Define(stage, '/Asset/Hidden')
        hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        instance = UsdGeom.Xform.Define(stage, '/World/Building').GetPrim()
        instance.GetReferences().AddInternalReference('/Asset')
        instance.SetInstanceable(True)
        self.assertTrue(stage.GetPrimAtPath('/World/Building/Surface').IsInstanceProxy())
        self.assertEqual(renderable_paths(stage), ['/World/Building/Surface'])
        self.assertTrue(instance.IsInstance())  # No mutation/de-instancing.


if __name__ == '__main__':
    unittest.main()
