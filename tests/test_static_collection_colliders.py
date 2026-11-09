"""USD-only checks: coverage, explicit VFX exclusion and source immutability."""
import unittest
from pxr import Usd, UsdGeom, UsdPhysics
from scripts.manual_collision_undo import prepare_static_colliders


class StaticCollectionColliderTests(unittest.TestCase):
    def test_shapes_meshes_and_instances_are_session_only(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/World/Drone')
        UsdGeom.Cube.Define(stage, '/World/Drone/body')
        UsdGeom.Cube.Define(stage, '/World/Box')
        UsdGeom.Cylinder.Define(stage, '/World/Pipe')
        UsdGeom.Mesh.Define(stage, '/World/Wall')
        effect = UsdGeom.Mesh.Define(stage, '/World/Effect')
        effect.GetPointsAttr().Set([(0, 0, 0)], 1.)
        before = stage.GetRootLayer().ExportToString()
        coverage = prepare_static_colliders(stage, '/World/Drone', exclude_paths=('/World/Effect',))
        self.assertEqual(coverage['mesh_paths'], ['/World/Wall'])
        self.assertEqual(set(coverage['shape_paths']), {'/World/Box', '/World/Pipe'})
        self.assertFalse(stage.GetPrimAtPath('/World/Drone/body').HasAPI(UsdPhysics.CollisionAPI))
        self.assertFalse(effect.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        self.assertTrue(stage.GetPrimAtPath('/World/Pipe').HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_animated_parent_transform_is_rejected(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/Drone')
        moving = UsdGeom.Xform.Define(stage, '/Moving')
        moving.AddTranslateOp().Set((1, 2, 3), 1.)
        UsdGeom.Cube.Define(stage, '/Moving/Box')
        with self.assertRaisesRegex(ValueError, 'Animated obstacle'):
            prepare_static_colliders(stage, '/Drone')

    def test_typo_in_exclusion_fails_explicitly(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/Drone')
        UsdGeom.Cube.Define(stage, '/Box')
        with self.assertRaisesRegex(ValueError, 'exclusion'):
            prepare_static_colliders(stage, '/Drone', exclude_paths=('/Typo',))


if __name__ == '__main__':
    unittest.main()
