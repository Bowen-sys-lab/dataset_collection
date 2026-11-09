import sys
import unittest
import numpy as np

sys.path.insert(0, 'D:/isaacsim/extscache/omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312')
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux


def box_mesh(stage, path='/World/Obstacle'):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                          (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)])
    mesh.CreateFaceVertexCountsAttr([3]*12)
    mesh.CreateFaceVertexIndicesAttr([0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
                                     1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7])
    mesh.CreateSubdivisionSchemeAttr('none')
    return mesh


class CollisionSceneTests(unittest.TestCase):
    def scene(self, start=(4,0,0), edit=None):
        from scripts.collision_scene import build_collision_scene
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.Xform.Define(stage, '/World')
        mesh = box_mesh(stage)
        if edit:
            edit(stage, mesh)
        return build_collision_scene(stage, start=start, bounds=[[-6,-6,-6],[6,6,6]], radius=.1)

    def test_escape_proves_start_outside_closed_box(self):
        scene = self.scene()
        self.assertTrue(scene.report['ready'])
        self.assertEqual(scene.report['start_escape']['status'], 'CLEAR_SURFACES')

    def test_inside_start_is_not_released_even_far_from_wall(self):
        scene = self.scene(start=(0,0,0))
        self.assertFalse(scene.report['ready'])

    def test_contiguous_path_checks_mesh_contact_and_bounds(self):
        from scripts.collision_scene import check_path
        scene = self.scene()
        self.assertEqual(check_path(scene, [[4,0,0],[4,2,0]])['status'], 'CLEAR')
        self.assertEqual(check_path(scene, [[4,0,0],[-4,0,0]])['status'], 'CONTACT')
        self.assertEqual(check_path(scene, [[4,0,0],[7,0,0]])['status'], 'OUT_OF_BOUNDS')
        self.assertEqual(check_path(scene, [[0,0,0],[4,0,0]])['status'], 'START_MISMATCH')

    def test_unsupported_solid_blocks_gate_but_light_does_not(self):
        self.assertFalse(self.scene(edit=lambda stage, mesh: UsdGeom.Cube.Define(stage, '/World/Unknown')).report['ready'])
        self.assertTrue(self.scene(edit=lambda stage, mesh: UsdLux.RectLight.Define(stage, '/World/Light')).report['ready'])

    def test_animated_ancestor_blocks_gate(self):
        def edit(stage, mesh):
            op = UsdGeom.Xformable(stage.GetPrimAtPath('/World')).AddTranslateOp()
            op.Set((0,0,0), 0)
            op.Set((1,0,0), 1)
        self.assertFalse(self.scene(edit=edit).report['ready'])

    def test_hidden_enabled_collision_mesh_is_included(self):
        def edit(stage, mesh):
            mesh.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        from scripts.collision_scene import check_path
        scene = self.scene(edit=edit)
        self.assertEqual(check_path(scene, [[4,0,0],[-4,0,0]])['status'], 'CONTACT')

    def test_robot_name_prefix_does_not_hide_other_obstacle(self):
        def edit(stage, mesh):
            other = box_mesh(stage, '/World/CrazyflieObstacle')
            UsdGeom.Xformable(other.GetPrim()).AddTranslateOp().Set((4,0,0))
        self.assertFalse(self.scene(edit=edit).report['ready'])

    def test_empty_authored_placeholder_is_explicitly_excluded(self):
        def edit(stage, mesh):
            empty = UsdGeom.Mesh.Define(stage, '/World/Empty')
            empty.CreatePointsAttr([])
            empty.CreateFaceVertexCountsAttr([])
            empty.CreateFaceVertexIndicesAttr([])
            empty.CreateSubdivisionSchemeAttr('none')
        scene = self.scene(edit=edit)
        self.assertTrue(scene.report['ready'])
        self.assertIn('/World/Empty', [item['path'] for item in scene.report['explicit_exclusions']])

    def test_invisible_mesh_that_appears_later_blocks_gate(self):
        def edit(stage, mesh):
            anchor = box_mesh(stage, '/World/Anchor')
            UsdGeom.Xformable(anchor.GetPrim()).AddTranslateOp().Set((0,5,0))
            visibility = mesh.CreateVisibilityAttr('invisible')
            visibility.Set('invisible', 0)
            visibility.Set('inherited', 1)
        self.assertFalse(self.scene(edit=edit).report['ready'])


if __name__ == '__main__':
    unittest.main()
