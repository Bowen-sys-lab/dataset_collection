"""Run with Isaac's USD library on PYTHONPATH (no renderer required)."""
import unittest
import numpy as np

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:
    Usd = None


@unittest.skipIf(Usd is None, 'requires Isaac USD library')
class MountTests(unittest.TestCase):
    def setUp(self):
        from scripts.dual_camera_runtime import CameraMount
        self.CameraMount = CameraMount
        self.stage = Usd.Stage.CreateInMemory()
        self.body = UsdGeom.Xform.Define(self.stage, '/World/Drone/body')
        self.body.AddTranslateOp().Set((2, 3, 4))
        self.body.AddRotateZOp().Set(90)
        mount = UsdGeom.Xform.Define(self.stage, '/World/Drone/body/mount')
        mount.AddTranslateOp().Set((.1, .2, .3))
        self.source = UsdGeom.Camera.Define(self.stage, '/World/Drone/body/mount/Camera')
        self.source.AddTranslateOp().Set((.4, 0, 0))
        self.source.AddRotateYOp().Set(30)
        self.source.GetFocalLengthAttr().Set(25)
        self.original = self.stage.GetRootLayer().ExportToString()

    def test_nested_mount_is_preserved_in_both_camera_world_poses(self):
        rig = self.CameraMount(self.stage, str(self.source.GetPath()), str(self.body.GetPath()))
        for path in (rig.rgb_path, rig.thermal_path):
            camera = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            np.testing.assert_allclose(camera.ComputeLocalToWorldTransform(0),
                                       self.source.ComputeLocalToWorldTransform(0), atol=1e-8)
            self.assertEqual(camera.GetFocalLengthAttr().Get(), 25)
        self.body.GetOrderedXformOps()[0].Set((7, 8, 9))
        for path in (rig.rgb_path, rig.thermal_path):
            np.testing.assert_allclose(UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0),
                                       self.source.ComputeLocalToWorldTransform(0), atol=1e-8)
        rig.close()

    def test_cleanup_restores_edit_target_and_keeps_source_unchanged(self):
        original_target = self.stage.GetEditTarget().GetLayer()
        rig = self.CameraMount(self.stage, str(self.source.GetPath()), str(self.body.GetPath()))
        self.assertTrue(self.stage.GetPrimAtPath(rig.rgb_path))
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), self.original)
        rig.close()
        rig.close()
        self.assertTrue(self.stage.GetPrimAtPath(rig.rgb_path))
        self.assertFalse(self.stage.GetPrimAtPath(rig.thermal_path))
        self.assertEqual(self.stage.GetEditTarget().GetLayer(), original_target)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), self.original)

    def test_existing_viewport_camera_does_not_block_startup_or_get_overwritten(self):
        # Kit recreates a camera targeted by a saved viewport, before our rig starts.
        saved = UsdGeom.Camera.Define(self.stage, '/World/Drone/body/Sensors/ThermalCamera')
        saved.AddRotateZOp().Set(17)
        for group in ('Sensors', 'Sensors_preview', 'Sensors_preview_1'):
            UsdGeom.Camera.Define(self.stage, f'/World/Drone/body/{group}/RGBCamera')
            if group != 'Sensors':
                UsdGeom.Camera.Define(self.stage, f'/World/Drone/body/{group}/ThermalCamera')
        from pxr import UsdRender
        product = UsdRender.Product.Define(self.stage, '/Render/Old')
        old_camera = '/World/Drone/body/Sensors_preview_1/RGBCamera'
        product.GetCameraRel().SetTargets([old_camera])
        saved_pose = saved.ComputeLocalToWorldTransform(0)
        before = self.stage.GetRootLayer().ExportToString()
        rig = self.CameraMount(self.stage, str(self.source.GetPath()), str(self.body.GetPath()))
        self.assertEqual(rig.root_path, '/World/Drone/body/Sensors')
        self.assertEqual(rig.rgb_path, str(self.source.GetPath()))
        cameras = [str(p.GetPath()) for p in self.stage.Traverse() if p.IsA(UsdGeom.Camera)]
        self.assertEqual(set(cameras), {rig.rgb_path, rig.thermal_path})
        self.assertEqual([str(p) for p in product.GetCameraRel().GetTargets()], [rig.rgb_path])
        np.testing.assert_allclose(saved.ComputeLocalToWorldTransform(0), saved_pose, atol=1e-8)
        self.assertEqual(saved.GetOrderedXformOps()[0].Get(), 17)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)
        rig.close()
        self.assertTrue(self.stage.GetPrimAtPath(saved.GetPath()))
        self.assertEqual([str(p) for p in product.GetCameraRel().GetTargets()], [old_camera])
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)

    def test_empty_camera_placeholder_uses_reference_pose_with_transformed_container(self):
        root = UsdGeom.Xform.Define(self.stage, '/World/Drone/body/Sensors')
        root.AddTranslateOp().Set((1, 2, 3))
        UsdGeom.Camera.Define(self.stage, str(root.GetPath()) + '/ThermalCamera')
        before = self.stage.GetRootLayer().ExportToString()
        for _ in range(2):
            rig = self.CameraMount(self.stage, str(self.source.GetPath()), str(self.body.GetPath()))
            np.testing.assert_allclose(UsdGeom.Xformable(self.stage.GetPrimAtPath(rig.thermal_path)).ComputeLocalToWorldTransform(0),
                                       self.source.ComputeLocalToWorldTransform(0), atol=1e-8)
            self.assertEqual(len([p for p in self.stage.Traverse() if p.IsA(UsdGeom.Camera)]), 2)
            rig.close()
            self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)

    def test_rejects_camera_outside_moving_link(self):
        UsdGeom.Xform.Define(self.stage, '/World/Other')
        with self.assertRaises(ValueError):
            self.CameraMount(self.stage, str(self.source.GetPath()), '/World/Other')

    def test_opt_in_alignment_overrides_saved_pose_only_in_session(self):
        saved = UsdGeom.Camera.Define(self.stage, '/World/Drone/body/Sensors/ThermalCamera')
        saved.AddRotateZOp().Set(17)
        pose = saved.ComputeLocalToWorldTransform(0)
        before = self.stage.GetRootLayer().ExportToString()
        rig = self.CameraMount(self.stage, str(self.source.GetPath()), str(self.body.GetPath()),
                               align_thermal_to_rgb=True)
        np.testing.assert_allclose(saved.ComputeLocalToWorldTransform(0),
                                   self.source.ComputeLocalToWorldTransform(0), atol=1e-8)
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), before)
        rig.close()
        np.testing.assert_allclose(saved.ComputeLocalToWorldTransform(0), pose, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
