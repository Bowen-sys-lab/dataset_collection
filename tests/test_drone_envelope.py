import sys
import unittest
import numpy as np

sys.path.insert(0, 'D:/isaacsim/extscache/omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312')
from pxr import Usd, UsdGeom, UsdPhysics, Gf


class ComposedEnvelopeTests(unittest.TestCase):
    def stage(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.Xform.Define(stage, '/World/Drone')
        UsdGeom.Xform.Define(stage, '/World/Drone/body')
        UsdGeom.Cube.Define(stage, '/World/Drone/body/Hull').CreateSizeAttr(.02)
        for index, (x, y) in enumerate([(.03,.03),(.03,-.03),(-.03,.03),(-.03,-.03)]):
            path = f'/World/Drone/m{index}_prop'
            UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set((x,y,.02))
            mesh = UsdGeom.Mesh.Define(stage, path+'/Blade')
            mesh.CreatePointsAttr([(-.02,0,0),(.02,0,0),(0,.002,.001)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0,1,2])
            joint = UsdPhysics.RevoluteJoint.Define(stage, f'/World/Drone/joint{index}')
            joint.CreateBody0Rel().SetTargets(['/World/Drone/body'])
            joint.CreateBody1Rel().SetTargets([path])
        return stage

    def measure(self, stage):
        from scripts.drone_envelope import measure_drone_envelope
        return measure_drone_envelope(stage, '/World/Drone', '/World/Drone/body')

    def test_four_rotors_and_primitive_hull_are_covered(self):
        result = self.measure(self.stage())
        self.assertEqual(len(result['rotors']), 4)
        self.assertLess(result['rotation_envelope_radius_m'], .08)

    def test_flight_layer_child_override_changes_measured_envelope(self):
        stage = self.stage()
        original = self.measure(stage)
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Drone/body/Hull')).AddTranslateOp().Set((.5,0,0))
        changed = self.measure(stage)
        self.assertGreater(changed['rotation_envelope_radius_m'], .49)
        self.assertNotEqual(original['geometry_sha256'], changed['geometry_sha256'])

    def test_missing_rotor_does_not_produce_complete_evidence(self):
        stage = self.stage()
        stage.RemovePrim('/World/Drone/joint0')
        with self.assertRaises(ValueError):
            self.measure(stage)

    def test_scaled_or_sheared_body_cannot_use_a_metre_sphere(self):
        for matrix in [np.diag([2.,2.,2.,1.]),
                       np.array([[1,0,0,0],[.6,.8,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)]:
            with self.subTest(matrix=matrix.tolist()):
                stage = self.stage()
                UsdGeom.Xformable(stage.GetPrimAtPath('/World/Drone')).AddTransformOp().Set(Gf.Matrix4d(matrix.tolist()))
                with self.assertRaises(ValueError):
                    self.measure(stage)


if __name__ == '__main__':
    unittest.main()
