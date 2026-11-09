"""CPU USD regression: thermal partitions must conserve geometry and RGB materials."""
import sys
from pathlib import Path
import unittest
import tempfile

sys.path.insert(0, 'D:/isaacsim/extscache/omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312')
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from scripts.city_material_partition import partition_mesh, verify_overlay_source


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        self.mesh = UsdGeom.Mesh.Define(self.stage, '/World/Combined')
        self.mesh.CreateSubdivisionSchemeAttr('none')
        self.mesh.CreatePointsAttr([(0,0,0), (1,0,0), (1,1,0), (0,1,0)])
        self.mesh.CreateFaceVertexCountsAttr([3,3])
        self.mesh.CreateFaceVertexIndicesAttr([0,1,2, 0,2,3])
        self.mesh.CreateNormalsAttr([(0,0,1)]*6)
        self.mesh.SetNormalsInterpolation('faceVarying')
        self.mesh.AddTranslateOp().Set((3,4,5))
        uv = UsdGeom.PrimvarsAPI(self.mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying')
        uv.Set([(0,0),(1,0),(1,1),(0,1)])
        uv.SetIndices([0,1,2,0,2,3])
        for name, face in [('Glass',0), ('Wall',1)]:
            mat = UsdShade.Material.Define(self.stage, '/World/Looks/'+name)
            sub = UsdShade.MaterialBindingAPI(self.mesh).CreateMaterialBindSubset(name, [face])
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
        self.stage.SetEditTarget(self.stage.GetSessionLayer())

    def test_preserves_faces_uvs_rgb_materials_transform_and_source_layer(self):
        source = self.stage.GetRootLayer().ExportToString()
        parts = partition_mesh(self.stage, self.mesh.GetPath(), lambda m: m.GetPrim().GetName().lower())
        self.assertEqual({p['category'] for p in parts}, {'glass','wall'})
        self.assertEqual(self.mesh.ComputeVisibility(), 'invisible')
        original_uv = UsdGeom.PrimvarsAPI(self.mesh).GetPrimvar('st').ComputeFlattened()
        for part in parts:
            mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(part['path']))
            face = 0 if part['category']=='glass' else 1
            self.assertEqual(list(mesh.GetFaceVertexCountsAttr().Get()), [3])
            self.assertEqual(mesh.ComputeVisibility(), 'inherited')
            self.assertEqual(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                             self.mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            pts = mesh.GetPointsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            original = self.mesh.GetFaceVertexIndicesAttr().Get()[face*3:face*3+3]
            self.assertEqual([pts[i] for i in indices], [self.mesh.GetPointsAttr().Get()[i] for i in original])
            self.assertEqual(list(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').ComputeFlattened()), list(original_uv[face*3:face*3+3]))
            subs = UsdShade.MaterialBindingAPI(mesh).GetMaterialBindSubsets()
            self.assertEqual(len(subs), 1)
            self.assertEqual(list(subs[0].GetIndicesAttr().Get()), [0])
            self.assertEqual(UsdShade.MaterialBindingAPI(subs[0]).ComputeBoundMaterial()[0].GetPrim().GetName().lower(), part['category'])
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), source)

    def test_overlap_rejected_without_hiding_original(self):
        sub = UsdShade.MaterialBindingAPI(self.mesh).CreateMaterialBindSubset('Overlap', [0])
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(UsdShade.Material(self.stage.GetPrimAtPath('/World/Looks/Wall')))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            partition_mesh(self.stage, self.mesh.GetPath(), lambda m: m.GetPrim().GetName())
        self.assertEqual(self.mesh.ComputeVisibility(), 'inherited')


class OverlayProvenanceTests(unittest.TestCase):
    def test_pinned_source_change_is_rejected_even_when_overlay_is_unchanged(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.usda';overlay=root/'derived.usda'
            original=Usd.Stage.CreateNew(str(source));UsdGeom.Xform.Define(original,'/World');original.GetRootLayer().Save()
            layer=Sdf.Layer.CreateNew(str(overlay));layer.subLayerPaths=['./source.usda']
            layer.customLayerData={'thermalRevision':'city_material_partition_v3',
                'sourceStageSha256':hashlib.sha256(source.read_bytes()).hexdigest()};layer.Save()
            stage=Usd.Stage.Open(layer)
            self.assertIsNotNone(verify_overlay_source(stage))
            self.assertEqual(verify_overlay_source(stage)['source_stage_sha256'],layer.customLayerData['sourceStageSha256'])
            overlay_bytes=overlay.read_bytes()
            source.write_bytes(source.read_bytes()+b'\n# source changed\n')
            with self.assertRaisesRegex(ValueError,'source USD changed'):
                verify_overlay_source(stage)
            self.assertEqual(overlay.read_bytes(),overlay_bytes)

    def test_legacy_stage_needs_no_overlay_pin(self):
        self.assertIsNone(verify_overlay_source(Usd.Stage.CreateInMemory()))


if __name__ == '__main__': unittest.main()

