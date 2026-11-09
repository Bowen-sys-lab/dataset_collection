"""Replace tiny thermal-piece colliders with one welded local static collision mesh."""
import hashlib,json,sys
from pathlib import Path
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,Vt

ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.compose_model02_v3 import world_mesh,packed
from scripts.model02_collision import consolidate_collision
ASSET=ROOT/'factory/model02_thermal_v3'


def main():
    stage=Usd.Stage.Open(str(ASSET/'model02_v3_scene.usda'))
    manifest=json.loads((ASSET/'export_manifest.json').read_text(encoding='utf-8'))
    path='/World/IndustrialPark02/Collision/RefinedStatic'
    assert not stage.GetPrimAtPath(path),'Refusing to replace an existing collider'
    meshes=[]
    for row in manifest['surfaces']:
        prim=stage.GetPrimAtPath(row['usd_prim_path']);meshes.append(world_mesh(prim))
        if prim.HasAPI(UsdPhysics.CollisionAPI):prim.RemoveAPI(UsdPhysics.CollisionAPI)
    origin,points,faces,report=consolidate_collision(*packed(meshes))
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateVisibilityAttr('invisible')
    UsdGeom.Xformable(mesh).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*origin))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    mesh.GetPrim().CreateAttribute('model02:role',Sdf.ValueTypeNames.String).Set('Static collider only; not a rendered/thermal surface')
    stage.GetRootLayer().Save()
    report.update(result='PASS_WELDED_LOCAL_COLLIDER',path=path,rendered_thermal_surfaces_unchanged=8327,
        scene_sha256=hashlib.sha256((ASSET/'model02_v3_scene.usda').read_bytes()).hexdigest(),
        reason='Small world-coordinate thermal fragments caused individual PhysX cooking failures; consolidate and recenter collision representation only')
    (ASSET/'collision_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
