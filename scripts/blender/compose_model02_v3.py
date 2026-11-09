"""Compose new v3 stage and measure actual exported scene geometry."""
import hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdPhysics,UsdShade
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import tessellate_polygon

ROOT=Path('D:/IsaacSim_Demo');ASSET=ROOT/'factory/model02_thermal_v3'


def world_mesh(prim):
    mesh=UsdGeom.Mesh(prim)
    p=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    m=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0),dtype=float)
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32)
    if np.all(counts==3):faces=indices.reshape(-1,3)
    else:
        faces=[];offset=0
        for count in counts:
            ids=indices[offset:offset+count];offset+=count
            if count==3:faces.append(ids)
            else:
                polygon=[Vector(p[i]) for i in ids]
                faces.extend([ids[i] for i in tri] for tri in tessellate_polygon([polygon]))
        faces=np.asarray(faces,dtype=np.int32)
    return (np.c_[p,np.ones(len(p))]@m)[:,:3],faces


def packed(meshes):
    vertices=[];faces=[];offset=0
    for p,f in meshes:
        vertices.append(p);faces.append(f+offset);offset+=len(p)
    return np.concatenate(vertices),np.concatenate(faces)


def main():
    export=json.loads((ASSET/'export_manifest.json').read_text(encoding='utf-8'))
    assert export['result']=='PASS_ACTUAL_USD_EXPORT'
    path=ASSET/'model02_v3_scene.usda';assert not path.exists()
    stage=Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().subLayerPaths=['./model02_v3_asset.usdc','../factory_inspection.usda']
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1)
    stage.SetTimeCodesPerSecond(60);stage.SetStartTimeCode(0);stage.SetEndTimeCode(1000000)
    inactive=[]
    for source in export['replaced_blender_sources']:
        original='/World/LimeExp_02/node_1_29_gc'+source.split('gc')[-1]
        assert stage.GetPrimAtPath(original),original
        stage.OverridePrim(original).SetActive(False);inactive.append(original)
    stage.OverridePrim('/Environment/defaultLight').SetActive(False)
    lights=[]
    for name in export['lights']:
        prim=stage.GetPrimAtPath(name);assert prim and prim.HasAPI(UsdLux.LightAPI)
        prim.CreateAttribute('omni:rtx:usdluxVersion',Sdf.ValueTypeNames.Int).Set(2505)
        intensity=prim.GetAttribute('inputs:intensity');intensity.Set(float(intensity.Get())*100)
        lights.append({'path':name,'intensity':intensity.Get()})
    moon=UsdLux.DomeLight.Define(stage,'/World/IndustrialPark02/Lighting/NightSky')
    moon.CreateIntensityAttr(.8);moon.CreateColorAttr(Gf.Vec3f(.25,.36,.6))
    moon.GetPrim().CreateAttribute('omni:rtx:usdluxVersion',Sdf.ValueTypeNames.Int).Set(2505)
    old=Usd.Stage.Open(str(ASSET.parent/'model02_thermal_v2/model02_refined_scene.usda'))
    settings=old.GetRootLayer().customLayerData['uavLighting']['settings']
    rows=export['surfaces'];by_path={r['usd_prim_path']:r for r in rows}
    meshes={};all_meshes=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible:continue
        p,f=world_mesh(prim);all_meshes.append((p,f))
        name=str(prim.GetPath())
        if name in by_path:
            meshes[by_path[name]['surface_id']]=(p,f)
            UsdPhysics.CollisionAPI.Apply(prim)
    assert len(meshes)==8327
    points,faces=packed(all_meshes)
    np.savez_compressed(ASSET/'scene_geometry.npz',vertices=points,triangles=faces)
    origin=np.array([700.,-800.,0.])
    bvh=BVHTree.FromPolygons((points-origin).tolist(),faces.tolist(),all_triangles=True)
    tasks=json.loads((ASSET/'inspection_tasks.json').read_text(encoding='utf-8'))['tasks'];task_reports=[]
    for task in tasks:
        kind=task['task_id'].split('_')[-1];source='11_29_gc075' if kind=='tank' else '11_29_gc089'
        core_ids=task['target_surface_ids'];core=packed([meshes[sid] for sid in core_ids])
        equipment=packed([meshes[r['surface_id']] for r in rows if r['source_object']==source])
        geometry=ASSET/(kind+'_inspection_geometry.npz')
        np.savez_compressed(geometry,equipment_vertices=equipment[0],equipment_triangles=equipment[1],hotspot_vertices=core[0],hotspot_triangles=core[1])
        triangles=core[0][core[1]];areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
        center=np.average(triangles.mean(1),axis=0,weights=areas)
        samples=[]
        for a,b in zip(task['reference_waypoints_m'],task['reference_waypoints_m'][1:]):
            a=np.array(a);b=np.array(b);n=math.ceil(np.linalg.norm(b-a)/.25)
            samples.extend(a+(b-a)*(i/n) for i in range(n+1))
        nearest=[bvh.find_nearest(Vector(p-origin))[3] for p in samples]
        clearance=min(nearest)-.2-.125-.001
        assert clearance>0,kind
        eye=np.array(task['observation_position_m']);visible=0
        for c in triangles.mean(1):
            d=c-eye;distance=float(np.linalg.norm(d));hit=bvh.ray_cast(Vector(eye-origin),Vector(d/distance),distance+.003)
            if hit[0] is not None and abs(hit[3]-distance)<.004:visible+=1
        assert visible>0,kind
        own_bvh=BVHTree.FromPolygons((equipment[0]-origin).tolist(),equipment[1].tolist(),all_triangles=True)
        obs_clearance=own_bvh.find_nearest(Vector(eye-origin))[3]-.2
        hotpaths=[r['usd_prim_path'] for r in rows if r['surface_id'] in core_ids]
        normal=next(r['usd_prim_path'] for r in rows if r['source_object']==source and 'Hotspot' not in r['surface_id'] and 'Transition' not in r['surface_id'])
        representative=hotpaths[0];material=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(representative)).ComputeBoundMaterial()[0]
        task_reports.append(dict(task,kind=kind,target_center_m=center.tolist(),hotspot_area_m2=float(areas.sum()),
            target_prim_path=representative,hotspot_prim_paths=hotpaths,normal_surface_prim_path=normal,
            equipment_prim_path=str(stage.GetPrimAtPath(representative).GetParent().GetParent().GetPath()),
            rgb_material_path=str(material.GetPath()),geometry_file=geometry.relative_to(ROOT).as_posix(),
            geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
            minimum_route_body_clearance_lower_bound_m=clearance,route_samples=len(samples),
            observation_equipment_clearance_m=obs_clearance,visible_core_triangles=visible,total_core_triangles=len(triangles),
            camera_to_hotspot_m=float(np.linalg.norm(center-eye)),reference_path_length_m=float(np.linalg.norm(np.diff(task['reference_waypoints_m'],axis=0),axis=1).sum())))
    dependencies={}
    for layer in stage.GetUsedLayers():
        if layer.realPath and Path(layer.realPath).resolve()!=path.resolve():
            dependencies[os.path.relpath(layer.realPath,ASSET).replace('\\','/')]=hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
    for texture in (ASSET/'textures').glob('*'):
        if texture.is_file():dependencies[os.path.relpath(texture,ASSET).replace('\\','/')]=hashlib.sha256(texture.read_bytes()).hexdigest()
    stage.GetRootLayer().customLayerData={'uavLighting':{'preset':'model02_v3_operating_night','settings':settings,'dependencies':dependencies,
        'note':'Estimated dry night20C,13 fixtures plus sky; prescribed independent surface temperatures, no photometric calibration'}}
    stage.GetRootLayer().Save()
    report={'result':'PASS_COMPOSED_USD_GEOMETRY','scene':str(path),'surfaces':rows,'inactive_original_nodes':inactive,
        'lights':lights,'settings':settings,'dependencies':dependencies,'tasks':task_reports,
        'full_scene_triangles':len(faces),'scene_geometry_sha256':hashlib.sha256((ASSET/'scene_geometry.npz').read_bytes()).hexdigest(),
        'clearance_method':'Full visible exported scene BVH, route step<=0.25m, body radius0.2m, subtract0.125m Lipschitz interval bound and0.001m numeric allowance',
        'intensity_multiplier':100.,'intensity_note':'Retained empirical v2 renderer conversion, not calibrated lux'}
    (ASSET/'isaac_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['surfaces','tasks','dependencies']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
