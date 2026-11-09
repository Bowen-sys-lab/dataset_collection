"""Compose only the refined Blender parts over the preserved factory background."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOT = Path('D:/IsaacSim_Demo')
ASSETS = ROOT/'factory/model02_thermal_v2'
REFINED = ['021','075','076','077','078']


def world_mesh(prim):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0),dtype=float)
    world = (np.c_[points,np.ones(len(points))]@transform)[:,:3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    assert np.all(counts==3), str(prim.GetPath())
    return world,indices.reshape(-1,3)


def build():
    path = ASSETS/'model02_refined_scene.usda'
    assert not path.exists(), 'Refusing to overwrite a released scene'
    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().subLayerPaths = ['./model02_refined_asset.usdc','../factory_inspection.usda']
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0);stage.SetEndTimeCode(1000000)
    for source_id in REFINED:
        stage.OverridePrim('/World/LimeExp_02/node_1_29_gc'+source_id).SetActive(False)
    stage.OverridePrim('/Environment/defaultLight').SetActive(False)
    surfaces = []
    lights = []
    vertices = []
    triangles = []
    hot_vertices = None
    hot_triangles = None
    equipment_path = '/World/IndustrialPark02/InspectionZone01/Equipment/TANK_GC075'
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith('/World/IndustrialPark02/'):
            continue
        if prim.HasAPI(UsdLux.LightAPI):
            prim.CreateAttribute('omni:rtx:usdluxVersion',Sdf.ValueTypeNames.Int).Set(2505)
            intensity = prim.GetAttribute('inputs:intensity')
            intensity.Set(float(intensity.Get())*100.)
            lights.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),
                           'intensity':intensity.Get(),'usd_lux_version':2505})
        if not prim.IsA(UsdGeom.Mesh):
            continue
        parent = prim.GetParent()
        props = {name:parent.GetAttribute('userProperties:'+name).Get() for name in
                 ['surface_id','component','ir_material','temperature_target_id','temperature_c']}
        assert all(value is not None for value in props.values()), str(prim.GetPath())
        props.update(usd_prim_path=str(prim.GetPath()))
        surfaces.append(props)
        UsdPhysics.CollisionAPI.Apply(prim)
        if str(prim.GetPath()).startswith(equipment_path+'/'):
            points,faces = world_mesh(prim)
            offset = sum(len(v) for v in vertices)
            vertices.append(points);triangles.append(faces+offset)
            if props['surface_id']=='GC075_TankShell_Hotspot':
                hot_vertices,hot_triangles = points,faces
                hotspot_path = str(prim.GetPath())
    assert hot_vertices is not None
    hot_faces = hot_vertices[hot_triangles]
    areas = np.linalg.norm(np.cross(hot_faces[:,1]-hot_faces[:,0],hot_faces[:,2]-hot_faces[:,0]),axis=1)/2
    hot_center = np.average(hot_faces.mean(axis=1),axis=0,weights=areas)
    normal_path = next(s['usd_prim_path'] for s in surfaces if s['surface_id']=='GC075_TankShell_Normal_South')
    comparison_path = next(s['usd_prim_path'] for s in surfaces if s['surface_id']=='GC076_TankShell_Normal_South')
    rgb_material = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(hotspot_path)).ComputeBoundMaterial()[0]
    geometry = ASSETS/'factory_inspection_geometry.npz'
    np.savez_compressed(geometry,equipment_vertices=np.concatenate(vertices),
        equipment_triangles=np.concatenate(triangles),hotspot_vertices=hot_vertices,hotspot_triangles=hot_triangles)
    sky = UsdLux.DomeLight.Define(stage,'/World/IndustrialPark02/Lighting/NightSky')
    sky.CreateIntensityAttr(.8);sky.CreateColorAttr(Gf.Vec3f(.25,.36,.6))
    sky.GetPrim().CreateAttribute('omni:rtx:usdluxVersion',Sdf.ValueTypeNames.Int).Set(2505)
    old = Usd.Stage.Open(str(ASSETS.parent/'model02_thermal_v1/model02_weaklight.usda'))
    exposure = old.GetRootLayer().customLayerData['uavLighting']['settings']
    dependencies = {}
    for layer in stage.GetUsedLayers():
        if layer.realPath and Path(layer.realPath).resolve()!=path.resolve():
            dependencies[os.path.relpath(layer.realPath,ASSETS).replace('\\','/')] = hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
    for texture in (ASSETS/'textures').glob('*'):
        if texture.is_file():
            dependencies[os.path.relpath(texture,ASSETS).replace('\\','/')] = hashlib.sha256(texture.read_bytes()).hexdigest()
    stage.GetRootLayer().customLayerData = {'uavLighting':{
        'preset':'low_light_refined','settings':exposure,'dependencies':dependencies,
        'note':'13 fixtures, 8 geometric windows, 7 doors, 4 component-separated tanks; local prescribed heat anomaly'}}
    stage.GetRootLayer().Save()
    report = {'stage':str(path),'surfaces':surfaces,'lights':lights,'settings':exposure,'dependencies':dependencies,
        'inactive_original_nodes':['/World/LimeExp_02/node_1_29_gc'+n for n in REFINED],
        'equipment_prim_path':equipment_path,'hotspot_prim_path':hotspot_path,'hotspot_center_m':hot_center.tolist(),
        'normal_surface_prim_path':normal_path,'comparison_equipment_prim_path':comparison_path,
        'rgb_material_path':str(rgb_material.GetPath()),
        'hotspot_area_m2':float(areas.sum()),'geometry_file':'factory/model02_thermal_v2/factory_inspection_geometry.npz',
        'geometry_sha256':hashlib.sha256(geometry.read_bytes()).hexdigest(),
        'blender_to_isaac_intensity_multiplier':100.,'source_surface_materials':'RGB shared; IR material/temperature independent'}
    (ASSETS/'refined_asset_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'usd':str(path),'explicit_surfaces':len(surfaces),'fixture_and_moon_lights':len(lights),
            'hotspot_center':hot_center.tolist(),'hotspot_area_m2':float(areas.sum()),
            'old_hotspot_inactive':not stage.GetPrimAtPath('/World/LimeExp_02/node_1_29_gc075').IsActive()}
