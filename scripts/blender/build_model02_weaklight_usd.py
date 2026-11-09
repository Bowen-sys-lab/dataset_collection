"""Compose the first weak-light scene over the existing inspected factory asset."""
import hashlib
import json
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

ROOT = Path('D:/IsaacSim_Demo')
ASSETS = ROOT/'factory/model02_thermal_v1'
INTENSITY_MULTIPLIER = 100.
EXPOSURE = {'/rtx/post/tonemap/op': 6, '/rtx/post/tonemap/filmIso': 400.,
            '/rtx/post/tonemap/cameraShutter': 30., '/rtx/post/tonemap/fNumber': 2.8,
            '/rtx/post/histogram/enabled': False}


def build():
    output = ASSETS/'model02_weaklight.usda'
    if output.exists():
        raise FileExistsError(output)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = ['./model02_lighting.usdc', '../factory_inspection.usda']
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1000000)
    stage.OverridePrim('/Environment/defaultLight').SetActive(False)
    lighting_root = '/World/IndustrialPark02/Lighting'
    lights = []
    surfaces = []
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(lighting_root+'/'):
            continue
        if prim.HasAPI(UsdLux.LightAPI):
            prim.CreateAttribute('omni:rtx:usdluxVersion', Sdf.ValueTypeNames.Int).Set(2505)
            intensity = prim.GetAttribute('inputs:intensity')
            intensity.Set(float(intensity.Get())*INTENSITY_MULTIPLIER)
            lights.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),
                           'intensity':prim.GetAttribute('inputs:intensity').Get(),
                           'usd_lux_version':2505})
        if prim.IsA(UsdGeom.Mesh):
            parent = prim.GetParent()
            props = {name:parent.GetAttribute('userProperties:'+name).Get()
                     for name in ['surface_id','component','ir_material','temperature_target_id','temperature_c']}
            assert all(value is not None for value in props.values()), str(prim.GetPath())
            props['usd_prim_path'] = str(prim.GetPath())
            surfaces.append(props)
            UsdPhysics.CollisionAPI.Apply(prim)
    sky = UsdLux.DomeLight.Define(stage, lighting_root+'/NightSky')
    sky.CreateIntensityAttr(.008*INTENSITY_MULTIPLIER)
    sky.CreateColorAttr(Gf.Vec3f(.25,.36,.6))
    sky.GetPrim().CreateAttribute('omni:rtx:usdluxVersion', Sdf.ValueTypeNames.Int).Set(2505)
    camera = UsdGeom.Camera.Define(stage, '/World/InspectionViews/Overview')
    camera.CreateFocalLengthAttr(32.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateVerticalApertureAttr(24.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05,10000.))
    camera.MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(646.,-867.,44.),Gf.Vec3d(700.,-809.,3.),Gf.Vec3d(0,0,1)).GetInverse())
    dependencies = {}
    for layer in stage.GetUsedLayers():
        if not layer.realPath or Path(layer.realPath).resolve() == output.resolve():
            continue
        path = Path(layer.realPath).resolve()
        dependencies[os.path.relpath(path,ASSETS).replace('\\','/')] = hashlib.sha256(path.read_bytes()).hexdigest()
    stage.GetRootLayer().customLayerData = {'uavLighting':{
        'preset':'low_light','settings':EXPOSURE,'dependencies':dependencies,
        'note':'Blender-authored fixture geometry and UsdLux emitters; fixed Isaac exposure. Existing factory geometry and thermal truth retained in this first lighting stage.'}}
    stage.GetRootLayer().Save()
    report = {'stage':str(output),'lights':lights,'sky_path':str(sky.GetPath()),
              'surfaces':surfaces,'settings':EXPOSURE,'dependencies':dependencies,
              'factory_geometry':'Existing factory_inspection.usda; no fine partition change in this lighting stage',
              'blender_to_isaac_intensity_multiplier':INTENSITY_MULTIPLIER,
              'remaining':'Actual Isaac rendering and exposure verification'}
    (ASSETS/'weaklight_asset_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'stage':str(output),'exported_fixture_meshes':len(surfaces),
            'fixture_emitters_and_moon':len(lights),'total_lights_with_sky':len(lights)+1,
            'disabled_daylight':not stage.GetPrimAtPath('/Environment/defaultLight').IsActive(),
            'source_layer_hashes':len(dependencies)}


if __name__ == '__main__':
    result = build()
