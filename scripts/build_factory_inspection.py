"""Build only the factory inspection overlay and its dedicated YAML files.

Run with Isaac Python plus the pxr module path (or from the Isaac environment).
The repaired model, robot source and all shared configs are read-only inputs.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import yaml
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    model_path = ROOT/'factory/import_fixed/factory_scene_fixed.usd'
    output = ROOT/'factory/factory_inspection.usda'
    if output.exists():
        raise FileExistsError(f'Overlay already exists: {output}')
    source = Usd.Stage.Open(str(model_path))
    equipment = '/World/LimeExp_02/node_1_29_gc075'
    original_path = equipment+'/mesh_1_29_gc075'
    mesh = UsdGeom.Mesh(source.GetPrimAtPath(original_path))
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise RuntimeError('Selected source must remain triangulated')
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    points = np.asarray(mesh.GetPointsAttr().Get())
    normals = np.asarray(mesh.GetNormalsAttr().Get())
    if mesh.GetNormalsInterpolation() != 'faceVarying':
        raise RuntimeError('Expected face-varying source normals')
    uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
    uvs = np.asarray(uv.ComputeFlattened()).reshape(-1, 3, 2)
    selected = np.array([536, 537, 538, 539])
    material_path = '/World/LimeExp_02/Looks/material_1_29_gc262_3'
    source_subset = source.GetPrimAtPath(original_path+'/subset_1_29_gc262_3')
    if not set(selected).issubset(set(UsdGeom.Subset(source_subset).GetIndicesAttr().Get())):
        raise RuntimeError('Hotspot selection no longer belongs to measured tank shell')
    remaining = np.setdiff1d(np.arange(len(counts)), selected)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths.append('./import_fixed/factory_scene_fixed.usd')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1000000)
    # Saved render products are UI leftovers, unrelated to the actual model.
    for path in ['/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0', '/OmniverseKit_Persp', '/OmniverseKit_Top',
                 '/OmniverseKit_Front', '/OmniverseKit_Right']:
        if stage.GetPrimAtPath(path):
            stage.OverridePrim(path).SetActive(False)
    stage.OverridePrim(original_path).SetActive(False)
    parent = UsdGeom.Xform.Define(stage, equipment+'/InspectionSurface')
    parent.MakeMatrixXform().Set(UsdGeom.Xformable(mesh).GetLocalTransformation())
    old_to_new = {int(old): new for new, old in enumerate(remaining)}
    for name, faces in [('Normal', remaining), ('Hotspot', selected)]:
        dest = UsdGeom.Mesh.Define(stage, str(parent.GetPath())+'/'+name)
        dest.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
        dest.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(faces)))
        dest.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices[faces].reshape(-1).astype('int32')))
        dest.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.reshape(-1,3,3)[faces].reshape(-1,3).astype('float32')))
        dest.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        dest.CreateSubdivisionSchemeAttr('none')
        dest.CreateDoubleSidedAttr(mesh.GetDoubleSidedAttr().Get())
        new_uv = UsdGeom.PrimvarsAPI(dest).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                       UsdGeom.Tokens.faceVarying)
        new_uv.Set(Vt.Vec2fArray.FromNumpy(uvs[faces].reshape(-1,2).astype('float32')))
        if name == 'Hotspot':
            UsdShade.MaterialBindingAPI.Apply(dest.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)))
        else:
            UsdGeom.Subset.SetFamilyType(dest, 'materialBind', UsdGeom.Tokens.partition)
            for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
                new_faces = [old_to_new[int(i)] for i in subset.GetIndicesAttr().Get() if int(i) in old_to_new]
                sub = UsdGeom.Subset.CreateGeomSubset(dest, subset.GetPrim().GetName(),
                    UsdGeom.Tokens.face, new_faces, 'materialBind')
                material = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
                UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(material.GetPath())))
    # Reuse only the existing robot subtree, never the rest of the lesson.
    drone = stage.DefinePrim('/World/Crazyflie', 'Xform')
    drone.GetReferences().AddReference('../lesson_crazyflie.usd', '/World/Crazyflie')
    waypoints = [[640.,-855.,15.],[662.,-832.,15.],[677.,-816.,15.],
                 [693.37,-797.25,3.5],[710.,-807.,15.],[700.,-840.,15.],[640.,-855.,15.]]
    observation = waypoints[3]
    start_yaw = math.degrees(math.atan2(23,22))
    xform = UsdGeom.Xformable(drone)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*waypoints[0]))
    xform.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),start_yaw).GetQuat()))
    xform.AddScaleOp().Set(Gf.Vec3f(1))
    # Forward-facing RGB on the existing mount; slight downward look.
    camera_path = '/World/Crazyflie/body/Camera_mount/Camera'
    camera = UsdGeom.Camera(stage.OverridePrim(camera_path))
    camera.GetPrim().SetTypeName('Camera')
    camera.MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.05,0,.02),
        Gf.Vec3d(1.05,0,-.05), Gf.Vec3d(0,0,1)).GetInverse())
    camera.CreateFocalLengthAttr(18.)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(16.764)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05,5000.))
    physics = UsdPhysics.Scene.Define(stage, '/PhysicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    physics.CreateGravityMagnitudeAttr(9.81)
    stage.GetRootLayer().Save()
    # Measurement is from the original source, before any face is removed.
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    world = (np.c_[points,np.ones(len(points))]@transform)[:,:3]
    triangles = world[indices]
    areas = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    hot_position = np.average(triangles[selected].mean(1),axis=0,weights=areas[selected]).tolist()
    geometry_path = ROOT/'factory/factory_inspection_geometry.npz'
    np.savez_compressed(geometry_path, equipment_vertices=world,
                        equipment_triangles=indices,
                        hotspot_vertices=world, hotspot_triangles=indices[selected])
    geometry_sha = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    length = sum(math.dist(a,b) for a,b in zip(waypoints,waypoints[1:]))
    data = {
        'equipment_prim_path': equipment,
        'hotspot_prim_path': equipment+'/InspectionSurface/Hotspot',
        'normal_surface_prim_path': equipment+'/InspectionSurface/Normal',
        'comparison_equipment_prim_path': '/World/LimeExp_02/node_1_29_gc076/mesh_1_29_gc076',
        'rgb_material_path': material_path,
        'hotspot_position_m': hot_position,
        'ground_world_z_m': 0.009999999776482582,
        'route_ground_height_range_m': [-0.10000000149011612, 6.36610221862793],
        'route_ground_height_semantics': 'highest model surface below each route sample; includes roofs and pipe structures, not only terrain',
        'observation_position': observation,
        'observation_yaw_rad': math.atan2(hot_position[1]-observation[1],hot_position[0]-observation[0]),
        'reference_waypoints': waypoints,
        'reference_waypoints_m': waypoints,
        'reference_waypoint_names': ['外围悬停起点','沿道路进入','储罐区入口','目标观察点','撤离转弯点','返回安全区','起点上空'],
        'reference_path_length_m': length,
        'geometry_file': 'factory/factory_inspection_geometry.npz',
        'geometry_sha256': geometry_sha,
        'body_envelope_radius_m': .2,
        'inspection_requirements': {
            'minimum_equipment_distance_m':5.,'observation_distance_range_m':[6.,8.],
            'hold_duration_s':6.,'maximum_observation_speed_m_s':.3,'maximum_sample_gap_s':.25,
            'body_envelope_radius_m':.2,'reference_length_range_m':[180.,220.],
            'actual_path_length_range_m':[180.,220.],
            'minimum_height_world_z_m':2., 'minimum_agl_m':2.5,
        },
        'thermal_truth': {'hotspot_temperature_c':78.,'normal_equipment_temperature_c':30.,
                          'source_face_indices':selected.tolist(),'hotspot_area_m2':float(areas[selected].sum())},
        'flight_settings': {'horizontal_speed_m_s':1.5,'speed_scale':1.,
                            'vertical_speed_m_s':.5,'yaw_speed_degrees_s':45.},
        'evidence_images': ['validation/factory-inspection/selection-20261016/rgb/equipment_row.png'],
        'distance_semantics':'equipment mesh surface clearance minus conservative 0.2 m body sphere; camera-to-hotspot distance separately reported',
    }
    task = {'task_id':'factory_tank_hotspot_01',
        'instruction':'请沿园区道路前往储罐区，检查边缘一排带蓝色护栏的圆柱储罐中最外侧的一座。先用可见光确认储罐外观及周围管廊、厂房，再用热红外查找罐壁上的异常热点。到达目标附近后，保持约 6–8 米安全观察距离悬停 6 秒，确认故障位置后按 T 标记完成，随后返回外围安全区域。',
        'target_prim_path':data['hotspot_prim_path'],
        'target_description':'蓝色护栏圆柱储罐的罐壁局部表面；模型未提供可读设备铭牌。',
        'environment_description':'储罐区位于管廊和低层厂房外侧，周围为铺装场地。',
        'environment_landmarks':['成排圆柱储罐','蓝色护栏','架空管廊','低层厂房'],
        'environment_evidence':{'factory_inspection':data},'start_position':waypoints[0],
        'target_position':hot_position,'target_position_semantics':'selected source shell triangles area-weighted world center, meters',
        'notes':'人工键盘驾驶；T 只记录人工确认，客观安全距离、悬停和视觉证据由独立评估器检查。'}
    config = {'schema_version':1,'task':{'file':'factory_inspection_task.yaml'},
        'scene':{'stage':'../factory/factory_inspection.usda',
            'thermal_scene':'../third_party/isaac-thermal-camera-simulation/configs/scenes/quad_flight_clear_noon.yaml',
            'static_targets':['/World/LimeExp_02']},
        'drone':{'prim_path':'/World/Crazyflie','body_path':'/World/Crazyflie/body',
            'camera_path':camera_path,'minimum_height':2.},
        'sensors':{'width':160,'height':128,
            'sensor_config':'../third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml',
            'thermal_config':'factory_inspection_thermal.yaml','preview_fast':True,
            'align_thermal_to_rgb':True,'flat_field_enabled':True,'sensor_seed':0,
            'display_config':'factory_inspection_display.yaml'}}
    targets = {name:{'mode':'fixed','temperature_c':temperature} for name,temperature in
               [('factory_building',24.),('factory_ground',26.),('factory_equipment',30.),
                ('factory_vegetation',25.),('factory_drone',35.),('factory_hotspot',78.)]}
    bindings = [dict(name='drone',prim_path='/World/Crazyflie',include_children=True,target='factory_drone',material='painted_composite')]
    # Coarse scene classes are explicit approximations; no material name is
    # treated as proof of an equipment purpose. Small unknowns keep the default.
    for p in source.Traverse():
        if not p.IsA(UsdGeom.Mesh): continue
        path=str(p.GetPath()); bbox=UsdGeom.BBoxCache(0,['default','render','proxy']).ComputeWorldBound(p).ComputeAlignedRange()
        if bbox.IsEmpty():continue
        size=bbox.GetSize(); maxz=bbox.GetMax()[2]
        if maxz<=.45:
            bindings.append(dict(name='ground_'+p.GetParent().GetName(),prim_path=path,include_children=False,target='factory_ground',material='concrete'))
    for num in range(75,86):
        bindings.append(dict(name=f'tank_{num}',prim_path=f'/World/LimeExp_02/node_1_29_gc{num:03}',include_children=True,target='factory_equipment',material='aircraft_aluminium_painted'))
    bindings.append(dict(name='local_shell_hotspot',prim_path=data['hotspot_prim_path'],include_children=False,target='factory_hotspot',material='aircraft_aluminium_painted'))
    thermal={'schema_version':1,'defaults':{'target':'factory_building','material':'concrete'},'targets':targets,'bindings':bindings}
    display={'schema_version':1,'name':'factory_fixed_apparent_temperature','mode':'fixed_apparent_temperature',
             'temperature_min_c':-30.,'temperature_max_c':100.,'dde_gain':0.,'polarity':'white_hot','palette':'gray'}
    for filename,obj in [('factory_inspection_scene.yaml',config),('factory_inspection_task.yaml',task),
                         ('factory_inspection_thermal.yaml',thermal),('factory_inspection_display.yaml',display)]:
        path=ROOT/'configs'/filename
        if path.exists():raise FileExistsError(path)
        path.write_text(yaml.safe_dump(obj,allow_unicode=True,sort_keys=False),encoding='utf8')
    selection={'equipment_prim_path':equipment,'hotspot_prim_path':data['hotspot_prim_path'],
        'rgb_material_path':material_path,'hotspot_world_position_m':hot_position,
        'ground_world_z_m':data['ground_world_z_m'],'start_position_m':waypoints[0],
        'observation_position_m':observation,'reference_waypoints_m':waypoints,
        'reference_path_length_m':length,'evidence_images':data['evidence_images'],
        'source_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'split':{'source_faces':len(counts),'normal_faces':len(remaining),'hotspot_faces':len(selected),
                 'same_points_uv_normals_materials':True,'original_mesh_disabled_in_overlay':True}}
    (ROOT/'validation/factory-inspection/selection-20261016/target_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
    print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
