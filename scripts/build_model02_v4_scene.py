"""Native Isaac composition: additive paving, installed rig and free-search task."""
import argparse,copy,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'factory/model02_thermal_v4';OLD=ROOT/'factory/model02_thermal_v3'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):path.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')


def camera_matrix(offset,pitch_deg):
    from pxr import Gf
    a=math.radians(pitch_deg);c,s=math.cos(a),math.sin(a)
    return Gf.Matrix4d(0.,-1.,0.,0.,s,0.,c,0.,-c,0.,s,0.,*offset,1.)


def install_cameras(stage,body_path,offset,pitch_deg):
    from pxr import Gf,UsdGeom
    body_world=UsdGeom.Xformable(stage.GetPrimAtPath(body_path)).ComputeLocalToWorldTransform(0)
    desired=camera_matrix(offset,pitch_deg)*body_world
    for path in [body_path+'/Camera_mount/Camera',body_path+'/Sensors/ThermalCamera']:
        camera=UsdGeom.Camera.Define(stage,path)
        parent=UsdGeom.Xformable(camera.GetPrim().GetParent()).ComputeLocalToWorldTransform(0)
        camera.MakeMatrixXform().Set(desired*parent.GetInverse())
        camera.CreateFocalLengthAttr(18.);camera.CreateHorizontalApertureAttr(20.955);camera.CreateVerticalApertureAttr(16.764)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.03,5000.))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--update',action='store_true');args=parser.parse_args()
    from scripts.model02_configuration_freshness import verify_inputs
    from scripts.model02_migration import validate_export_snapshot
    compiled=json.loads((OUT/'configuration/blender_parameters.json').read_text(encoding='utf-8'))
    verify_inputs(ROOT,compiled)
    preflight_export=json.loads((OUT/'export_manifest.json').read_text(encoding='utf-8'))
    validate_export_snapshot(preflight_export,OUT/'model02_thermal_v4_configured.blend',OUT/'model02_v4_paving.usdc')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'multi_gpu':False})
    try:
        import omni.usd
        from pxr import Gf,Usd,UsdGeom,UsdPhysics
        output=OUT/'model02_v4_scene.usda'
        if output.exists() and not args.update:raise ValueError('v4 scene exists; use --update for explicit own-release regeneration')
        exported=json.loads((OUT/'export_manifest.json').read_text(encoding='utf-8'))
        geometry=json.loads((OUT/'paving_geometry_report.json').read_text(encoding='utf-8'))
        original=json.loads((OLD/'isaac_manifest.json').read_text(encoding='utf-8'))
        source_rows=json.loads((OUT/'surfaces.json').read_text(encoding='utf-8'))['surfaces']
        source_ids={r['surface_id']:r for r in source_rows};replaced=set(geometry['superseded_surface_ids'])
        stage=Usd.Stage.Open(str(output)) if output.exists() else Usd.Stage.CreateNew(str(output))
        stage.GetRootLayer().subLayerPaths=['./model02_v4_paving.usdc','../model02_thermal_v3/model02_v3_scene.usda']
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
        stage.SetTimeCodesPerSecond(60);stage.SetStartTimeCode(0);stage.SetEndTimeCode(1000000)
        for row in original['surfaces']:
            if row['surface_id'] in replaced:stage.OverridePrim(row['usd_prim_path']).SetActive(False)
        for path in geometry.get('replaced_original_nodes',[]):stage.OverridePrim(path).SetActive(False)
        options=OUT/'mission_and_camera.json'
        if not options.exists():options.write_text(json.dumps({'units':'m','up_axis':'Z','start_position_m':[700.,-845.,8.5],
            'start_yaw_deg':90.,'camera_pitch_down_deg':15.,'camera_offset_body_m':None,
            'reference_waypoints_m':[[700,-845,8.5],[643,-850,12],[662,-832,12],[677,-816,12],[693.37,-797.25,3.5],
                [693.37,-797.25,12],[710,-807,12],[700,-840,12],[700,-845,8.5]],
            'require_path_length':False,'note':'Reference route for scene QA only; human freely searches. No autopilot.'},indent=2),encoding='utf-8')
        parameters=json.loads(options.read_text());assert parameters['units']=='m' and parameters['up_axis']=='Z'
        drone=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Crazyflie'))
        matrix=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),parameters['start_yaw_deg']))
        matrix.SetTranslateOnly(Gf.Vec3d(*parameters['start_position_m']));drone.MakeMatrixXform().Set(matrix)
        stage.GetRootLayer().Save()
        context=omni.usd.get_context();assert context.open_stage(str(output))
        for _ in range(100):app.update()
        stage=context.get_stage();stage.SetEditTarget(stage.GetRootLayer())
        root=stage.GetPrimAtPath('/World/Crazyflie');body=stage.GetPrimAtPath('/World/Crazyflie/body');assert body
        cache=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render]);body_inverse=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0).GetInverse()
        points=[];paths=[]
        for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Gprim) or UsdGeom.Imageable(p).ComputeVisibility()=='invisible':continue
            oriented=cache.ComputeWorldBound(p);box=oriented.GetRange()
            if box.IsEmpty():continue
            local_to_body=oriented.GetMatrix()*body_inverse
            points.extend(local_to_body.Transform(box.GetCorner(i)) for i in range(8));paths.append(str(p.GetPath()))
        assert points,'UAV payload did not load; cannot certify camera mount'
        points=np.array(points);lower,upper=points.min(0),points.max(0)
        offset=parameters['camera_offset_body_m']
        if offset is None:
            offset=[float(upper[0]+.05),0.,float(lower[2]-.03)]
            parameters['camera_offset_body_m']=offset;options.write_text(json.dumps(parameters,indent=2),encoding='utf-8')
        assert offset[0]>=upper[0]+.049 and offset[2]<=lower[2]-.029
        install_cameras(stage,'/World/Crazyflie/body',offset,parameters['camera_pitch_down_deg'])
        old_stage=Usd.Stage.Open(str(OLD/'model02_v3_scene.usda'));profile=copy.deepcopy(dict(old_stage.GetRootLayer().customLayerData['uavLighting']))
        road_source=UsdGeom.Mesh(old_stage.GetPrimAtPath('/World/LimeExp_02/node_1_29_gc051/mesh_1_29_gc051'))
        assert road_source
        road_collision=UsdGeom.Mesh.Define(stage,'/World/IndustrialPark02/Collision/RoadGC051')
        road_collision.CreatePointsAttr(road_source.GetPointsAttr().Get())
        road_collision.CreateFaceVertexCountsAttr(road_source.GetFaceVertexCountsAttr().Get())
        road_collision.CreateFaceVertexIndicesAttr(road_source.GetFaceVertexIndicesAttr().Get())
        road_collision.CreateSubdivisionSchemeAttr('none');road_collision.CreateVisibilityAttr('invisible')
        road_collision.MakeMatrixXform().Set(road_source.ComputeLocalToWorldTransform(0))
        UsdPhysics.CollisionAPI.Apply(road_collision.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(road_collision.GetPrim()).CreateApproximationAttr('none')
        dependencies={os.path.relpath(OLD/'model02_v3_scene.usda',OUT).replace('\\','/'):sha(OLD/'model02_v3_scene.usda'),
            './model02_v4_paving.usdc':sha(OUT/'model02_v4_paving.usdc')}
        for relative,digest in profile['dependencies'].items():dependencies[os.path.relpath((OLD/relative).resolve(),OUT).replace('\\','/')]=digest
        profile['dependencies']=dependencies
        stage.GetRootLayer().customLayerData={'uavLighting':profile,'v4Mission':{'parameter_file':'mission_and_camera.json','parameter_sha256':sha(options),
            'note':'Manual free search; semantic target clues only. Installed optical axes match; synchronization verification separate.'}}
        stage.GetRootLayer().Save()
        rows=[r for r in original['surfaces'] if r['surface_id'] not in replaced]+[dict(source_ids[r['surface_id']],**r) for r in exported['surfaces']]
        assert {r['surface_id'] for r in rows}==set(source_ids)
        for r in rows:assert stage.GetPrimAtPath(r['usd_prim_path']).IsActive()
        assert not any(stage.GetPrimAtPath(r['usd_prim_path']).IsActive() for r in original['surfaces'] if r['surface_id'] in replaced)
        report={'result':'PASS_V4_NATIVE_COMPOSITION','surfaces':rows,'surface_count':len(rows),'source_scene_sha256':sha(OLD/'model02_v3_scene.usda'),
            'scene_sha256':sha(output),'inactive_superseded_surface_ids':sorted(replaced),'uav_body_bounds_m':[lower.tolist(),upper.tolist()],
            'loaded_uav_gprim_paths':paths,'camera_offset_body_m':offset,'camera_pitch_down_deg':parameters['camera_pitch_down_deg'],
            'static_collision':'v3 consolidated collider retained; original GC051 road copied into independent invisible RoadGC051 collider before disabling original rendered node. All elevations unchanged.',
            'start_position_m':parameters['start_position_m'],'original_tasks':original['tasks'],'manual_success':None}
        (OUT/'isaac_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        from scripts.model02_migration import bind_surface_rows
        authoring=yaml.safe_load((OUT/'configuration/thermal_authoring.yaml').read_text(encoding='utf-8'))
        from scripts.model02_v4_configuration import compile_night_environment
        template=yaml.safe_load((ROOT/'configs/model02_v3_night_environment.yaml').read_text(encoding='utf-8'))
        environment,weather_text=compile_night_environment(template,authoring['condition'],OUT/'night_weather.csv')
        (OUT/'night_weather.csv').write_text(weather_text,encoding='utf-8')
        dump(ROOT/'configs/model02_v4_night_environment.yaml',environment)
        thermal=yaml.safe_load((ROOT/'configs/model02_v3_thermal.yaml').read_text(encoding='utf-8'))
        oldtargets={r['temperature_target_id'] for r in json.loads((OLD/'surfaces.json').read_text(encoding='utf-8'))['surfaces']}
        thermal['bindings']=[b for b in thermal['bindings'] if b['name'] not in oldtargets and not any(
            b['prim_path']==p or b['prim_path'].startswith(p+'/') for p in geometry.get('replaced_original_nodes',[]))]
        thermal['targets']={k:v for k,v in thermal['targets'].items() if k not in oldtargets}
        thermal['targets'].update(authoring['targets']);thermal['bindings'].extend(bind_surface_rows(rows,authoring))
        dump(ROOT/'configs/model02_v4_thermal.yaml',thermal)
        task=yaml.safe_load((ROOT/'configs/model02_v3_tank_task.yaml').read_text(encoding='utf-8'))
        task.update(task_id='model02_v4_tank_free_search',start_position=parameters['start_position_m'],
            instruction='从厂房附近起飞，观察厂房外侧道路与蓝色护栏储罐区，自行选择路线接近储罐；用红外比较罐壁，找到异常后安全悬停并按T手动确认。无需沿固定航线或凑够里程。')
        evidence=task['environment_evidence']['factory_inspection'];route=parameters['reference_waypoints_m']
        evidence.update(reference_waypoints=route,reference_waypoints_m=route,reference_path_length_m=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum()),
            reference_waypoint_names=['屋顶以上的起点','道路外围','道路进入点','储罐入口','观察位置','垂直爬升撤离','外围转弯','返回区','起点'],
            geometry_source='v3 actual tank geometry unchanged; v4 replaces only coplanar pavement surfaces')
        evidence['inspection_requirements']['require_path_length']=False
        from scripts.model02_migration import core_temperature_truth
        evidence['thermal_truth'].update(core_temperature_truth(next(t for t in original['tasks'] if t['kind']=='tank'),authoring))
        evidence['thermal_truth']['source_mapping']='factory/model02_thermal_v4/configuration/thermal_authoring.yaml'
        task['environment_evidence']['lighting']['manifest']='factory/model02_thermal_v4/isaac_manifest.json'
        task['environment_evidence']['condition_assumptions']['authoring_condition']=copy.deepcopy(authoring['condition'])
        task['environment_description']+=' 厂房与设备区接近位置新增设计铺装砖及填缝；其他地面为混凝土板与路缘。'
        dump(ROOT/'configs/model02_v4_tank_task.yaml',task)
        display=yaml.safe_load((ROOT/'configs/model02_v3_display.yaml').read_text(encoding='utf-8'));display['name']='model02_v4_fault_0_85C'
        dump(ROOT/'configs/model02_v4_display.yaml',display)
        detail=copy.deepcopy(display);detail.update(name='model02_v4_environment_18_28C',temperature_min_c=18.,temperature_max_c=28.)
        dump(ROOT/'configs/model02_v4_detail_display.yaml',detail)
        scene=yaml.safe_load((ROOT/'configs/model02_v3_tank_scene.yaml').read_text(encoding='utf-8'))
        scene['scene']['stage']='../factory/model02_thermal_v4/model02_v4_scene.usda';scene['task']['file']='model02_v4_tank_task.yaml'
        scene['scene']['thermal_scene']='model02_v4_night_environment.yaml'
        scene['sensors'].update(thermal_config='model02_v4_thermal.yaml',display_config='model02_v4_display.yaml',material_directory='../factory/model02_thermal_v4/configuration/materials')
        dump(ROOT/'configs/model02_v4_tank_scene.yaml',scene)
        scene['sensors']['display_config']='model02_v4_detail_display.yaml';dump(ROOT/'configs/model02_v4_detail_scene.yaml',scene)
        from scripts.scene_config import load_scene_config
        from scripts.thermal_config import load_thermal_config
        load_scene_config(ROOT/'configs/model02_v4_tank_scene.yaml');t=load_thermal_config(ROOT/'configs/model02_v4_thermal.yaml')
        summary={k:v for k,v in report.items() if k not in ['surfaces','original_tasks','loaded_uav_gprim_paths','inactive_superseded_surface_ids']}
        summary.update(total_bindings=len(t.bindings),thermal_targets=len(t.targets))
        (OUT/'binding_configuration_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('[v4] '+json.dumps(summary),flush=True)
    finally:app.close()


if __name__=='__main__':main()
