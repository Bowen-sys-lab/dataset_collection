"""Controlled native AOV probe; pure comparison is available without Kit."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def compare_controlled_pair(control,variant):
    experiment=control['experiment'];changed=[]
    keys=['camera','geometry','environment','clock','sensor_state']
    if experiment!='ir_material':keys.append('ir_material')
    for key in keys:
        if control[key]!=variant[key]:changed.append(key)
    arrays=['distance','temperature']+([] if experiment=='ir_material' else ['material_id'])
    for key in arrays:
        if not np.array_equal(control[key],variant[key]):changed.append(key)
    if changed:return {'result':'inconclusive','changed_controls':changed}
    report={'result':'controlled','experiment':experiment}
    for key in ['normal','distance','temperature','radiance','rgb']:
        a,b=np.asarray(control[key],dtype=float),np.asarray(variant[key],dtype=float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return {'result':'inconclusive','invalid_array':key}
        report[key+'_max_abs_difference']=float(np.max(np.abs(a-b)))
    report['normal_effect']='observed' if report['normal_max_abs_difference']>1e-6 else 'not_observed'
    return report


def final_probe_result(report,require_distance=False):
    if 'error' in report or 'traceback' in report:return 'FAIL'
    cases=['repeat','normal_off','normal_restored','color_only']
    if any(report.get('experiments',{}).get(case,{}).get('result')!='controlled' for case in cases):
        return 'INCONCLUSIVE_CONTROLLED_PROBE'
    if require_distance:
        return 'PASS_NATIVE_SURFACE_AND_DISTANCE_PROBE' if report.get('distance_probe',{}).get('result')=='PASS_NATIVE_DISTANCE_PROBE' else 'INCONCLUSIVE_DISTANCE_PROBE'
    return 'PASS_CONTROLLED_PROBE'


def main():
    import argparse,pickle,traceback
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--fast',action='store_true');parser.add_argument('--distance-probe',action='store_true');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);report={'result':'FAIL','manual_success':None,'experiments':{}}
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':'PathTracing','multi_gpu':False,'samples_per_pixel_per_frame':16})
    preview=None
    try:
        import omni.timeline
        from pxr import Gf,UsdGeom,UsdShade,UsdPhysics
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.build_model02_v4_scene import install_cameras
        from irsim_isaac.pipeline.gbuffer_isaac import geometry_planes
        config=load_scene_config(ROOT/'configs/model02_v5_tank_scene.yaml')
        preview,_=open_preview(app,scene_config=config,width=160,height=128,thermal_width=160,thermal_height=128,fast=args.fast,start_controller=False,show_ui=False)
        stage=preview.stage;stage.SetEditTarget(stage.GetSessionLayer())
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.IsA(UsdPhysics.Joint):UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        matrix=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),90.));matrix.SetTranslateOnly(Gf.Vec3d(700.,-845.,4.))
        UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.prim_path)).MakeMatrixXform().Set(matrix)
        mission=json.loads((ROOT/'factory/model02_thermal_v4/mission_and_camera.json').read_text())
        install_cameras(stage,config.drone.body_path,mission['camera_offset_body_m'],55.)
        shader=next(UsdShade.Shader(p) for p in stage.Traverse() if p.IsA(UsdShade.Shader) and 'V5_grass' in str(p.GetPath()) and UsdShade.Shader(p).GetIdAttr().Get()=='UsdPreviewSurface')
        normal=shader.GetInput('normal');color=shader.GetInput('diffuseColor')
        normal_connections=normal.GetAttr().GetConnections();color_connections=color.GetAttr().GetConnections()
        assert normal_connections and color_connections
        base_state=copy.deepcopy(preview.thermal.state)
        state_hash=hashlib.sha256(pickle.dumps(base_state,protocol=5)).hexdigest()
        source_hash=hashlib.sha256((ROOT/'factory/model02_thermal_v5/model02_v5_ground.usdc').read_bytes()).hexdigest()
        env_hash=hashlib.sha256(config.scene.thermal_scene.read_bytes()).hexdigest()
        captures={}
        for index,case in enumerate(['base','repeat','normal_off','normal_restored','color_only']):
            normal.GetAttr().SetConnections(normal_connections);color.GetAttr().SetConnections(color_connections)
            if case=='normal_off':normal.DisconnectSource();normal.Set(Gf.Vec3f(0.,0.,1.))
            if case=='color_only':color.DisconnectSource();color.Set(Gf.Vec3f(.7,.07,.03))
            preview.thermal.state=copy.deepcopy(base_state)
            assert hashlib.sha256(pickle.dumps(preview.thermal.state,protocol=5)).hexdigest()==state_hash
            timeline.set_current_time(0.);timeline.commit()
            for _ in range(8):app.update()
            rgb,thermal,meta=preview.capture(render=True,simulation_time=0.,frozen_state=True)
            raw=preview.thermal._reader.read();last=preview.thermal._last
            assert meta['thermal_coverage']['complete_for_this_view'] and meta['pair_pose_max_error']<1e-6
            snapshot={'experiment':'normal_map','camera':json.dumps(meta['camera_world_matrix']),
                'geometry':source_hash,'ir_material':hashlib.sha256(config.sensors.thermal_config.read_bytes()).hexdigest(),
                'environment':env_hash,'clock':0.,'sensor_state':state_hash,
                'normal':np.asarray(raw.normal).copy(),'distance':np.asarray(raw.distance_m).copy(),
                'temperature':np.asarray(last.planes['temperature_k']).copy(),
                'material_id':np.asarray(last.material_id).copy(),'radiance':np.asarray(thermal.radiance).copy(),
                'rgb':np.asarray(rgb).copy()}
            captures[case]=snapshot
            for key,value in snapshot.items():
                if isinstance(value,np.ndarray):np.save(args.out/f'{case}_{key}.npy',value)
            for key in ['normal_dot_view','sky_view_factor']:
                np.save(args.out/f'{case}_{key}.npy',last.planes[key])
            geometry=geometry_planes(raw,up_axis=preview.thermal._up_axis or 'Y',
                position_frame=preview.thermal.position_frame,camera_position=preview.thermal._camera_position,
                camera_to_world=preview.thermal._camera_to_world)
            np.save(args.out/f'{case}_normal_dot_up.npy',geometry.normal_dot_up)
            preview.save_frame(args.out,index,rgb,thermal,{**meta,'case':case,'manual_success':None,'data_usage':'controlled static channel probe'})
            print('[IR surface probe]',case,'captured',flush=True)
        for case in ['repeat','normal_off','normal_restored','color_only']:
            a,b=copy.copy(captures['base']),copy.copy(captures[case])
            if case=='color_only':a['experiment']=b['experiment']='rgb_color'
            report['experiments'][case]=compare_controlled_pair(a,b)
        report.update(resolved_aovs=preview.thermal._reader.resolved,normal_shader=str(shader.GetPath()),
            normal_connection=[str(p) for p in normal_connections],resolution=[160,128],
            source_geometry_sha256=source_hash,source_material_config_sha256=captures['base']['ir_material'])
        import carb.settings
        settings=carb.settings.get_settings()
        report['renderer']=preview.metadata['renderer']
        report['preview_fast']=args.fast
        report['fast_scope']='PreviewSky approximation; fast does not select a different renderer'
        report['renderer_settings']={p:settings.get(p) for p in ['/renderer/active','/rtx/rendermode','/rtx/pathtracing/spp','/rtx/pathtracing/totalSpp']}
        report['intrinsics_px']={k:preview.metadata[k] for k in ['rgb_intrinsics_px','thermal_intrinsics_px']}
        report['channel_result']='PASS_CONTROLLED_PROBE' if all(r['result']=='controlled' for r in report['experiments'].values()) else 'INCONCLUSIVE_CONTROLLED_PROBE'
        report['result']=report['channel_result']
        if args.distance_probe:
            normal.GetAttr().SetConnections(normal_connections);color.GetAttr().SetConnections(color_connections)
            from scripts.validate_model02_ir_distance_probe import capture_distance_views
            report['distance_probe']=capture_distance_views(app,preview,config,args.out/'distance',base_state)
            if report['channel_result']=='PASS_CONTROLLED_PROBE' and report['distance_probe']['result']=='PASS_NATIVE_DISTANCE_PROBE':
                report['result']='PASS_NATIVE_SURFACE_AND_DISTANCE_PROBE'
            else:report['result']='INCONCLUSIVE_DISTANCE_PROBE'
    except BaseException as exc:report.update(result='FAIL',error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        report['result']=final_probe_result(report,require_distance=args.distance_probe)
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if preview:preview.close()
        app.close()
    return 0 if report['result'] in ['PASS_CONTROLLED_PROBE','PASS_NATIVE_SURFACE_AND_DISTANCE_PROBE'] else 1


if __name__=='__main__':sys.exit(main())
