"""Native mounted-camera, full-site multiscale evidence; never human flight."""
import argparse,copy,hashlib,json,math,sys,time,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'


def diagnostic_eye(center,normal,body,distance,scale,ground=False,max_pitch_deg=35.):
    center=np.asarray(center,dtype=float);direction=np.asarray(normal,dtype=float).copy()
    if ground:direction[2]=abs(direction[2])
    elif np.dot(direction,center-np.asarray(body))<0:direction=-direction
    if np.linalg.norm(direction[:2])<1e-8:
        direction[:2]=(center-np.asarray(body))[:2]
        if np.linalg.norm(direction[:2])<1e-8:direction[0]=1.
    horizontal=np.linalg.norm(direction[:2]);direction[2]=min(max(.45 if scale=='overall' else .2,direction[2]),horizontal*math.tan(math.radians(max_pitch_deg)));direction/=np.linalg.norm(direction)
    distance=max(distance,2.5 if ground else distance,(2.-center[2])/direction[2])
    eye=center+direction*distance;eye[2]=max(2.,eye[2])
    return eye


def projected_pixels(size_m,distance_m,width,focal_length_mm,horizontal_aperture_mm):
    return size_m/distance_m*width*focal_length_mm/horizontal_aperture_mm


def native_roi(mask,shape):
    h,w=shape;mh,mw=mask.shape
    if mh%h or mw%w:raise ValueError('AOV and sensor grids are not aligned integer scales')
    return np.asarray(mask,dtype=bool).reshape(h,mh//h,w,mw//w).any(axis=(1,3))


def control_smoke(app,stage,config,root_matrix,physics_flags):
    import omni.timeline
    from pxr import UsdGeom
    from scripts.drone_keyboard_controller import DroneKeyboardController
    from scripts.drone_physics_runtime import DronePhysicsRuntime
    from scripts.start_crazyflie_dual_camera_view import preview_motion_config
    for attr,value in physics_flags:
        if value is None:attr.Clear()
        else:attr.Set(value)
    UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.prim_path)).MakeMatrixXform().Set(root_matrix)
    timeline=omni.timeline.get_timeline_interface();timeline.set_auto_update(True)
    runtime=DronePhysicsRuntime(config.drone.prim_path);timeline.play()
    for _ in range(60):app.update()
    assert runtime.ready,'Physics tensor articulation did not become ready'
    controller=DroneKeyboardController(config.drone.prim_path,config=preview_motion_config(horizontal_speed=10.),minimum_height=config.drone.minimum_height)
    controller._runtime=runtime;controller._initialize_hover();before=runtime.snapshot();controller.keys.press('W');trace=[]
    for i in range(60):
        controller._advance(1./60.);controller._write_pose();app.update()
        if i%10==0:trace.append(runtime.snapshot())
    after=runtime.snapshot();controller.keys.emergency_stop();runtime.prepare_hover()
    distance=float(np.linalg.norm(np.asarray(after['position'])[:2]-np.asarray(before['position'])[:2]))
    assert .5<distance<=10.01 and not controller.task_success_marked,(distance,after)
    timeline.stop();timeline.set_auto_update(False)
    return {'result':'PASS_SCRIPTED_PHYSX_CONTROL_SMOKE','horizontal_speed_limit_m_s':10.,'command_duration_s':1.,'measured_horizontal_displacement_m':distance,
        'before':before,'after':after,'trace':trace,'manual_success':None,'scope':'Scripted existing motion integration and PhysX tensor pose writes; no human keyboard flight or T confirmation'}


def validate_view_evidence(views,registry):
    required=defaultdict(set)
    for row in registry['sources']:required[row['zone_id']].add(row['role'])
    for v in views:
        if not v.get('native_capture') or v.get('manual_success') is not None:raise ValueError('Native diagnostics cannot claim human confirmation')
        if not v.get('intrinsics') or not v.get('distance_m',0.)>0.:raise ValueError('Missing measured camera information')
    for zone,roles in required.items():
        for resolution in [[160,128],[640,512]]:
            selected=[v for v in views if v['zone_id']==zone and v['resolution']==resolution]
            if not {'overall','mid','near'}<={v['scale'] for v in selected}:raise ValueError('Missing multiscale views: '+zone)
            for role in roles:
                if not sum(v['role_pixels'].get(role,0) for v in selected)>0:raise ValueError('No actual '+role+' pixels: '+zone)
    return {'complete':True,'zones':len(required),'views':len(views),'human_flight_validated':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);parser.add_argument('--resolution',type=int,choices=[160,640]);parser.add_argument('--aggregate',action='store_true');args=parser.parse_args()
    if args.aggregate:
        summaries=[json.loads((args.out/name/'summary.json').read_text()) for name in ['live','diagnostic']]
        assert all(s['result']=='PASS_V6_NATIVE_RESOLUTION' for s in summaries)
        assert summaries[0]['scene_sha256']==summaries[1]['scene_sha256'] and summaries[0]['thermal_config_sha256']==summaries[1]['thermal_config_sha256']
        views=[v for s in summaries for v in s['views']];registry=json.loads((OUT/'source_registry.json').read_text())
        report={'result':'PASS_V6_NATIVE_MULTISCALE_VIEWS','coverage':validate_view_evidence(views,registry),'views':views,'resolutions':summaries,'manual_success':None}
        (args.out/'summary.json').write_text(json.dumps(report,indent=2));return 0
    args.out.mkdir(parents=True,exist_ok=False)
    report={'result':'FAIL','manual_success':None,'data_usage':'Native static mounted-camera diagnostics; no human flight or T confirmation','views':[],'responses':[]}
    app=preview=annotator=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'renderer':'PathTracing','samples_per_pixel_per_frame':16,'multi_gpu':False})
        import omni.timeline,omni.replicator.core as rep
        from pxr import Gf,UsdGeom,UsdPhysics
        from PIL import Image
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.build_model02_v4_scene import install_cameras
        from scripts.thermal_display import fixed_temperature_display
        from irsim_isaac.pipeline.material_ids import labels_from_payload
        from irsim.pipeline.frame import run_frame
        from irsim.thermal.solvers import PrescribedSolver
        from irsim_isaac.pipeline.aerial_bridge import AerialThermalBridge
        from scripts.model02_response import check_local_temperature_response
        config=load_scene_config(ROOT/'configs/model02_v6_tank_scene.yaml');manifest=json.loads((OUT/'isaac_manifest.json').read_text(encoding='utf-8'))
        report['scene_sha256']=hashlib.sha256(config.scene.stage.read_bytes()).hexdigest();report['thermal_config_sha256']=hashlib.sha256(config.sensors.thermal_config.read_bytes()).hexdigest()
        registry=json.loads((OUT/'source_registry.json').read_text());sources={r['source_object']:r for r in registry['sources']}
        rows={r['usd_prim_path']:r for r in manifest['surfaces']};mission=json.loads((ROOT/'factory/model02_thermal_v4/mission_and_camera.json').read_text())
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        targets=defaultdict(list)
        for path,row in rows.items():
            source=sources.get(row.get('source_object'))
            if source and row.get('area_m2',0.)>.01 and row.get('source_object')!='11_29_gc1750':targets[(source['zone_id'],source['role'])].append(row)
        # Old preserved sources remain represented by their actual component rows.
        for source in registry['sources']:
            key=(source['zone_id'],source['role'])
            if not targets[key]:
                targets[key]=[r for r in rows.values() if r.get('source_object')==source['source_object']]
        views=[]
        for zone in sorted({s['zone_id'] for s in registry['sources']}):
            choices=[]
            for (z,role),items in sorted(targets.items()):
                if z!=zone or not items:continue
                preferred=[r for r in items if r['component'] in ['Wall','WindowGlass','EquipmentShell','PipeWall','StructuralMetal','ConcreteRoadSlab','Landscape']]
                choice=max(preferred or items,key=lambda r:r.get('area_m2',1.))
                choices.append((role,choice));views.append((zone,'mid',role,choice,12.))
            assert choices,('No registered view target',zone)
            role,choice=next(((role,r) for role,r in choices if role=='building'),choices[0])
            views.extend([(zone,'overall',role,choice,70.),(zone,'near',role,choice,3.)])
        # Explicit near views complement per-zone samples for physical interfaces.
        for component in ['WindowFrame','EquipmentShell','PipeWall','ConcreteRoadSlab','PavingBrick','TankShell']:
            choices=[r for r in rows.values() if r['component']==component and r.get('source_object') in sources]
            if not choices:continue
            choice=max(choices,key=lambda r:r.get('area_m2',1.));source=sources[choice['source_object']]
            views.append((source['zone_id'],'interface',source['role'],choice,1.2 if component in ['WindowFrame','PavingBrick'] else 3.))
        tasks=json.loads((ROOT/'factory/model02_thermal_v4/isaac_manifest.json').read_text(encoding='utf-8'))['original_tasks']
        for task in tasks:
            choice=dict(rows[task['hotspot_prim_paths'][0]]);choice['_observation_eye_m']=task['observation_position_m'];choice['_fault_kind']=task['kind'];choice['center_m']=task['target_center_m'];source=sources[choice['source_object']]
            views.append((source['zone_id'],'fault',source['role'],choice,3.))
        responded=set();index=0
        sizes=[(160,128),(640,512)] if args.resolution is None else [(args.resolution,128 if args.resolution==160 else 512)]
        for width,height in sizes:
            started=time.perf_counter()
            preview,_=open_preview(app,scene_config=config,width=width,height=height,thermal_width=width,thermal_height=height,fast=width==160,start_controller=False,show_ui=False)
            stage=preview.stage;stage.SetEditTarget(stage.GetSessionLayer());audit=preview.thermal.thermal_config_report
            assert not audit['fallback_paths'] and not audit['unmatched_bindings'],('Unbound visible surfaces',len(audit['fallback_paths']),len(audit['unmatched_bindings']))
            report.setdefault('startup_seconds',{})[str(width)]=time.perf_counter()-started
            report.setdefault('renderer_profiles',{})[str(width)]=preview.metadata['renderer']
            report['bindings']={'active_paths':len(audit['assignments']),'fallbacks':0,'unmatched':0}
            report['resolved_aovs']={k:str(v) for k,v in preview.thermal._reader.resolved.items()}
            if width==160:(args.out/'binding_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
            assert UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
            assert preview.metadata['authored_lighting']['configuration_verified']
            condition=json.loads((OUT/'configuration/blender_parameters.json').read_text())['condition'];weather=preview.thermal.scene.weather.at(preview.thermal.scene.t0_s)
            assert weather.t_air_k==condition['air_temperature_c']+273.15 and weather.wind_speed_m_s==condition['wind_m_s'] and weather.dni_w_m2==weather.dhi_w_m2==0.
            report['condition']=condition
            physics_flags=[]
            for p in stage.Traverse():
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    api=UsdPhysics.RigidBodyAPI(p);physics_flags.append((api.GetRigidBodyEnabledAttr(),api.GetRigidBodyEnabledAttr().Get()));api.CreateRigidBodyEnabledAttr(False)
                if p.IsA(UsdPhysics.Joint):
                    api=UsdPhysics.Joint(p);physics_flags.append((api.GetJointEnabledAttr(),api.GetJointEnabledAttr().Get()));api.CreateJointEnabledAttr(False)
            annotator=rep.AnnotatorRegistry.get_annotator('instance_id_segmentation',init_params={'colorize':False});annotator.attach(preview.rgb_product)
            drone=UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.prim_path));root_matrix=drone.GetLocalTransformation();base_state=copy.deepcopy(preview.thermal.state)
            body_relative=UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.body_path)).ComputeLocalToWorldTransform(0)*drone.ComputeLocalToWorldTransform(0).GetInverse()
            from scripts.drone_control_math import AttitudeConfig
            pitch_limit=mission['camera_pitch_down_deg']+math.degrees(AttitudeConfig().max_angle)
            for zone,scale,role,focus,distance in views:
                mesh=UsdGeom.Mesh(stage.GetPrimAtPath(focus['usd_prim_path']));points=np.asarray(mesh.GetPointsAttr().Get());matrix=np.asarray(mesh.ComputeLocalToWorldTransform(0));points=(np.c_[points,np.ones(len(points))]@matrix)[:,:3]
                face=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())[:3];normal=np.cross(points[face[1]]-points[face[0]],points[face[2]]-points[face[0]]);normal/=np.linalg.norm(normal)
                center=np.asarray(focus['center_m']);bounds=np.asarray(sources[focus['source_object']]['bounds_m']);body=bounds.mean(0)
                eye=np.asarray(focus.get('_observation_eye_m',diagnostic_eye(center,normal,body,distance,scale,role=='ground',pitch_limit)));delta=center-eye;yaw=math.degrees(math.atan2(delta[1],delta[0]));pitch=math.degrees(math.atan2(-delta[2],np.linalg.norm(delta[:2])))
                transform=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),yaw));offset=np.asarray((body_relative*transform).Transform(Gf.Vec3d(*mission['camera_offset_body_m'])))
                if eye[2]-offset[2]<config.drone.minimum_height:eye=center+(eye-center)*((config.drone.minimum_height+offset[2]-center[2])/(eye[2]-center[2]))
                transform.SetTranslateOnly(Gf.Vec3d(*(eye-offset)));drone.MakeMatrixXform().Set(transform)
                install_cameras(stage,config.drone.body_path,mission['camera_offset_body_m'],pitch);timeline.set_current_time(0.);timeline.commit()
                preview.thermal.state=copy.deepcopy(base_state)
                for _ in range(6):app.update()
                started=time.perf_counter();rgb,thermal,meta=preview.capture(render=True,simulation_time=0.,frozen_state=True);capture_s=time.perf_counter()-started
                assert meta['pair_pose_max_error']<1e-6 and meta['thermal_coverage']['complete_for_this_view']
                assert np.isfinite(thermal.apparent_t).all() and np.isfinite(thermal.radiance).all()
                last=preview.thermal._last;role_pixels=defaultdict(int);component_pixels=defaultdict(int)
                report['resolved_aovs']={k:str(v) for k,v in preview.thermal._reader.resolved.items()}
                for ident,count in zip(*np.unique(last.instance_id,return_counts=True)):
                    row=rows.get(last.labels.get(int(ident)))
                    if row and row.get('source_object') in sources:
                        if sources[row['source_object']]['zone_id']==zone:role_pixels[sources[row['source_object']]['role']]+=int(count)
                        component_pixels[row['component']]+=int(count)
                self_ids=[i for i,p in last.labels.items() if p.startswith('/World/Crazyflie/')];assert not np.isin(last.instance_id,self_ids).any()
                segmentation=annotator.get_data();rgb_labels=labels_from_payload(segmentation);assert rgb_labels
                unknown=set(int(i) for i in np.unique(segmentation['data']))-set(rgb_labels)-{0};assert not unknown
                rgb_self=[i for i,p in rgb_labels.items() if str(p).startswith('/World/Crazyflie/')];assert not np.isin(segmentation['data'],rgb_self).any()
                camera=UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.rgb_path));camera_world=camera.ComputeLocalToWorldTransform(0);actual_eye=np.asarray(camera_world.ExtractTranslation())
                assert np.allclose(actual_eye,eye,atol=.0001,rtol=0.),('Mounted camera offset mismatch',actual_eye,eye)
                intrinsics={'focal_length_mm':float(camera.GetFocalLengthAttr().Get()),'horizontal_aperture_mm':float(camera.GetHorizontalApertureAttr().Get()),'vertical_aperture_mm':float(camera.GetVerticalApertureAttr().Get())}
                thermal_camera=UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.thermal_path))
                assert intrinsics=={'focal_length_mm':float(thermal_camera.GetFocalLengthAttr().Get()),'horizontal_aperture_mm':float(thermal_camera.GetHorizontalApertureAttr().Get()),'vertical_aperture_mm':float(thermal_camera.GetVerticalApertureAttr().Get())}
                entry={'zone_id':zone,'scale':scale,'requested_role':role,'resolution':[width,height],'role_pixels':dict(role_pixels),'component_pixels':dict(component_pixels),
                    'native_capture':True,'manual_success':None,'intrinsics':intrinsics,'distance_m':float(np.linalg.norm(center-actual_eye)),
                    'focus_surface':focus['surface_id'],'camera_world_position_m':actual_eye.tolist(),'pair_pose_max_error':meta['pair_pose_max_error'],'capture_seconds':capture_s,'thermal_self_pixels':0,'rgb_self_pixels':0}
                focus_ids=[i for i,p in last.labels.items() if p==focus['usd_prim_path']]
                focus_mask=np.isin(last.instance_id,focus_ids);entry['focus_pixels']=int(focus_mask.sum())
                if scale=='interface':assert entry['focus_pixels']>0,('Requested interface not observed',focus['surface_id'])
                if entry['focus_pixels']:
                    ranges=last.planes['distance_m'][focus_mask];entry['focus_surface_range_m']={'minimum':float(ranges.min()),'median':float(np.median(ranges)),'maximum':float(ranges.max())}
                entry['view_pitch_down_deg']=pitch;entry['equivalent_body_pitch_deg']=pitch-mission['camera_pitch_down_deg'];entry['reachable_optical_pitch_range_deg']=[mission['camera_pitch_down_deg']-math.degrees(AttitudeConfig().max_angle),pitch_limit]
                assert abs(entry['equivalent_body_pitch_deg'])<=math.degrees(AttitudeConfig().max_angle)+1e-6
                feature_range=entry.get('focus_surface_range_m',{}).get('median',entry['distance_m'])
                entry['feature_projected_pixels']={name:projected_pixels(size,feature_range,width,intrinsics['focal_length_mm'],intrinsics['horizontal_aperture_mm']) for name,size in [('frame_35mm',.035),('joint_8mm',.008),('relief_2mm',.002)]}
                if '_fault_kind' in focus:
                    task=next(t for t in tasks if t['kind']==focus['_fault_kind']);hotpaths=set(task['hotspot_prim_paths']);ids=[i for i,p in last.labels.items() if p in hotpaths]
                    entry['fault_kind']=task['kind'];entry['hotspot_pixels']=int(np.isin(last.instance_id,ids).sum());assert entry['hotspot_pixels']>0.
                meta.update(entry);preview.save_frame(args.out,index,rgb,thermal,meta)
                np.savez_compressed(args.out/f'frame_{index:06d}_gbuffer.npz',**last.planes,instance_id=last.instance_id)
                (args.out/f'frame_{index:06d}_paths.json').write_text(json.dumps(last.labels,indent=2))
                Image.fromarray(fixed_temperature_display(thermal.apparent_t,18.,35.)).save(args.out/f'frame_{index:06d}_detail.png')
                report['views'].append(entry);index+=1;print('[v6] capture',zone,scale,role,width,dict(role_pixels),flush=True)
                # Actual captured AOVs drive local +1C tests; no synthetic image/pose evidence.
                for component in ['Wall','WindowFrame','EquipmentShell','PipeWall','PavingBrick','ConcreteRoadSlab','TankShell']:
                    if width!=160 or component in responded:continue
                    candidates=[(int(count),int(ident),rows[last.labels[int(ident)]]) for ident,count in zip(*np.unique(last.instance_id,return_counts=True)) if last.labels.get(int(ident)) in rows and rows[last.labels[int(ident)]]['component']==component and not any(s in rows[last.labels[int(ident)]]['surface_id'].lower() for s in ['hotspot','transition'])]
                    if not candidates:continue
                    count,ident,row=max(candidates,key=lambda x:x[0]);mask=last.instance_id==ident
                    planes={k:np.array(v,copy=True) for k,v in last.planes.items()}
                    target_name=next(a['target'] for a in audit['assignments'] if a['prim_path']==row['usd_prim_path'])
                    scene=preview.thermal.scene;before=preview.thermal.bridge.temperatures();temp=before[target_name]
                    solver=PrescribedSolver([float(scene.weather.time_s[0]),float(scene.weather.time_s[-1])],[temp+1.,temp+1.],t0_s=scene.t0_s)
                    changed_scene=replace(scene,targets={**scene.targets,target_name:solver})
                    bridge=AerialThermalBridge(changed_scene,preview.thermal.bridge.prim_to_target,band=preview.thermal.bridge.band,sky=preview.thermal.bridge.sky)
                    bridge.advance_to(preview.thermal.bridge.t_rel_s);after=bridge.temperatures()
                    assert all(abs(value-after[name])<1e-8 for name,value in before.items() if name!=target_name)
                    new_temperature=bridge.temperature_plane(last.instance_id,last.labels,sky_mask=planes.get('sky_mask'),elevation_rad=last.elevation_rad)
                    response=check_local_temperature_response(planes['temperature_k'],new_temperature,mask,1.)
                    outputs=[run_frame(dict(planes,temperature_k=t),copy.deepcopy(preview.thermal.config),copy.deepcopy(base_state)) for t in [planes['temperature_k'],new_temperature]]
                    delta=outputs[1].radiance-outputs[0].radiance;roi=native_roi(mask,delta.shape);assert float(np.max(delta[roi]))>0.
                    response.update(component=component,surface_id=row['surface_id'],target=target_name,pixels=count,other_targets_unchanged=len(before)-1,peak_radiance_change=float(np.max(delta)),selected_native_pixels=int(roi.sum()),selected_peak_radiance_change=float(np.max(delta[roi])))
                    folder=args.out/('response_'+component);folder.mkdir();np.save(folder/'selected_mask.npy',mask);np.save(folder/'native_sensor_roi.npy',roi);np.save(folder/'edited_temperature_k.npy',new_temperature)
                    np.save(folder/'baseline_radiance.npy',outputs[0].radiance);np.save(folder/'plus1C_radiance.npy',outputs[1].radiance)
                    (folder/'result.json').write_text(json.dumps(response,indent=2));report['responses'].append(response);responded.add(component)
                    if 'same_temperature_materials' not in report:
                        a,b='industrial_painted_steel','bare_aluminium';names=preview.thermal.config.materials.names
                        variants=[];fixed_t=planes['temperature_k'].copy();fixed_t[mask]=300.15
                        for material in [a,b]:
                            ids=planes['material_id'].copy();ids[mask]=names.index(material)
                            variant=run_frame(dict(planes,temperature_k=fixed_t,material_id=ids),copy.deepcopy(preview.thermal.config),copy.deepcopy(base_state));variants.append(variant)
                        peak=float(np.max(np.abs(variants[0].radiance-variants[1].radiance)))
                        assert peak>0.;report['same_temperature_materials']={'materials':[a,b],'source_temperature_c':27.,'peak_radiance_difference':peak,'scope':'Same actual AOV geometry, orientation, environment and sensor state; library optical proxies only'}
                        for material,variant in zip([a,b],variants):np.save(args.out/(material+'_same27C_radiance.npy'),variant.radiance)
            if width==160:report['control_smoke']=control_smoke(app,stage,config,root_matrix,physics_flags)
            annotator.detach(preview.rgb_product);annotator=None;preview.close();preview=None
        if args.resolution is None:
            report['coverage']=validate_view_evidence(report['views'],registry);report['result']='PASS_V6_NATIVE_MULTISCALE_VIEWS'
        else:report['result']='PASS_V6_NATIVE_RESOLUTION';report['coverage']={'complete':False,'awaiting_other_resolution':True}
        if args.resolution in [None,160]:assert {'Wall','WindowFrame','EquipmentShell','PipeWall','ConcreteRoadSlab','TankShell'}<=responded
    except BaseException as exc:report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if annotator and preview:annotator.detach(preview.rgb_product)
        if preview:preview.close()
        if app:app.close()
    return int(report['result'] not in ['PASS_V6_NATIVE_MULTISCALE_VIEWS','PASS_V6_NATIVE_RESOLUTION'])


if __name__=='__main__':sys.exit(main())
