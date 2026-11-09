"""Actual mounted-UAV captures: no manual-flight success is fabricated."""
import argparse,copy,hashlib,json,math,sys,time,traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ASSET=ROOT/'factory/model02_thermal_v4'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(exist_ok=False,parents=True)
    report={'result':'FAIL','manual_success':None,'data_usage':'Mounted camera validation; static scripted poses, not human flight','views':[],'responses':[]}
    app=preview=annotator=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'renderer':'PathTracing','samples_per_pixel_per_frame':16,'multi_gpu':False})
        import omni.replicator.core as rep
        import omni.timeline
        from pxr import Gf,Usd,UsdGeom,UsdPhysics
        from PIL import Image
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.thermal_display import load_display_profile,fixed_temperature_display
        from scripts.thermal_config import load_thermal_config
        from scripts.thermal_scene_config import apply_thermal_config
        from scripts.build_model02_v4_scene import install_cameras
        from scripts.model02_response import check_local_temperature_response
        from irsim_isaac.pipeline.aerial_bridge import AerialThermalBridge
        from irsim.pipeline.frame import run_frame
        from irsim_isaac.pipeline.material_ids import labels_from_payload
        config=load_scene_config(ROOT/'configs/model02_v4_tank_scene.yaml')
        manifest=json.loads((ASSET/'isaac_manifest.json').read_text(encoding='utf-8'))
        report['scene_sha256']=hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
        report['thermal_config_sha256']=hashlib.sha256(config.sensors.thermal_config.read_bytes()).hexdigest()
        parameters=json.loads((ASSET/'mission_and_camera.json').read_text())
        rows={r['usd_prim_path']:r for r in manifest['surfaces']}
        preview,_=open_preview(app,scene_config=config,width=640,height=512,thermal_width=640,thermal_height=512,fast=False,start_controller=False,show_ui=False)
        stage=preview.stage;stage.SetEditTarget(preview.mount.layer)
        assert UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
        ground=stage.GetPrimAtPath('/World/IndustrialPark02/Collision/RoadGC051')
        assert ground.HasAPI(UsdPhysics.CollisionAPI) and UsdGeom.Imageable(ground).ComputeVisibility()=='invisible'
        audit=preview.thermal.thermal_config_report
        compiled=json.loads((ASSET/'configuration/blender_parameters.json').read_text(encoding='utf-8'))
        assert audit['material_library_sha256']==compiled['material_library_sha256']
        (args.out/'binding_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        assert not audit['fallback_paths'] and not audit['unmatched_bindings']
        report['bindings']={'explicit_surfaces':len(rows),'active_paths':len(audit['assignments']),'fallbacks':len(audit['fallback_paths']),'unmatched':len(audit['unmatched_bindings'])}
        assert preview.metadata['authored_lighting']['configuration_verified']
        rgb_camera=UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.rgb_path));ir_camera=UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.thermal_path))
        for getter in ['GetFocalLengthAttr','GetHorizontalApertureAttr','GetVerticalApertureAttr','GetClippingRangeAttr']:
            np.testing.assert_allclose(getattr(rgb_camera,getter)().Get(),getattr(ir_camera,getter)().Get(),atol=1e-6)
        annotator=rep.AnnotatorRegistry.get_annotator('instance_id_segmentation',init_params={'colorize':False});annotator.attach(preview.rgb_product)
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        drone=UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.prim_path))
        detail=load_display_profile(ROOT/'configs/model02_v4_detail_display.yaml')
        task=next(t for t in manifest['original_tasks'] if t['kind']=='tank')
        eye=task['observation_position_m'];target=task['target_center_m'];yaw=math.degrees(math.atan2(target[1]-eye[1],target[0]-eye[0]))
        paved=[r for r in manifest['surfaces'] if r['component']=='PavingBrick']
        chosen=min(paved,key=lambda r:np.linalg.norm(np.asarray(r['center_m'])[:2]-[718,-836]))
        center=chosen['center_m']
        views=[('start',parameters['start_position_m'],90.,15.,None),
            ('road_approach',[677,-816,12],55.,15.,None),
            ('tank',eye,yaw,15.,'TankShell'),
            ('facade',[718,-836,4.],90.,4.,'Wall'),
            ('paving_close',[center[0],center[1]-1.5,3.5],90.,65.,'PavingBrick')]
        thermal_config=load_thermal_config(config.sensors.thermal_config)
        all_targets={r['target'] for r in json.loads((ASSET/'configuration/blender_parameters.json').read_text(encoding='utf-8'))['objects'].values()}
        scene=preview.thermal.scene
        weather=scene.weather.at(scene.t0_s)
        assert weather.t_air_k==compiled['condition']['air_temperature_c']+273.15
        assert weather.wind_speed_m_s==compiled['condition']['wind_m_s'] and weather.dni_w_m2==weather.dhi_w_m2==0.
        assert all(s.weather is scene.weather for s in scene.sky_models.values())
        report['environment']={'authoring_condition':compiled['condition'],'runtime_air_c':weather.t_air_k-273.15,
            'runtime_wind_m_s':weather.wind_speed_m_s,'shared_sky_weather':True,
            'scene_config_sha256':hashlib.sha256(config.source_path.read_bytes()).hexdigest(),
            'environment_config_sha256':hashlib.sha256(config.scene.thermal_scene.read_bytes()).hexdigest()}
        for index,(name,position,yaw_deg,pitch,response_component) in enumerate(views):
            matrix=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),yaw_deg));matrix.SetTranslateOnly(Gf.Vec3d(*position));drone.MakeMatrixXform().Set(matrix)
            # Set the body-local installation; never teleport an optical camera
            # independently in world coordinates.65deg is an explicit close-ground
            # installation diagnostic; released mission remains15deg.
            install_cameras(stage,config.drone.body_path,parameters['camera_offset_body_m'],pitch)
            preview.thermal.state=replace(preview.thermal.state)
            timeline.set_current_time(0.);timeline.commit()
            for _ in range(8):app.update()
            started=time.perf_counter();rgb,thermal,meta=preview.capture(render=True,simulation_time=0.,frozen_state=True)
            assert meta['pair_pose_max_error']<1e-6
            assert meta['thermal_coverage']['complete_for_this_view']
            assert np.isfinite(thermal.apparent_t).all() and np.isfinite(thermal.radiance).all()
            last=preview.thermal._last
            thermal_ids=[i for i,p in last.labels.items() if p.startswith('/World/Crazyflie/')]
            thermal_self=int(np.isin(last.instance_id,thermal_ids).sum())
            segmentation=annotator.get_data();rgb_labels=labels_from_payload(segmentation);rgb_ids=[]
            assert any(str(value).startswith('/World/') for value in rgb_labels.values()),'RGB instance annotator has no actual prim paths'
            unknown_ids=set(int(i) for i in np.unique(segmentation['data']))-set(rgb_labels)-{0}
            assert not unknown_ids,('Unmapped RGB IDs cannot certify self-body exclusion',sorted(unknown_ids))
            for ident,value in rgb_labels.items():
                if '/World/Crazyflie/' in str(value):rgb_ids.append(int(ident))
            rgb_self=int(np.isin(segmentation['data'],rgb_ids).sum())
            assert thermal_self==rgb_self==0,(name,thermal_self,rgb_self)
            meta.update(view=name,body_position_m=position,body_yaw_deg=yaw_deg,camera_pitch_down_deg=pitch,manual_success=None,data_usage=report['data_usage'])
            preview.save_frame(args.out,index,rgb,thermal,meta)
            detailed=fixed_temperature_display(thermal.apparent_t,detail.temperature_min_c,detail.temperature_max_c)
            Image.fromarray(detailed).save(args.out/f'frame_{index:06d}_thermal_18_28C.png')
            np.save(args.out/f'frame_{index:06d}_instance_id.npy',last.instance_id)
            np.save(args.out/f'frame_{index:06d}_source_temperature_k.npy',last.planes['temperature_k'])
            np.save(args.out/f'frame_{index:06d}_rgb_instance_id.npy',segmentation['data'])
            (args.out/f'frame_{index:06d}_instance_paths.json').write_text(json.dumps(last.labels,indent=2),encoding='utf-8')
            item={'view':name,'position_m':position,'yaw_deg':yaw_deg,'pitch_down_deg':pitch,'thermal_self_pixels':thermal_self,'rgb_self_pixels':rgb_self,
                'capture_wall_s':time.perf_counter()-started,'pair_pose_max_error':meta['pair_pose_max_error'],
                'apparent_range_c':[float(thermal.apparent_t.min()-273.15),float(thermal.apparent_t.max()-273.15)],'visible_components':{},'rgb_instance_labels':rgb_labels}
            for ident,count in zip(*np.unique(last.instance_id,return_counts=True)):
                row=rows.get(last.labels.get(int(ident)))
                if row:item['visible_components'][row['component']]=item['visible_components'].get(row['component'],0)+int(count)
            if name=='tank':
                hotids=[i for i,p in last.labels.items() if p in set(task['hotspot_prim_paths'])]
                item['hotspot_pixels']=int(np.isin(last.instance_id,hotids).sum());assert item['hotspot_pixels']>0
            if name=='paving_close':
                assert item['visible_components'].get('PavingBrick',0)>100
                assert item['visible_components'].get('PavingJoint',0)>0
                item['paving_temperature_statistics']={}
                for component in ['PavingBrick','PavingJoint']:
                    ids=[i for i,p in last.labels.items() if rows.get(p,{}).get('component')==component]
                    mask=np.isin(last.instance_id,ids)
                    source=last.planes['temperature_k'][mask]-273.15
                    item['paving_temperature_statistics'][component]={'pixels':int(mask.sum()),'surface_temperature_min_max_c':[float(source.min()),float(source.max())]}
            report['views'].append(item);print('[v4] capture '+name+' PASS',flush=True)
            if response_component is None:continue
            candidates=[]
            for ident,count in zip(*np.unique(last.instance_id,return_counts=True)):
                row=rows.get(last.labels.get(int(ident)))
                if row and row['component']==response_component and 'Hotspot' not in row['surface_id'] and 'Transition' not in row['surface_id']:
                    candidates.append((int(count),int(ident),row))
            assert candidates,'No visible editable '+response_component
            count,ident,row=max(candidates,key=lambda v:v[0]);name_target=thermal_config.target_for(row['usd_prim_path'])
            changed=replace(thermal_config,targets=tuple(replace(t,temperature_k=t.temperature_k+1.) if t.name==name_target else t for t in thermal_config.targets))
            base_scene=replace(scene,targets={k:v for k,v in scene.targets.items() if k not in {t.name for t in thermal_config.targets}})
            changed_scene,mapping,_,_=apply_thermal_config(base_scene,changed,list(preview.thermal.bridge.prim_to_target),preview.thermal.config.materials.names)
            bridge=AerialThermalBridge(changed_scene,mapping,band=preview.thermal.bridge.band,sky=preview.thermal.bridge.sky);bridge.advance_to(preview.thermal.bridge.t_rel_s)
            before=preview.thermal.bridge.temperatures();after=bridge.temperatures()
            assert all(abs(before[t]-after[t])<1e-8 for t in all_targets if t!=name_target)
            planes={k:np.array(v,copy=True) for k,v in last.planes.items()};mask=last.instance_id==ident
            new_temperature=bridge.temperature_plane(last.instance_id,last.labels,sky_mask=planes.get('sky_mask'),elevation_rad=last.elevation_rad)
            response=check_local_temperature_response(planes['temperature_k'],new_temperature,mask,1.)
            outputs=[run_frame(dict(planes,temperature_k=v),copy.deepcopy(preview.thermal.config),copy.deepcopy(preview.thermal.state)) for v in [planes['temperature_k'],new_temperature]]
            delta=np.asarray(outputs[1].radiance)-np.asarray(outputs[0].radiance)
            h,w=delta.shape;k=mask.shape[0]//h;native_mask=mask.reshape(h,k,w,k).any(axis=(1,3))
            assert float(np.max(delta[native_mask]))>0
            folder=args.out/('response_'+response_component);folder.mkdir()
            np.savez_compressed(folder/'actual_gbuffer.npz',**planes,instance_id=last.instance_id)
            np.save(folder/'edited_temperature_k.npy',new_temperature);np.save(folder/'selected_mask.npy',mask)
            for label,out in zip(['baseline','plus1C'],outputs):
                np.save(folder/(label+'_radiance.npy'),out.radiance);np.save(folder/(label+'_apparent_t.npy'),out.apparent_t)
            response.update(component=response_component,surface_id=row['surface_id'],target=name_target,other_targets_unchanged=len(all_targets)-1,
                peak_radiance_change=float(np.max(delta[native_mask])))
            (folder/'result.json').write_text(json.dumps(response,indent=2),encoding='utf-8');report['responses'].append(response)
        # Actual PhysX/controller smoke check, separate from human flight evidence.
        matrix=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),parameters['start_yaw_deg']))
        matrix.SetTranslateOnly(Gf.Vec3d(*parameters['start_position_m']));drone.MakeMatrixXform().Set(matrix)
        install_cameras(stage,config.drone.body_path,parameters['camera_offset_body_m'],parameters['camera_pitch_down_deg'])
        from scripts.start_crazyflie_controller import start_crazyflie
        from scripts.start_crazyflie_dual_camera_view import preview_motion_config
        controller=start_crazyflie(prim_path=config.drone.prim_path,minimum_height=config.drone.minimum_height,
            motion_config=preview_motion_config(1.,1.5))
        try:
            timeline.set_auto_update(True);timeline.play()
            for _ in range(180):
                app.update()
                if controller._ready:break
            assert controller._ready,'Manual controller failed to initialize with new floor collision'
            first=UsdGeom.Xformable(stage.GetPrimAtPath(preview.mount.rgb_path)).ComputeLocalToWorldTransform(0)
            controller.position=(controller.position[0]+.2,controller.position[1],controller.position[2]+.1);controller.yaw+=.1
            for _ in range(8):app.update()
            rgb_world=UsdGeom.Xformable(stage.GetPrimAtPath(preview.mount.rgb_path)).ComputeLocalToWorldTransform(0)
            ir_world=UsdGeom.Xformable(stage.GetPrimAtPath(preview.mount.thermal_path)).ComputeLocalToWorldTransform(0)
            error=float(np.max(np.abs(np.array(rgb_world)-np.array(ir_world))))
            distance=float(np.linalg.norm(np.array(rgb_world.ExtractTranslation())-np.array(first.ExtractTranslation())))
            assert error<1e-6 and distance>.1,(error,distance)
            report['controller_smoke']={'ready':True,'actual_camera_translation_m':distance,'pair_pose_max_error':error,
                'data_usage':'Scripted controller movement, not human keyboard flight','manual_success':None}
        finally:controller.stop();timeline.pause()
        report['result']='PASS_V4_MOUNTED_CAMERAS_AND_THERMAL_SURFACES'
    except BaseException as exc:
        report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('[v4] '+report['result'],flush=True)
        try:
            if annotator and preview:annotator.detach(preview.rgb_product)
            if preview:preview.close()
            if app:app.close()
        except BaseException as exc:
            report.update(result='FAIL',cleanup_error=str(exc));(args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return int(report['result']!='PASS_V4_MOUNTED_CAMERAS_AND_THERMAL_SURFACES')


if __name__=='__main__':sys.exit(main())
