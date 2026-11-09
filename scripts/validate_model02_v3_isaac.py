"""Native Isaac actual RGB/LWIR checks and controlled replay of local target edits."""
import argparse,copy,hashlib,json,sys,traceback
from dataclasses import asdict,replace
from pathlib import Path
from datetime import timedelta
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ASSET=ROOT/'factory/model02_thermal_v3'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    report={'result':'FAIL','data_usage':'scene migration validation, not manual flight collection','manual_success':None,'views':[],'local_responses':[]}
    preview=app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'renderer':'PathTracing','multi_gpu':False,'samples_per_pixel_per_frame':16})
        import omni.timeline
        from pxr import Gf,UsdGeom,UsdLux,UsdPhysics
        from PIL import Image
        from scipy.ndimage import maximum_filter
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.thermal_config import load_thermal_config
        from scripts.thermal_scene_config import apply_thermal_config
        from scripts.model02_response import check_local_temperature_response
        from scripts.thermal_imaging import display_from_outputs
        from irsim_isaac.pipeline.aerial_bridge import AerialThermalBridge
        from irsim.pipeline.frame import run_frame
        config=load_scene_config(ROOT/'configs/model02_v3_tank_scene.yaml')
        manifest=json.loads((ASSET/'isaac_manifest.json').read_text(encoding='utf-8'))
        by_path={r['usd_prim_path']:r for r in manifest['surfaces']}
        preview,_=open_preview(app,scene_config=config,width=960,height=768,thermal_width=640,thermal_height=512,
            fast=False,start_controller=False,show_ui=False)
        stage=preview.stage;stage.SetEditTarget(stage.GetSessionLayer())
        assert UsdGeom.GetStageUpAxis(stage)=='Z' and UsdGeom.GetStageMetersPerUnit(stage)==1
        assert stage.GetPrimAtPath(config.drone.body_path).HasAPI(UsdPhysics.RigidBodyAPI)
        physics=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)];assert len(physics)==1
        for path in manifest['inactive_original_nodes']:assert not stage.GetPrimAtPath(path).IsActive()
        assert len([p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)])==14
        collider=json.loads((ASSET/'collision_report.json').read_text(encoding='utf-8'))
        collision_prim=stage.GetPrimAtPath(collider['path'])
        assert collision_prim.HasAPI(UsdPhysics.CollisionAPI) and UsdGeom.Imageable(collision_prim).ComputeVisibility()=='invisible'
        assert all(not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI) for path in by_path)
        report['collision']=collider
        thermal_report=preview.thermal.thermal_config_report
        (args.out/'thermal_binding_audit.json').write_text(json.dumps(thermal_report,ensure_ascii=False,indent=2),encoding='utf-8')
        assert not thermal_report['fallback_paths'],str(thermal_report['fallback_paths'][:5])
        assert not thermal_report['unmatched_bindings'],str(thermal_report['unmatched_bindings'][:5])
        assert preview.metadata['authored_lighting']['configuration_verified']
        bound_parameters=json.loads((ASSET/'binding_configuration_report.json').read_text(encoding='utf-8'))
        assert thermal_report['material_library_sha256']==bound_parameters['material_library_sha256']
        report['binding']={'explicit_surfaces':len(by_path),'active_renderable_paths':len(thermal_report['assignments']),
            'fallback_paths':len(thermal_report['fallback_paths']),'unmatched_bindings':len(thermal_report['unmatched_bindings']),
            'preserved_background_coarse_paths':thermal_report['binding_counts'].get('preserved_background_coarse',0),
            'material_library_sha256':thermal_report['material_library_sha256'],'physics_scenes':physics}
        scene=preview.thermal.scene;weather=scene.weather.at(scene.t0_s)
        condition=json.loads((ASSET/'configuration/blender_parameters.json').read_text(encoding='utf-8'))['condition']
        assert weather.t_air_k==condition['air_temperature_c']+273.15 and weather.wind_speed_m_s==condition['wind_m_s'] and weather.dni_w_m2==weather.dhi_w_m2==0
        assert all(s.weather is scene.weather for s in scene.sky_models.values())
        report['night_environment']={'weather':asdict(weather),'start_utc':str(scene.weather.start_utc+timedelta(seconds=scene.t0_s)),
            'weather_series_start_utc':str(scene.weather.start_utc),
            'scene_t0_s':scene.t0_s,'shared_weather_identity_verified':True,'site':'Nominal31.2N121.5E; original site unknown'}
        from scripts.runtime_parameter_provenance import thermal_input_snapshot,validate_thermal_inputs
        snapshot=thermal_input_snapshot(config);validate_thermal_inputs(config,snapshot)
        assert snapshot['material_library_sha256']==thermal_report['material_library_sha256']
        assert snapshot['weather_content_sha256']==scene.weather.content_hash
        report['thermal_input_snapshot']=snapshot
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        tasks={t['kind']:t for t in manifest['tasks']}
        views=[('overview',(646.,-867.,44.),(700.,-809.,3.),None),
            ('facade',(740.,-850.,8.),(720.,-825.,2.8),'wall'),
            ('window_detail',(718.,-836.,4.),(718.3,-828.8,3.65),None),
            ('tank',tasks['tank']['observation_position_m'],tasks['tank']['target_center_m'],'tank'),
            ('pipe',tasks['pipe']['observation_position_m'],tasks['pipe']['target_center_m'],'pipe')]
        thermal_config=load_thermal_config(config.sensors.thermal_config)
        authored_targets={r['target'] for r in json.loads((ASSET/'configuration/blender_parameters.json').read_text(encoding='utf-8'))['objects'].values()}
        for index,(name,eye,target,response_kind) in enumerate(views):
            transform=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
            for path in [preview.mount.rgb_path,preview.mount.thermal_path]:
                prim=stage.GetPrimAtPath(path);parent=UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
                UsdGeom.Xformable(prim).MakeMatrixXform().Set(transform*parent.GetInverse())
            timeline.set_current_time(0.);timeline.commit()
            # Independent static viewpoints must not inherit the preceding
            # view's membrane image. Public state fork resets history buffers.
            preview.thermal.state=replace(preview.thermal.state)
            assert not preview.thermal.state.buffers
            rgb,thermal,meta=preview.capture(render=True,simulation_time=0.,frozen_state=True)
            assert meta['thermal_coverage']['complete_for_this_view'],name
            assert np.isfinite(thermal.radiance).all() and np.isfinite(thermal.apparent_t).all(),name
            assert thermal.dn16 is not None
            meta.update(view_name=name,data_usage=report['data_usage'],manual_success=None,
                detector_history_policy='Independent static view; state fork clears previous viewpoint buffers. Live flight unchanged.')
            preview.save_frame(args.out,index,rgb,thermal,meta)
            last=preview.thermal._last
            np.save(args.out/f'frame_{index:06d}_instance_id.npy',last.instance_id)
            np.save(args.out/f'frame_{index:06d}_surface_temperature_k.npy',last.planes['temperature_k'])
            (args.out/f'frame_{index:06d}_instance_paths.json').write_text(json.dumps(last.labels,ensure_ascii=False,indent=2),encoding='utf-8')
            pixels=np.asarray(rgb)[...,:3].astype(float);luma=pixels@np.array([.2126,.7152,.0722])
            item={'view':name,'frame':index,'eye':list(eye),'target':list(target),'coverage':meta['thermal_coverage'],
                'rgb_luma_percentiles_8bit':np.percentile(luma,[5,50,95,99]).tolist(),
                'nonblack_fraction':float((luma>3).mean()),'saturated_fraction':float((pixels>=254).all(-1).mean()),
                'apparent_temperature_range_c':[float(thermal.apparent_t.min()-273.15),float(thermal.apparent_t.max()-273.15)]}
            assert item['saturated_fraction']<.02,name
            assert item['nonblack_fraction']>(.08 if name=='overview' else .15),name
            if name in tasks:
                corepaths=set(tasks[name]['hotspot_prim_paths'])
                core_ids=[i for i,p in last.labels.items() if p in corepaths]
                item['visible_hotspot_gbuffer_pixels']=int(np.isin(last.instance_id,core_ids).sum())
                assert item['visible_hotspot_gbuffer_pixels']>0,name
            report['views'].append(item);print('[v3] captured '+name,flush=True)
            if response_kind is None:continue
            candidates=[]
            for ident,count in zip(*np.unique(last.instance_id,return_counts=True)):
                path=last.labels.get(int(ident));row=by_path.get(path)
                if row is None:continue
                match=(row['source_object']=='11_29_gc021' and row['component']=='Wall' if response_kind=='wall' else
                       row['source_object']=='11_29_gc075' and 'TankShell' in row['surface_id'] and 'Hotspot' not in row['surface_id'] and 'Transition' not in row['surface_id'] if response_kind=='tank' else
                       row['source_object']=='11_29_gc089' and 'PipeDetail' in row['surface_id'])
                if match:candidates.append((int(count),int(ident),row))
            assert candidates,'No visible independently editable '+response_kind
            count,ident,row=max(candidates,key=lambda v:v[0]);target_name=thermal_config.target_for(row['usd_prim_path'])
            original_target=thermal_config.target(target_name);assert original_target.mode=='fixed'
            changed=replace(thermal_config,targets=tuple(replace(t,temperature_k=t.temperature_k+10.) if t.name==target_name else t for t in thermal_config.targets))
            base_scene=replace(scene,targets={k:v for k,v in scene.targets.items() if k not in {t.name for t in thermal_config.targets}})
            changed_scene,mapping,_,_=apply_thermal_config(base_scene,changed,list(preview.thermal.bridge.prim_to_target),preview.thermal.config.materials.names)
            bridge=AerialThermalBridge(changed_scene,mapping,band=preview.thermal.bridge.band,sky=preview.thermal.bridge.sky)
            bridge.advance_to(preview.thermal.bridge.t_rel_s)
            before_values=preview.thermal.bridge.temperatures();after_values=bridge.temperatures()
            assert all(abs(after_values[t]-before_values[t])<1e-8 for t in authored_targets if t!=target_name)
            mask=last.instance_id==ident;planes={k:np.array(v,copy=True) for k,v in last.planes.items()}
            changed_temperature=bridge.temperature_plane(last.instance_id,last.labels,sky_mask=planes.get('sky_mask'),elevation_rad=last.elevation_rad)
            result=check_local_temperature_response(planes['temperature_k'],changed_temperature,mask,10.)
            # Reuse actual rendered geometry and identical sensor state/random seed.
            # This isolates a temperature edit from detector noise and renderer jitter.
            outputs=[]
            for temperatures in [planes['temperature_k'],changed_temperature]:
                p=dict(planes,temperature_k=temperatures)
                outputs.append(run_frame(p,copy.deepcopy(preview.thermal.config),copy.deepcopy(preview.thermal.state)))
            before,after=outputs;delta=np.asarray(after.radiance)-np.asarray(before.radiance)
            h,w=delta.shape;k=mask.shape[0]//h
            native_mask=mask.reshape(h,k,w,k).any(axis=(1,3))
            psf=preview.thermal.config.psf;radius=0 if psf is None else int(np.ceil(max(psf.shape)//2/k))+1
            support=maximum_filter(native_mask.astype(np.uint8),size=2*radius+1)>0 if radius else native_mask
            outside=float(np.max(np.abs(delta[~support]),initial=0));inside=float(np.max(delta[native_mask],initial=0))
            assert inside>1e-5 and outside<1e-4,(response_kind,inside,outside)
            folder=args.out/('response_'+response_kind);folder.mkdir()
            np.savez_compressed(folder/'actual_gbuffer.npz',**planes,instance_id=last.instance_id)
            np.save(folder/'changed_temperature_k.npy',changed_temperature);np.save(folder/'selected_mask.npy',mask)
            for label,out in zip(['baseline','plus10C'],outputs):
                np.save(folder/(label+'_radiance.npy'),out.radiance);np.save(folder/(label+'_apparent_t.npy'),out.apparent_t)
                np.save(folder/(label+'_dn16.npy'),out.dn16)
                display=display_from_outputs(preview.thermal.config,out,preview.display_profile).display8
                Image.fromarray(np.asarray(display)).save(folder/(label+'_thermal.png'))
            result.update(kind=response_kind,surface_id=row['surface_id'],object=row['object'],usd_prim_path=row['usd_prim_path'],
                target=target_name,original_temperature_c=original_target.temperature_k-273.15,test_temperature_c=original_target.temperature_k-263.15,
                other_authored_targets_unchanged=len(authored_targets)-1,radiance_peak_increase=inside,
                radiance_max_change_outside_optical_support=outside,optical_support_radius_native_pixels=radius,
                validation_method='Controlled radiometric replay of actual Isaac G-buffer, identical sensor configuration/state/seed; no saved production parameter changed')
            (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            report['local_responses'].append(result);print('[v3] local response '+response_kind+' PASS',flush=True)
        report['route_geometry']=[{k:t[k] for k in ['kind','minimum_route_body_clearance_lower_bound_m','observation_equipment_clearance_m','visible_core_triangles','total_core_triangles','camera_to_hotspot_m']} for t in manifest['tasks']]
        report['authored_lighting']=preview.metadata['authored_lighting']
        report['result']='PASS_ISAAC_MIGRATION_BINDING_AND_LOCAL_RESPONSE'
    except BaseException as exc:
        report['error']=str(exc);report['traceback']=traceback.format_exc();traceback.print_exc()
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('[v3] '+report['result'],flush=True)
        try:
            if preview:preview.close()
            if app:app.close()
        except BaseException as exc:
            report.update(result='FAIL',cleanup_error=str(exc))
            (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['result']=='PASS_ISAAC_MIGRATION_BINDING_AND_LOCAL_RESPONSE' else 1


if __name__=='__main__':sys.exit(main())
