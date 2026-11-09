"""Actual Isaac ground captures and same-view comparison with the preserved v4 layer."""
import argparse,json,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    report={'result':'FAIL','manual_success':None,'data_usage':'Static mounted-camera validation, not human flight','views':[]}
    app=preview=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'renderer':'PathTracing','samples_per_pixel_per_frame':16,'multi_gpu':False})
        import omni.timeline
        from pxr import Gf,UsdGeom,UsdPhysics
        from PIL import Image,ImageDraw
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.build_model02_v4_scene import install_cameras
        from scripts.thermal_display import fixed_temperature_display
        config=load_scene_config(ROOT/'configs/model02_v5_tank_scene.yaml')
        manifest=json.loads((ROOT/'factory/model02_thermal_v4/isaac_manifest.json').read_text(encoding='utf-8'))
        rows={r['usd_prim_path']:r for r in manifest['surfaces']}
        preview,_=open_preview(app,scene_config=config,width=640,height=512,thermal_width=640,thermal_height=512,fast=False,start_controller=False,show_ui=False)
        stage=preview.stage;stage.SetEditTarget(stage.GetSessionLayer())
        assert stage.GetPrimAtPath(config.drone.body_path).IsValid()
        assert UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
        audit=preview.thermal.thermal_config_report
        assert not audit['fallback_paths'] and not audit['unmatched_bindings']
        report['binding_counts']={'active_paths':len(audit['assignments']),'fallbacks':0,'unmatched':0}
        assert preview.metadata['authored_lighting']['configuration_verified']
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.IsA(UsdPhysics.Joint):UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        drone=UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.prim_path))
        mission=json.loads((ROOT/'factory/model02_thermal_v4/mission_and_camera.json').read_text())
        source=json.loads((ROOT/'factory/model02_thermal_v4/surfaces.json').read_text(encoding='utf-8'))['surfaces']
        brick=min((r for r in source if r['component']=='PavingBrick'),key=lambda r:np.linalg.norm(np.array(r['center_m'])[:2]-[718,-839.9]))
        c=brick['center_m'];close=[c[0],c[1]-1.2,1.4]
        views=[('start',[700.,-845.,8.5],90.,15.),('paving_close',close,90.,47.),
               ('grass',[700.,-845.,4.],90.,55.),('road',[708.,-842.,4.5],90.,60.)]
        ground=str((ROOT/'factory/model02_thermal_v5/model02_v5_ground.usdc').resolve())
        images={};index=0
        for name,position,yaw,pitch in views:
            matrix=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),yaw));matrix.SetTranslateOnly(Gf.Vec3d(*position));drone.MakeMatrixXform().Set(matrix)
            install_cameras(stage,config.drone.body_path,mission['camera_offset_body_m'],pitch)
            for mode in (['v5','v4_baseline'] if name=='paving_close' else ['v5']):
                if mode=='v4_baseline':stage.MuteLayer(ground)
                timeline.set_current_time(0.);timeline.commit()
                for _ in range(8):app.update()
                rgb,thermal,meta=preview.capture(render=True,simulation_time=0.,frozen_state=True)
                assert meta['pair_pose_max_error']<1e-6 and meta['thermal_coverage']['complete_for_this_view']
                assert np.isfinite(thermal.apparent_t).all() and np.isfinite(thermal.radiance).all()
                last=preview.thermal._last;components={}
                for ident,count in zip(*np.unique(last.instance_id,return_counts=True)):
                    row=rows.get(last.labels.get(int(ident)))
                    if row:components[row['component']]=components.get(row['component'],0)+int(count)
                own=[ident for ident,path in last.labels.items() if path.startswith('/World/Crazyflie/')]
                self_pixels=int(np.isin(last.instance_id,own).sum());assert self_pixels==0
                meta.update(view=name,variant=mode,muted_layers=stage.GetMutedLayers(),manual_success=None,data_usage=report['data_usage'])
                preview.save_frame(args.out,index,rgb,thermal,meta)
                Image.fromarray(fixed_temperature_display(thermal.apparent_t,18.,28.)).save(args.out/f'{name}_{mode}_thermal_detail.png')
                images[(name,mode)]=Image.fromarray(rgb[:,:,:3])
                images[(name,mode)].save(args.out/f'{name}_{mode}_rgb.png')
                np.save(args.out/f'{name}_{mode}_instance_id.npy',last.instance_id)
                (args.out/f'{name}_{mode}_paths.json').write_text(json.dumps(last.labels,indent=2),encoding='utf-8')
                item={'view':name,'variant':mode,'components':components,'thermal_self_pixels':self_pixels,'pair_pose_max_error':meta['pair_pose_max_error']}
                if name=='road':assert components.get('ConcreteRoadSlab',0)>1000 and components.get('RoadJoint',0)>0
                if name=='grass':assert components.get('Landscape',0)>1000
                if name=='paving_close':
                    assert components.get('PavingBrick',0)>1000 and components.get('PavingJoint',0)>0
                    item['temperature_c']={}
                    for component in ['PavingBrick','PavingJoint']:
                        ids=[ident for ident,path in last.labels.items() if rows.get(path,{}).get('component')==component]
                        values=last.planes['temperature_k'][np.isin(last.instance_id,ids)]-273.15
                        item['temperature_c'][component]=[float(values.min()),float(values.max())]
                report['views'].append(item);index+=1
                print('[v5] capture',name,mode,'PASS',flush=True)
                if mode=='v4_baseline':stage.UnmuteLayer(ground)
        assert sum(v['components'].get('Landscape',0) for v in report['views'] if v['variant']=='v5')>0
        before=images[('paving_close','v4_baseline')];after=images[('paving_close','v5')]
        comparison=Image.new('RGB',(1280,552),(25,27,30));comparison.paste(before,(0,40));comparison.paste(after,(640,40))
        draw=ImageDraw.Draw(comparison);draw.text((12,12),'V4: original flat joints',fill='white');draw.text((652,12),'V5: shallow filled joints + small edge chamfers',fill='white')
        comparison.save(args.out/'paving_comparison.png')
        report['rgb_comparison_mean_abs_difference']=float(np.abs(np.array(before).astype(float)-np.array(after)).mean())
        assert report['rgb_comparison_mean_abs_difference']>0
        report['result']='PASS_V5_NATIVE_GROUND_VIEWS'
    except BaseException as exc:report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if preview:preview.close()
        if app:app.close()
    return 0 if report['result']=='PASS_V5_NATIVE_GROUND_VIEWS' else 1


if __name__=='__main__':sys.exit(main())
