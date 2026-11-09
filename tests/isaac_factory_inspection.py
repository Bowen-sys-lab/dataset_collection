"""Factory-only asset/physics/RGB/IR acceptance and synthetic flight recording."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.scene_config import load_scene_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/factory_inspection_scene.yaml')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--width',type=int,default=160)
    parser.add_argument('--height',type=int,default=128)
    parser.add_argument('--thermal-width',type=int)
    parser.add_argument('--thermal-height',type=int)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--record-reference',action='store_true')
    args=parser.parse_args()
    config=load_scene_config(args.config)
    args.out.mkdir(parents=True,exist_ok=False)
    report={'result':'FAIL','strict_sync_verified':False,'data_usage':'synthetic development validation'}
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':'PathTracing','multi_gpu':False,
                       'samples_per_pixel_per_frame':1})
    preview=controller=None
    status=1
    try:
        import numpy as np
        import omni.usd
        import omni.timeline
        from pxr import Gf,Usd,UsdGeom,UsdPhysics,UsdUtils,Sdf
        from scripts.start_crazyflie_dual_camera_view import open_preview,preview_motion_config
        from scripts.evaluate_factory_inspection import point_to_mesh_distances
        context=omni.usd.get_context()
        if not context.open_stage(str(config.scene.stage)):
            raise RuntimeError('Could not open factory inspection stage')
        for _ in range(120):app.update()
        stage=context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        data=config.task['environment_evidence']['factory_inspection']
        required=[config.drone.prim_path,config.drone.body_path,config.drone.camera_path,
                  config.task['target_prim_path'],data['normal_surface_prim_path']]
        for path in required:
            prim=stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive():raise RuntimeError('Missing active prim: '+path)
        body=stage.GetPrimAtPath(config.drone.body_path)
        drone=stage.GetPrimAtPath(config.drone.prim_path)
        if not drone.HasAPI(UsdPhysics.ArticulationRootAPI):raise RuntimeError('Crazyflie articulation missing')
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):raise RuntimeError('Crazyflie body rigid API missing')
        physics=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(physics)!=1:raise RuntimeError(f'Expected one physics scene: {physics}')
        report['preflight']={'required_paths':required,'physics_scenes':physics,
            'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'up_axis':UsdGeom.GetStageUpAxis(stage)}
        geometry=np.load(ROOT/'factory/factory_inspection_geometry.npz')
        distances=point_to_mesh_distances(np.array([data['observation_position']]),
            geometry['equipment_vertices'],geometry['equipment_triangles'],body_radius_m=.2)
        report['observation_equipment_clearance_m']=float(distances[0])
        if not 6<=distances[0]<=8:raise RuntimeError('Observation is outside 6-8m clearance')
        if args.preflight_only:
            report['result']='PASS';status=0
            return status
        preview,controller=open_preview(app,scene_config=config,width=args.width,height=args.height,
            thermal_width=args.thermal_width,thermal_height=args.thermal_height,
            motion_config=preview_motion_config(1,1.5),show_ui=False)
        timeline=omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(180):
            app.update()
            if controller._ready:break
        if not controller._ready:raise RuntimeError('Controller did not initialize: '+str(controller.diagnostics()))
        before=controller.position
        controller.position=(before[0]+.2,before[1],before[2]+.1)
        controller.yaw+=.1
        for _ in range(12):app.update()
        report['controller_smoke']={'ready':controller._ready,'before':list(before),
            'after':list(controller.position),'diagnostics':controller.diagnostics()}
        timeline.stop();app.update()
        controller.stop();controller=None
        timeline.pause();timeline.set_auto_update(False)
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.IsA(UsdPhysics.Joint):UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
        from scripts.replay_pose import FrozenPoseRestorer
        from scripts.drone_control_math import orientation_wxyz_from_rpy
        restorer=FrozenPoseRestorer(stage,config)
        frames=[]
        for i,(name,position,yaw) in enumerate([
            ('start',config.task['start_position'],math.atan2(23,22)),
            ('observation',data['observation_position'],data['observation_yaw_rad'])]):
            timeline.set_current_time(float(i));timeline.commit()
            record={'position':position,'orientation':orientation_wxyz_from_rpy(0,0,yaw)}
            restorer.restore(record);app.update()
            rgb,thermal,meta=preview.capture(render=True,simulation_time=float(i),frozen_state=True)
            preview.save_frame(args.out,i,rgb,thermal,meta)
            last=preview.thermal._last
            hot_ids=[int(k) for k,v in last.labels.items() if v==config.task['target_prim_path']]
            normal_ids=[int(k) for k,v in last.labels.items() if v==data['normal_surface_prim_path']]
            hot=np.isin(last.instance_id,hot_ids);normal=np.isin(last.instance_id,normal_ids)
            planes=preview.thermal.last_frame.planes
            np.save(args.out/f'{name}_instance_id.npy',last.instance_id)
            np.save(args.out/f'{name}_hotspot_mask.npy',hot)
            temperature=np.asarray(planes['temperature_k'])
            np.save(args.out/f'{name}_prescribed_temperature_k.npy',temperature)
            frame={'name':name,'hotspot_pixels':int(hot.sum()),'normal_pixels':int(normal.sum()),
                'hotspot_apparent_median_k':float(np.median(thermal.apparent_t[hot])) if hot.any() else None,
                'normal_apparent_median_k':float(np.median(thermal.apparent_t[normal])) if normal.any() else None,
                'hotspot_temperature_k':float(np.median(temperature[hot])) if hot.any() else None,
                'pair_pose_max_error':meta['pair_pose_max_error'],'thermal_coverage':meta['thermal_coverage']}
            frames.append(frame)
            if not meta['thermal_coverage']['complete_for_this_view']:raise RuntimeError('Unmapped factory pixels')
            if not np.isfinite(thermal.apparent_t).all():raise RuntimeError('Non-finite apparent temperature')
            if name=='observation':
                if frame['hotspot_pixels']<20:raise RuntimeError('Hotspot covers fewer than 20 pixels')
                if not normal.any():raise RuntimeError('No normal equipment visible')
                if frame['hotspot_apparent_median_k']-frame['normal_apparent_median_k']<10:raise RuntimeError('Insufficient thermal contrast')
        report['frames']=frames
        if args.record_reference:
            from scripts.trajectory_recorder import TrajectoryRecorder,trajectory_state_from_world_transform
            recorder=TrajectoryRecorder.create(args.out,{
                'scene_config_path':str(config.source_path),'scene_config_sha256':config.sha256,
                'stage_path':str(config.scene.stage),'stage_sha256':hashlib.sha256(config.scene.stage.read_bytes()).hexdigest(),
                'task':config.task,'trajectory_source':'synthetic_reference_route',
                'task_result':{'status':'unjudged'},'success':None,'sample_period_s':.1,
                'data_usage':'development geometry trajectory; not human teacher data'},run_id='reference_run')
            t=0.;fid=0
            waypoints=np.asarray(data['reference_waypoints'],float)
            for segment,(a,b) in enumerate(zip(waypoints,waypoints[1:])):
                length=float(np.linalg.norm(b-a));duration=max(float(np.linalg.norm((b-a)[:2]))/1.5,abs(float(b[2]-a[2]))/.5);steps=int(math.ceil(duration/.1))
                yaw=math.atan2(b[1]-a[1],b[0]-a[0])
                for k in range(steps+1):
                    if segment and k==0:continue
                    p=a+(b-a)*(k/steps)
                    row={'position':p.tolist(),'orientation':orientation_wxyz_from_rpy(0,0,yaw),
                         'frame_id':fid,'simulation_time':t+k*duration/steps,
                         'linear_velocity':((b-a)/duration).tolist()}
                    restorer.restore(row)
                    bodyworld=UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0)
                    row=trajectory_state_from_world_transform(row,bodyworld)
                    row['rgb_camera_world_matrix']=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(preview.mount.rgb_path)).ComputeLocalToWorldTransform(0)).tolist()
                    recorder.append(row);fid+=1
                t+=duration
                if segment==2:
                    for k in range(1,66):
                        row={'position':b.tolist(),'orientation':orientation_wxyz_from_rpy(0,0,data['observation_yaw_rad']),
                             'frame_id':fid,'simulation_time':t+k*.1,'linear_velocity':[0,0,0]}
                        restorer.restore(row)
                        row=trajectory_state_from_world_transform(row,UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0))
                        row['rgb_camera_world_matrix']=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(preview.mount.rgb_path)).ComputeLocalToWorldTransform(0)).tolist()
                        recorder.append(row);fid+=1
                    t+=6.5
            report['reference_trajectory']=str(recorder.close())
        report['result']='PASS';status=0
    except Exception:
        report['error']=traceback.format_exc()
        (args.out/'failure.txt').write_text(report['error'],encoding='utf8')
        print(report['error'],flush=True)
    finally:
        if controller:controller.stop()
        if preview:preview.close()
        (args.out/'summary.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf8')
        print(json.dumps(report,default=str),flush=True)
        app.close()
    return status


if __name__=='__main__':raise SystemExit(main())
