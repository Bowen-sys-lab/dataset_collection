"""Actual v5 opening/equipment/ground distance evidence from mounted cameras."""
import copy
import json
import math
import traceback
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def validate_distance_evidence(views):
    if len(views)!=9 or len({(v['role'],v['scale']) for v in views})!=9:
        raise ValueError('Expected exactly nine distinct role/scale views')
    for role in ['opening','equipment','ground']:
        selected={v['scale']:v for v in views if v['role']==role}
        if set(selected)!={'overall','mid','near'}:
            raise ValueError('Missing distance scales: '+role)
        distances=[]
        for scale in ['near','mid','overall']:
            v=selected[scale]
            if not v['native_capture'] or v['manual_success'] is not None:
                raise ValueError('Distance evidence must be a native diagnostic')
            if v['intrinsics_px']['rgb']!=v['intrinsics_px']['thermal']:
                raise ValueError('Mounted camera intrinsics differ')
            if not math.isfinite(v['pair_pose_max_error']) or v['pair_pose_max_error']>=1e-6 or not v['coverage_complete']:
                raise ValueError('Incomplete or misaligned native view')
            distance=v['distance_to_focus_center_m']
            if not math.isfinite(distance) or distance<=0:
                raise ValueError('Invalid actual distance')
            if scale!='overall' and v['focus_pixels']<=0:
                raise ValueError('Requested focus not observed: '+role+'/'+scale)
            if not v['assembly_pixels']:
                raise ValueError('Requested assembly not observed: '+role+'/'+scale)
            distances.append(distance)
        if not distances[0]<distances[1]<distances[2]:
            raise ValueError('Distances are not ordered: '+role)
    return {'roles':3,'views':len(views),'complete':True,
            'scope':'v5 baseline only; full v6 factory acceptance remains separate'}


def capture_distance_views(app,preview,config,out,base_state):
    import omni.timeline
    from pxr import Gf,UsdGeom
    from scripts.build_model02_v4_scene import install_cameras
    from scripts.drone_control_math import AttitudeConfig
    from scripts.validate_model02_v6 import diagnostic_eye
    from irsim_isaac.pipeline.gbuffer_isaac import geometry_planes

    out.mkdir(parents=True,exist_ok=False)
    report={'result':'FAIL','manual_success':None,'views':[]}
    try:
        manifest=json.loads((ROOT/'factory/model02_thermal_v4/isaac_manifest.json').read_text(encoding='utf-8'))
        rows=manifest['surfaces']
        registry=json.loads((ROOT/'factory/model02_thermal_v6/source_registry.json').read_text(encoding='utf-8'))
        sources={r['source_object']:r for r in registry['sources']}
        mission=json.loads((ROOT/'factory/model02_thermal_v4/mission_and_camera.json').read_text())
        stage=preview.stage
        drone=UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.prim_path))
        body=UsdGeom.Xformable(stage.GetPrimAtPath(config.drone.body_path))
        body_relative=body.ComputeLocalToWorldTransform(0)*drone.ComputeLocalToWorldTransform(0).GetInverse()
        timeline=omni.timeline.get_timeline_interface()
        pitch_limit=mission['camera_pitch_down_deg']+math.degrees(AttitudeConfig().max_angle)
        targets=[]
        for role,component in [('opening','WindowGlass'),('equipment','TankShell'),('ground','Landscape')]:
            candidates=[r for r in rows if r['component']==component and stage.GetPrimAtPath(r['usd_prim_path']).IsValid()]
            if role=='ground':
                point=np.asarray(mission['start_position_m'][:2])
                local=[r for r in candidates if 2.<=r['area_m2']<=8.]
                focus=min(local,key=lambda r:np.linalg.norm(np.asarray(r['bounds']).mean(axis=0)[:2]-point))
            else:focus=max(candidates,key=lambda r:r['area_m2'])
            targets.append((role,focus))
        index=0
        for role,focus in targets:
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(focus['usd_prim_path']))
            points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
            matrix=np.asarray(mesh.ComputeLocalToWorldTransform(0))
            points=(np.c_[points,np.ones(len(points))]@matrix)[:,:3]
            face=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())[:3]
            normal=np.cross(points[face[1]]-points[face[0]],points[face[2]]-points[face[0]])
            normal/=np.linalg.norm(normal)
            center=np.asarray(focus['bounds']).mean(axis=0)
            source_center=np.asarray(sources[focus['source_object']]['bounds_m']).mean(axis=0)
            assembly_paths={r['usd_prim_path'] for r in rows if r['source_object']==focus['source_object']}
            for scale,requested_distance in [('overall',70.),('mid',12.),('near',3.)]:
                eye=diagnostic_eye(center,normal,source_center,requested_distance,scale,role=='ground',pitch_limit)
                delta=center-eye
                yaw=math.degrees(math.atan2(delta[1],delta[0]))
                pitch=math.degrees(math.atan2(-delta[2],np.linalg.norm(delta[:2])))
                transform=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),yaw))
                offset=np.asarray((body_relative*transform).Transform(Gf.Vec3d(*mission['camera_offset_body_m'])))
                if eye[2]-offset[2]<config.drone.minimum_height:
                    eye=center+(eye-center)*((config.drone.minimum_height+offset[2]-center[2])/(eye[2]-center[2]))
                transform.SetTranslateOnly(Gf.Vec3d(*(eye-offset)))
                drone.MakeMatrixXform().Set(transform)
                install_cameras(stage,config.drone.body_path,mission['camera_offset_body_m'],pitch)
                timeline.set_current_time(0.);timeline.commit()
                preview.thermal.state=copy.deepcopy(base_state)
                for _ in range(8):app.update()
                rgb,thermal,meta=preview.capture(render=True,simulation_time=0.,frozen_state=True)
                raw=preview.thermal._reader.read();last=preview.thermal._last
                camera=UsdGeom.Camera(stage.GetPrimAtPath(preview.mount.rgb_path))
                actual_eye=np.asarray(camera.ComputeLocalToWorldTransform(0).ExtractTranslation())
                if not np.allclose(actual_eye,eye,atol=1e-4,rtol=0.):
                    raise ValueError('Mounted camera position mismatch: '+str((actual_eye,eye)))
                intrinsics={}
                for label,path in [('rgb',preview.mount.rgb_path),('thermal',preview.mount.thermal_path)]:
                    cam=UsdGeom.Camera(stage.GetPrimAtPath(path));w,h=rgb.shape[1],rgb.shape[0]
                    f=float(cam.GetFocalLengthAttr().Get());ha=float(cam.GetHorizontalApertureAttr().Get());va=float(cam.GetVerticalApertureAttr().Get())
                    intrinsics[label]={'fx':w*f/ha,'fy':h*f/va,'cx':w/2.,'cy':h/2.,'width':w,'height':h}
                focus_ids=[i for i,p in last.labels.items() if p==focus['usd_prim_path']]
                assembly_ids=[i for i,p in last.labels.items() if p in assembly_paths]
                mask=np.isin(last.instance_id,focus_ids)
                actual_distance=float(np.linalg.norm(center-actual_eye))
                measured=np.asarray(raw.distance_m)[mask]
                footprint_distance=float(np.median(measured)) if measured.size else actual_distance
                footprints={k:intrinsics['thermal']['fx']*size/footprint_distance
                    for k,size in {'frame_35mm':.035,'joint_8mm':.008,'relief_2mm':.002}.items()}
                entry={'role':role,'scale':scale,'focus_component':focus['component'],'focus_surface':focus['surface_id'],
                    'source_object':focus['source_object'],'focus_path':focus['usd_prim_path'],
                    'requested_distance_m':requested_distance,'distance_to_focus_center_m':actual_distance,
                    'focus_pixels':int(mask.sum()),'assembly_pixels':int(np.isin(last.instance_id,assembly_ids).sum()),
                    'focus_range_m':{'min':float(measured.min()),'median':float(np.median(measured)),'max':float(measured.max())} if measured.size else None,
                    'intrinsics_px':intrinsics,'camera_eye_m':actual_eye.tolist(),'view_pitch_down_deg':pitch,
                    'equivalent_body_pitch_deg':pitch-mission['camera_pitch_down_deg'],
                    'resolution':[rgb.shape[1],rgb.shape[0]],'native_capture':True,'manual_success':None,
                    'coverage_complete':meta['thermal_coverage']['complete_for_this_view'],
                    'pair_pose_max_error':meta['pair_pose_max_error'],'feature_projected_pixels_estimate':footprints,
                    'feature_resolution':{k:'below_one_pixel' if v<1. else 'one_pixel_or_more' for k,v in footprints.items()},
                    'footprint_scope':'Pinhole estimate for stated physical dimensions; not proof that a feature is modeled or resolved'}
                focus_diagonal=float(np.linalg.norm(np.ptp(points,axis=0)))
                entry['focus_projected_bounding_diagonal_px_estimate']=intrinsics['thermal']['fx']*focus_diagonal/actual_distance
                entry['focus_observation']='observed' if entry['focus_pixels'] else 'not_observed'
                entry['focus_visibility_limit']=None if entry['focus_pixels'] else ('subpixel_or_occluded' if entry['focus_projected_bounding_diagonal_px_estimate']<1. else 'occluded_or_unresolved')
                assert abs(entry['equivalent_body_pitch_deg'])<=math.degrees(AttitudeConfig().max_angle)+1e-6
                assert np.isfinite(thermal.radiance).all() and np.isfinite(thermal.apparent_t).all()
                geometry=geometry_planes(raw,up_axis=preview.thermal._up_axis or 'Y',position_frame=preview.thermal.position_frame,
                    camera_position=preview.thermal._camera_position,camera_to_world=preview.thermal._camera_to_world)
                np.savez_compressed(out/f'frame_{index:06d}_aovs.npz',normal=raw.normal,distance_m=raw.distance_m,
                    instance_id=last.instance_id,material_id=last.material_id,temperature_k=last.planes['temperature_k'],
                    normal_dot_up=geometry.normal_dot_up,normal_dot_view=last.planes['normal_dot_view'],
                    sky_view_factor=last.planes['sky_view_factor'])
                preview.save_frame(out,index,rgb,thermal,{**meta,**entry})
                (out/f'labels_{index:06d}.json').write_text(json.dumps(last.labels,indent=2),encoding='utf-8')
                report['views'].append(entry);index+=1
                print('[IR distance probe]',role,scale,'focus_pixels',entry['focus_pixels'],'assembly_pixels',entry['assembly_pixels'],'range',actual_distance,flush=True)
        report['coverage']=validate_distance_evidence(report['views'])
        report['result']='PASS_NATIVE_DISTANCE_PROBE'
        report['resolved_aovs']=dict(preview.thermal._reader.resolved)
        report['unobserved_focus_views']=[{'role':v['role'],'scale':v['scale'],'reason':v['focus_visibility_limit']} for v in report['views'] if not v['focus_pixels']]
    except BaseException as exc:
        report.update(result='FAIL',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    import argparse
    import sys
    sys.path.insert(0,str(ROOT))
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fast',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    app=preview=None
    report={'result':'FAIL','manual_success':None}
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'renderer':'PathTracing','multi_gpu':False,'samples_per_pixel_per_frame':16})
        from pxr import UsdPhysics
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        import omni.timeline
        import carb.settings
        config=load_scene_config(ROOT/'configs/model02_v5_tank_scene.yaml')
        preview,_=open_preview(app,scene_config=config,width=160,height=128,thermal_width=160,thermal_height=128,
            fast=args.fast,start_controller=False,show_ui=False)
        stage=preview.stage;stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        report=capture_distance_views(app,preview,config,args.out,copy.deepcopy(preview.thermal.state))
        settings=carb.settings.get_settings()
        report.update(renderer=preview.metadata['renderer'],preview_fast=args.fast,
            renderer_settings={p:settings.get(p) for p in ['/rtx/rendermode','/rtx/pathtracing/spp','/rtx/pathtracing/totalSpp']},
            fast_scope='PreviewSky approximation; fast does not select a different renderer')
    except BaseException as exc:
        if (args.out/'summary.json').exists():report=json.loads((args.out/'summary.json').read_text())
        report.update(result='FAIL',error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if preview:preview.close()
        if app:app.close()
    return int(report['result']!='PASS_NATIVE_DISTANCE_PROBE')


if __name__=='__main__':raise SystemExit(main())
