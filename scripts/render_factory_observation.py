"""Render per-sample visibility evidence for a recorded factory observation.

Only distance-qualified stationary samples are captured. Renderer instance IDs
establish that the actual hotspot surface is visible; this is not automatic
fault recognition and never manufactures a pilot T confirmation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from scripts.scene_config import load_scene_config
    from scripts.trajectory_recorder import read_trajectory
    from scripts.evaluate_factory_inspection import point_to_mesh_distances
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/factory_inspection_scene.yaml')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    config=load_scene_config(args.config)
    metadata=json.loads((args.run/'run_metadata.json').read_text(encoding='utf8'))
    trajectory=args.run/'trajectory.jsonl'
    if metadata['scene_config_sha256']!=config.sha256 or metadata['stage_sha256']!=hashlib.sha256(config.scene.stage.read_bytes()).hexdigest():
        raise RuntimeError('Recorded scene differs from current configuration')
    task=metadata['task'];data=task['environment_evidence']['factory_inspection']
    records=read_trajectory(trajectory)
    geom=np.load(ROOT/data['geometry_file'],allow_pickle=False)
    positions=np.array([r['position'] for r in records])
    distances=point_to_mesh_distances(positions,geom['equipment_vertices'],geom['equipment_triangles'],body_radius_m=.2)
    selected=[r for r,d in zip(records,distances) if 6<=d<=8 and
              np.linalg.norm(r.get('linear_velocity',[999,999,999]))<=.3]
    if not selected:raise RuntimeError('No stationary observation samples')
    args.out.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':'PathTracing','multi_gpu':False,
                       'samples_per_pixel_per_frame':1})
    preview=None;rows=[];report={'result':'FAIL'}
    try:
        import omni.timeline
        from pxr import UsdGeom,UsdPhysics
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.replay_pose import FrozenPoseRestorer
        preview,_=open_preview(app,scene_config=config,start_controller=False,show_ui=False)
        stage=preview.stage;stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        restorer=FrozenPoseRestorer(stage,config)
        trajectory_sha=hashlib.sha256(trajectory.read_bytes()).hexdigest()
        for index,record in enumerate(selected):
            time=float(record['simulation_time']);timeline.set_current_time(time);timeline.commit()
            restorer.restore(record);app.update()
            rgb,thermal,meta=preview.capture(render=True,simulation_time=time,frozen_state=True)
            preview.save_frame(args.out,index,rgb,thermal,meta)
            last=preview.thermal._last
            ids=[int(key) for key,path in last.labels.items() if path==task['target_prim_path']]
            pixels=int(np.isin(last.instance_id,ids).sum())
            matrix=np.array(meta['camera_world_matrix'])
            pose_error=float(np.max(np.abs(matrix-np.array(record['rgb_camera_world_matrix']))))
            if pose_error>1e-5:raise RuntimeError('Recorded evidence camera differs')
            visible=pixels>=20 and meta['thermal_coverage']['complete_for_this_view']
            rows.append({'frame_id':record['frame_id'],'simulation_time':time,
                'trajectory_sha256':trajectory_sha,'scene_config_sha256':config.sha256,
                'hotspot_prim_path':task['target_prim_path'],'hotspot_visible':visible,
                'unobstructed':visible,'camera_world_matrix':meta['camera_world_matrix'],
                'evidence_image':f'frame_{index:06d}_rgb.png','thermal_evidence_image':f'frame_{index:06d}_thermal.png',
                'hotspot_visible_pixels':pixels,'method':'actual renderer surface IDs under aligned RGB/IR cameras',
                'strict_sync_verified':False})
            if index%10==0:print(f'[factory evidence] {index+1}/{len(selected)}',flush=True)
        report={'result':'PASS' if all(r['hotspot_visible'] for r in rows) else 'FAIL',
            'frames':len(rows),'first_time_s':rows[0]['simulation_time'],'last_time_s':rows[-1]['simulation_time'],
            'trajectory_sha256':trajectory_sha,'minimum_hotspot_pixels':min(r['hotspot_visible_pixels'] for r in rows),
            'qualification':metadata.get('trajectory_source'),'manual_confirmation_created':False,
            'strict_sync_verified':False}
    except Exception:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:
        (args.out/'visual_evidence.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        if preview:preview.close()
        app.close()
    return 0 if report['result']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
