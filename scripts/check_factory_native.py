"""Native spawn and actual hotspot-view checks; synthetic evidence, never teacher data."""
from pathlib import Path
import json
import math
import sys
import time
import traceback
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.factory_collection_config import PLAN, READINESS, resolve_selection, digest


def main():
    source=json.loads(READINESS.read_text(encoding='utf-8'))
    plan=yaml.safe_load(PLAN.read_text(encoding='utf-8'))
    out=READINESS.parent/'native-views'
    out.mkdir(exist_ok=False)
    report={'trajectory_source':'synthetic_geometry_probe','manual_confirmation_created':False,
            'source_inputs':source.get('source_inputs',source['inputs']),
            'starts':{},'targets':{},'evidence_hashes':{},'result':'FAIL'}
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':'PathTracing','multi_gpu':False,'samples_per_pixel_per_frame':1})
    preview=controller=None
    try:
        import omni.timeline
        from pxr import Usd, UsdGeom, UsdPhysics
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.scene_config import load_scene_config
        from scripts.drone_control_math import orientation_wxyz_from_rpy
        from scripts.trajectory_recorder import orientation_wxyz_from_world_transform
        from scripts.drone_control_math import rpy_from_orientation_wxyz
        from scripts.replay_pose import FrozenPoseRestorer
        from scripts.replay_factory_inspection import hotspot_evidence
        config=load_scene_config(ROOT/'configs/model02_v7m_bc_pilot_scene.yaml')
        preview,controller=open_preview(app,scene_config=config,show_ui=False)
        timeline=omni.timeline.get_timeline_interface()
        for group in plan['start_groups']:
            controller.initial_pose=(tuple(group['position_m']),orientation_wxyz_from_rpy(0.,0.,math.radians(group['yaw_degrees'])))
            controller._ready=False
            timeline.play()
            deadline=time.perf_counter()+30.
            while not controller._ready and time.perf_counter()<deadline: app.update()
            if not controller._ready: raise RuntimeError('Physics initialization timed out')
            for _ in range(3): app.update()
            matrix=np.asarray(UsdGeom.Xformable(preview.stage.GetPrimAtPath(config.drone.body_path)).ComputeLocalToWorldTransform(0))
            actual=matrix[3,:3]; yaw=rpy_from_orientation_wxyz(orientation_wxyz_from_world_transform(matrix))[2]
            yaw_error=abs(math.atan2(math.sin(yaw-math.radians(group['yaw_degrees'])),math.cos(yaw-math.radians(group['yaw_degrees']))))
            error=float(np.linalg.norm(actual-group['position_m']))
            report['starts'][group['group']]={'position_m':actual.tolist(),'position_error_m':error,
                'yaw_error_degrees':math.degrees(yaw_error),'passed':error<=.25 and yaw_error<=math.radians(1.)}
            print(f'Native spawn {group["group"]}: {report["starts"][group["group"]]}',flush=True)
        controller.stop(); controller=None
        from scripts.start_crazyflie_dual_camera_view import _stop_timeline
        _stop_timeline()
        app.update()
        stage=preview.stage;stage.SetEditTarget(stage.GetSessionLayer())
        for prim in Usd.PrimRange(stage.GetPrimAtPath(config.drone.prim_path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI): UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint): UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        timeline.set_auto_update(False)
        restorer=FrozenPoseRestorer(stage,config)
        for index,target in enumerate(('tank','pipe')):
            inspection=resolve_selection(target,1,'A',require_ready=False)['task']['environment_evidence']['factory_inspection']
            position=inspection['observation_position']; yaw=inspection['observation_yaw_rad']
            restorer.restore({'position':position,'orientation':orientation_wxyz_from_rpy(0.,0.,yaw)})
            for _ in range(2):
                rgb,thermal,meta=preview.capture(render=True,simulation_time=float(timeline.get_current_time()),frozen_state=True)
            preview.save_frame(out,index,rgb,thermal,meta)
            last=preview.thermal._last
            evidence=hotspot_evidence(last.instance_id,{str(k):v for k,v in last.labels.items()},
                inspection['hotspot_prim_path'],20,target_paths=inspection.get('hotspot_prim_paths'))
            path=out/f'{target}_instance_id.npy';np.save(path,np.asarray(last.instance_id))
            evidence['passed']=bool(evidence['hotspot_visible'] and meta['thermal_coverage']['complete_for_this_view'])
            evidence['resolution']=[160,128]
            evidence['camera_world_matrix']=meta['camera_world_matrix']
            report['targets'][target]=evidence
            print(f'Native view {target}: pixels={evidence["hotspot_visible_pixels"]}, passed={evidence["passed"]}',flush=True)
        for path in out.iterdir():
            if path.is_file():report['evidence_hashes'][path.relative_to(ROOT).as_posix()]=digest(path)
        report['result']='PASS' if all(r['passed'] for r in [*report['starts'].values(),*report['targets'].values()]) else 'FAIL'
    except Exception:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:
        (READINESS.parent/'native_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if controller:controller.stop()
        if preview:preview.close()
        app.close()


if __name__=='__main__':main()
