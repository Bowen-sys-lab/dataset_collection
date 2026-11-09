"""Matched native scene/preview performance evidence at the saved mounted-camera pose."""
import argparse,json,statistics,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--fast',action='store_true');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);app=preview=None;report={'result':'FAIL','manual_success':None}
    try:
        import psutil
        from isaacsim import SimulationApp
        started=time.perf_counter();app=SimulationApp({'headless':True,'renderer':'PathTracing','samples_per_pixel_per_frame':16,'multi_gpu':False});report['app_startup_seconds']=time.perf_counter()-started
        import omni.timeline
        from pxr import UsdGeom,UsdPhysics
        import carb.settings
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        config=load_scene_config(args.config);timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        started=time.perf_counter();preview,_=open_preview(app,scene_config=config,width=160,height=128,thermal_width=160,thermal_height=128,fast=args.fast,start_controller=False,show_ui=False)
        report['stage_sensor_startup_seconds']=time.perf_counter()-started
        for p in preview.stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.IsA(UsdPhysics.Joint):UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
        started=time.perf_counter()
        for _ in range(6):app.update()
        report['warmup_seconds']=time.perf_counter()-started;durations=[]
        for index in range(3):
            started=time.perf_counter();rgb,thermal,meta=preview.capture(render=True,simulation_time=0.,frozen_state=True);durations.append(time.perf_counter()-started)
            assert meta['pair_pose_max_error']<1e-6 and meta['thermal_coverage']['complete_for_this_view'] and np.isfinite(thermal.apparent_t).all()
            if index==2:preview.save_frame(args.out,0,rgb,thermal,meta)
        live_times=[]
        for _ in range(3):
            started=time.perf_counter();app.update();preview.capture(render=False,simulation_time=0.,publish=False);live_times.append(time.perf_counter()-started)
        camera=UsdGeom.Camera(preview.stage.GetPrimAtPath(preview.mount.rgb_path));last=preview.thermal._last
        self_ids=[i for i,p in last.labels.items() if p.startswith(config.drone.prim_path+'/')];assert not np.isin(last.instance_id,self_ids).any()
        weather=preview.thermal.scene.weather.at(preview.thermal.scene.t0_s);audit=preview.thermal.thermal_config_report
        assert not audit['fallback_paths'] and not audit['unmatched_bindings']
        report.update(result='PASS_MATCHED_NATIVE_PERFORMANCE_CAPTURE',scene=str(config.scene.stage),resolution=[160,128],renderer=preview.metadata['renderer'],requested_pathtracing_spp=16,
            camera_world_matrix=meta['camera_world_matrix'],focal_length_mm=camera.GetFocalLengthAttr().Get(),horizontal_aperture_mm=camera.GetHorizontalApertureAttr().Get(),
            camera_pair_pose_max_error=meta['pair_pose_max_error'],capture_seconds=durations,median_capture_seconds=statistics.median(durations),
            process_memory=psutil.Process().memory_info()._asdict(),system_memory=psutil.virtual_memory()._asdict(),explicit_paths=len(audit['assignments']),
            live_update_capture_seconds=live_times,median_live_update_capture_seconds=statistics.median(live_times),live_sync_scope='Runtime live preview timing; strict live pixel-buffer synchronization is not claimed',
            air_temperature_k=weather.t_air_k,wind_m_s=weather.wind_speed_m_s,thermal_self_pixels=0,lighting=preview.metadata['authored_lighting'])
        settings=carb.settings.get_settings();report['renderer_settings']={p:settings.get(p) for p in ['/renderer/active','/rtx/pathtracing/spp','/rtx/pathtracing/totalSpp']}
    except BaseException as exc:report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if preview:preview.close()
        if app:app.close()
    return int(report['result']!='PASS_MATCHED_NATIVE_PERFORMANCE_CAPTURE')


if __name__=='__main__':sys.exit(main())
