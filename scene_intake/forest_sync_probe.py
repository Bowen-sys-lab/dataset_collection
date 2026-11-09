"""Independent moving-marker check of the forest frozen dynamic pair adapter.

Diagnostic data only. Does not change the shared project's sync evidence or
pretend the runtime itself supplies independently verified AOV timestamps.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

def probe_thermal_config(path):
    import yaml
    raw=yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    raw['targets']['sync_hot']=dict(mode='fixed',temperature_c=80.)
    raw['targets']['sync_cold']=dict(mode='fixed',temperature_c=10.)
    for name,target in [('Marker','sync_hot'),('Backdrop','sync_cold')]:
        raw['bindings'].append(dict(name='forest_probe_'+name,prim_path='/World/ForestSync'+name,
                                    include_children=True,target=target,material='concrete'))
    return raw

def probe_marker_positions(count,width,height):
    """Keep the marker above foreground terrain; do not relax pixel tolerances."""
    if not 4<=count<=16:raise ValueError('probe supports 4 through 16 independent states')
    return [[width*x,height*y] for y in (.20,.40,.30,.16) for x in (.2,.4,.6,.8)][:count]


def evaluate_marker(rgb,apparent_t,truth_t,expected_rgb,expected_ir):
    from scripts.sync_validation_support import rgb_marker_mask,largest_component,compare_paired_masks
    color=rgb_marker_mask(rgb)
    geometry=largest_component(np.abs(truth_t-353.15)<.1)
    thermal=largest_component((apparent_t>323.15)&(apparent_t<403.15))
    g=compare_paired_masks(color,geometry,expected_rgb,expected_ir)
    t=compare_paired_masks(color,thermal,expected_rgb,expected_ir)
    return dict(result='PASS' if g['pass'] and t['pass'] else 'FAIL',
                **{'pass':bool(g['pass'] and t['pass'])},geometry=g,temperature=t)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=8)
    ap.add_argument('--thermal-width',type=int,choices=[320,640],default=640)
    args=ap.parse_args(argv)
    if not 4<=args.frames<=16:raise ValueError('4 through 16 independent marker states required')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))
    sys.path.insert(0,str(ROOT/'forest_fire_scene_local/scripts'))
    from isaacsim import SimulationApp
    assets=ROOT/'validation/isaac_local_assets';(assets/'Isaac').mkdir(parents=True,exist_ok=True)
    app=SimulationApp(dict(headless=True,renderer='PathTracing',multi_gpu=False,samples_per_pixel_per_frame=8,
                          extra_args=['--/persistent/isaac/asset_root/default='+str(assets).replace('\\','/')]))
    preview=None;report=dict(result='INCOMPLETE',scope='forest dynamic pair adapter, diagnostic marker at fixed exposure +2 EV',
                            frames=[],data_usage='diagnostic only; never dataset frames',
                            rgb_resolution=[640,512],thermal_resolution=[args.thermal_width,args.thermal_width*4//5],
                            strict_per_frame_pixel_sync_verified=False)
    try:
        import yaml
        import carb
        import omni.usd,omni.timeline
        from pxr import UsdGeom,Gf
        from dataclasses import replace
        from PIL import Image
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.sync_validation_support import unique_marker_positions,centroid,rgb_marker_mask,registered_overlay,largest_component
        from tests.isaac_dual_camera_sync import sheet
        from scene_intake.forest_collection import capture_dynamic_pair,set_world_camera,TARGET_POSITION
        from scene_intake.merge_replay_passes import sha256
        from fire_thermal_proxy import thermal_flame_surfaces
        base=load_scene_config(ROOT/'configs/forest_fire_scene.yaml',include_task=False)
        raw=probe_thermal_config(base.sensors.thermal_config)
        thermal_path=args.out/'probe_thermal.yaml';thermal_path.write_text(yaml.safe_dump(raw),encoding='utf-8')
        cfg=replace(base,sensors=replace(base.sensors,thermal_config=thermal_path))
        ctx=omni.usd.get_context();ctx.open_stage(str(cfg.scene.stage))
        for _ in range(60):app.update()
        stage=ctx.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
        marker=sheet(stage,'/World/ForestSyncMarker',.6,.45,(0.,1.,0.))
        backdrop=sheet(stage,'/World/ForestSyncBackdrop',20.,20.,(.01,.01,.01))
        preview,_=open_preview(app,scene_config=cfg,width=640,height=512,thermal_width=args.thermal_width,
                              thermal_height=args.thermal_width*4//5,fast=False,start_controller=False,show_ui=False,
                              frozen_subframes=4)
        stage.SetEditTarget(stage.GetSessionLayer())
        settings=carb.settings.get_settings()
        for k,v in {'/rtx/post/tonemap/op':6,'/rtx/post/tonemap/filmIso':400.,
                    '/rtx/post/tonemap/cameraShutter':50.,'/rtx/post/tonemap/fNumber':5.,
                    '/rtx/post/histogram/enabled':False}.items():settings.set(k,v)
        eye=Gf.Vec3d(TARGET_POSITION[0],TARGET_POSITION[1]+15.,TARGET_POSITION[2]+1.5)
        world=np.asarray(Gf.Matrix4d().SetLookAt(eye,Gf.Vec3d(*TARGET_POSITION),Gf.Vec3d(0,0,1)).GetInverse())
        for path in (preview.mount.rgb_path,preview.mount.thermal_path):set_world_camera(stage,path,world)
        local=np.eye(4);local[3,:3]=[0,0,-5]
        set_world_camera(stage,str(backdrop.GetPath()),local@world)
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        positions=probe_marker_positions(args.frames,args.thermal_width,args.thermal_width*4//5)
        kr=preview.metadata['rgb_intrinsics_px'];ki=preview.metadata['thermal_intrinsics_px']
        expected_ir=positions
        expected_rgb=[[(u+.5)*640/args.thermal_width-.5,(v+.5)*512/(args.thermal_width*4//5)-.5] for u,v in positions]
        observed=[]
        for index,(u,v) in enumerate(positions):
            seconds=index/24.;timeline.set_current_time(seconds);timeline.commit()
            local=np.eye(4);local[3,:3]=[(u-ki['cx_px'])*4/ki['fx_px'],(ki['cy_px']-v)*4/ki['fy_px'],-4.]
            set_world_camera(stage,str(marker.GetPath()),local@world);app.update()
            rgb,thermal,meta,truth,_=capture_dynamic_pair(preview,seconds,index,thermal_flame_surfaces)
            check=evaluate_marker(rgb,thermal.apparent_t,truth['temperature_k'],expected_rgb[index],expected_ir[index])
            color=rgb_marker_mask(rgb);geom=largest_component(np.abs(truth['temperature_k']-353.15)<.1)
            observed.append(centroid(geom))
            check.update(frame_id=index,time_code=index,pair_validation=meta['pair_validation'])
            report['frames'].append(check)
            Image.fromarray(registered_overlay(rgb,color,geom)).save(args.out/f'marker_{index:02d}_overlay.png')
            print('[forest-sync]',index,check['result'],flush=True)
        # Non-repeating states independently identify the requested frame, not a previous AOV.
        history=[None if xy is None else int(np.argmin(np.linalg.norm(np.asarray(expected_ir)-xy,axis=1))) for xy in observed]
        report.update(result='PASS' if all(r['pass'] for r in report['frames']) and history==list(range(args.frames)) else 'FAIL',
                      marker_history_verified=history==list(range(args.frames)),observed_marker_indices=history,
                      strict_per_frame_pixel_sync_verified=False,
                      independent_marker_probe_passed=all(r['pass'] for r in report['frames']),
                      source_stage_sha256=sha256(base.scene.stage),scene_config_sha256=base.sha256,
                      collector_sha256=sha256(ROOT/'scene_intake/forest_collection.py'),
                      source_flame_proxy_sha256=sha256(ROOT/'forest_fire_scene_local/scripts/fire_thermal_proxy.py'),
                      note='Independent pixel-marker evidence for this adapter/config/resolution; does not turn all future production frames into individually pixel-verified captures.')
        return 0 if report['result']=='PASS' else 2
    except Exception:
        import traceback
        report.update(result='FAIL',error=traceback.format_exc());raise
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if preview:preview.close()
        app.close()

if __name__=='__main__':raise SystemExit(main())
