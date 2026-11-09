"""Forest-only paired capture. Imports do not start Isaac Sim.

Uses the existing preview and file schema; edits are session-only. RGB is read
before the current-time opaque IR surrogate, at one frozen camera pose/time.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.collection_capture_defaults import RGB_SIZE, THERMAL_SIZE
from scene_intake.merge_replay_passes import quality_record,validate_pair,sha256
TARGET_PATH='/World/Site/Target_BurningBarrel_001'
TARGET_POSITION=[55.,6.86424822537128,15.314128450870785]
# 默认值只为兼容旧调用；实际以 <scene>_target.yaml 为准（见 load_target_config）。
DEFAULT_TARGET=dict(prim_path=TARGET_PATH,position_m=TARGET_POSITION,label='barrel',
                    prescribed_temperature_c=110.,infrared_material='rusted_steel',
                    dataset_scene='forest_fire')

def target_config_path(scene_config):
    """同名规则：foo_scene.yaml -> foo_target.yaml；其它情况 -> <stem>_target.yaml。"""
    scene_config=Path(scene_config);stem=scene_config.stem
    name=(stem[:-6] if stem.endswith('_scene') else stem)+'_target.yaml'
    return scene_config.with_name(name)

def load_target_config(scene_config,override=None):
    """场景身份（目标 prim、锚点、数据集名）从配置读；缺省回落到内置常量。"""
    import yaml
    path=Path(override) if override else target_config_path(scene_config)
    target=dict(DEFAULT_TARGET);target['config_path']=None
    if path.is_file():
        data=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        block=data.get('target') or {}
        unknown=set(block)-set(DEFAULT_TARGET)-{'verification_profile','expected_temperatures_c'}
        if unknown:raise ValueError(f'{path}: unknown target fields {sorted(unknown)}')
        target.update(block);target['config_path']=str(path)
    elif override:raise FileNotFoundError(f'target config not found: {path}')
    return target

PROFILES={
 'baseline':dict(thermal_resolution=list(THERMAL_SIZE),rgb_ev=0.),
 'readable':dict(thermal_resolution=list(THERMAL_SIZE),rgb_ev=2.),
 'native-readable':dict(thermal_resolution=list(THERMAL_SIZE),rgb_ev=2.),
}

def capture_profile(name, *, recorded_task=None):
    from scripts.forest_collection_config import is_two_target

    profile = deepcopy(PROFILES[name])
    profile.update(name=name, rgb_resolution=list(RGB_SIZE),
                   thermal_resolution=list(THERMAL_SIZE),
                   scene_config=str(ROOT/'configs/forest_fire_scene.yaml'),
                   sensor_range='original_200c_transfer')
    episode = ((recorded_task or {}).get('environment_evidence') or {}).get('collection_episode') or {}
    if name == 'native-readable' and is_two_target(episode):
        profile.update(name='factory-sized-readable', rgb_resolution=list(RGB_SIZE), thermal_resolution=list(THERMAL_SIZE))
    return profile

def animation_time_code(seconds,fps,start,end):
    if not all(math.isfinite(float(x)) for x in (seconds,fps,start,end)) or seconds<0 or fps<=0 or end<=start:
        raise ValueError('invalid animation time/range')
    return start+(seconds*fps)%(end-start)

def write_manifest(directory,input_files):
    directory=Path(directory)
    path=directory/'collection_manifest.json'
    if path.exists():raise FileExistsError('capture is already sealed; its manifest cannot be renewed')
    files={p.relative_to(directory).as_posix():sha256(p) for p in directory.rglob('*')
           if p.is_file() and p.name!='collection_manifest.json'}
    manifest=dict(schema_version=1,files=files,inputs={str(p):sha256(p) for p in input_files},
                  raw_arrays_are_original=True)
    pending=directory/'.collection_manifest.pending'
    with pending.open('x',encoding='utf-8') as stream:stream.write(json.dumps(manifest,indent=2))
    pending.replace(path)
    return manifest

def set_world_camera(stage,path,world):
    from pxr import UsdGeom,Gf
    prim=stage.GetPrimAtPath(path)
    parent=UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(np.asarray(world).tolist())*parent.GetInverse())

def capture_truth(camera):
    """Copy the legacy camera's public last_frame without requiring newer truth()."""
    frame=camera.last_frame
    if frame is None:raise RuntimeError('thermal camera did not produce geometry truth')
    planes={k:np.asarray(v).copy() for k,v in frame.planes.items() if isinstance(v,np.ndarray)}
    if getattr(frame,'instance_id',None) is not None:
        planes['instance_id']=np.asarray(frame.instance_id).copy()
    if 'temperature_k' not in planes:raise RuntimeError('thermal truth lacks temperature_k')
    return planes,dict(instance_id=deepcopy(frame.labels),
                       meaning='assigned scene surfaces before optics/detector; not measured temperatures')

def capture_dynamic_pair(preview,seconds,time_code,proxy):
    """Never retain mutable AOV buffers across the IR pass."""
    rgb,_,a=preview.capture(render=True,simulation_time=seconds,frozen_state=True,publish=False)
    rgb=np.asarray(rgb).copy()
    with proxy(preview.stage,time_code) as meshes:
        _,thermal,b=preview.capture(render=True,simulation_time=seconds,frozen_state=True,publish=False)
        planes,legends=capture_truth(preview.thermal)
    pair=validate_pair(a,b)
    b['rgb_capture_pass']=dict(capture_started_wall_s=a['capture_started_wall_s'],
                              capture_finished_wall_s=a['capture_finished_wall_s'],
                              timeline_time_before=a['timeline_time_before'],timeline_time_after=a['timeline_time_after'])
    b['pair_validation']=pair
    b['flame_thermal_proxy']=dict(mode='per_capture_current_time_silhouette',time_code=time_code,
                                 meshes=meshes,animation_time_matches_rgb=True,
                                 model='650 C opaque surface surrogate; not combustion gas radiometry')
    b.update(strict_sync_verified=False,strict_per_frame_pixel_sync_verified=False,
             sync_status='frozen_dynamic_pair_pose_time_verified',sync_validation_status='runtime_contract_only',
             sync_validation_evidence=None,
             sync_note='RGB and IR use the same frozen pose/animation time. Independent moving-marker evidence is separate.')
    return rgb,thermal,b,planes,legends

def build_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--profile',choices=PROFILES,default='native-readable')
    ap.add_argument('--config',type=Path,default=ROOT/'configs/forest_fire_scene.yaml')
    ap.add_argument('--target-config',type=Path,
                    help='场景身份 YAML（目标 prim/锚点/数据集名）；默认按 <scene>_target.yaml 自动查找')
    ap.add_argument('--distances',nargs='+',type=float,default=[8.,15.,25.])
    ap.add_argument('--times',nargs='+',type=float,default=[0.,1.,2.])
    ap.add_argument('--rgb-ev',type=float)
    ap.add_argument('--sensor-range',choices=['original','synthetic700'],default='original')
    ap.add_argument('--trajectory',type=Path,help='optional existing trajectory; reads recorded camera matrices without flight control')
    ap.add_argument('--max-frames',type=int,default=0)
    ap.add_argument('--rt-subframes',type=int,default=4)
    ap.add_argument('--spp',type=int,default=8)
    return ap

def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.out.exists():raise FileExistsError(args.out)
    if any(not math.isfinite(x) or x<=0 for x in args.distances):raise ValueError('distances must be finite and positive')
    if any(not math.isfinite(x) or x<0 for x in args.times):raise ValueError('times must be finite and nonnegative')
    if min(args.spp,args.rt_subframes)<=0 or args.max_frames<0:raise ValueError('invalid capture count/quality')
    sys.path.insert(0,str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))
    # 场景脚本目录从配置里的 stage 推导，而不是写死 forest_fire_scene_local：
    # 换一个场景包也能直接用，不需要改这段代码。
    import yaml as _yaml
    _raw=_yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    _stage=Path(_raw['scene']['stage'])
    if not _stage.is_absolute():_stage=(Path(args.config).resolve().parent/_stage).resolve()
    SCENE_DIR=_stage.parent
    sys.path.insert(0,str(SCENE_DIR/'scripts'))
    from scripts.scene_config import load_scene_config
    config=load_scene_config(args.config,include_task=not bool(args.trajectory))
    task=config.task or {}
    target=load_target_config(args.config,args.target_config)
    if args.sensor_range=='synthetic700':
        config=replace(config,sensors=replace(config.sensors,sensor_config=ROOT/'scene_intake/configs/sensor_synthetic_700c.yaml'))
    input_files={config.source_path,config.scene.stage,config.sensors.sensor_config,
                 config.sensors.thermal_config,config.scene.thermal_scene,Path(__file__),
                 ROOT/'scene_intake/merge_replay_passes.py',
                 ROOT/'scene_intake/fire_thermal_proxy.py',
                 ROOT/'scene_intake/forest_source_clock.py',
                 ROOT/'scene_intake/forest_replay_pose.py',ROOT/'scripts/replay_pose.py',
                 ROOT/'scene_intake/forest_source_clock.py',
                 ROOT/'scene_intake/forest_provenance.py',ROOT/'scene_intake/capture_integrity.py',
                 ROOT/'scene_intake/make_replay_metadata.py',ROOT/'scripts/trajectory_recorder.py',
                 ROOT/'scripts/start_crazyflie_dual_camera_view.py',ROOT/'scripts/frame_metadata.py',
                 ROOT/'scripts/dual_camera_runtime.py',ROOT/'scripts/runtime_parameter_provenance.py'}
    if target.get('config_path'):input_files.add(Path(target['config_path']))
    from scripts.runtime_parameter_provenance import thermal_input_snapshot
    thermal_inputs=thermal_input_snapshot(config)
    recording_context=None
    if args.trajectory:
        from scene_intake.forest_provenance import validate_recording_source
        recording_context=validate_recording_source(config,args.trajectory,thermal_inputs=thermal_inputs)
        task=recording_context['recording']['task'] or {}
        input_files.add(Path(recording_context['recording']['source_run_metadata_path']))
        input_files.add(Path(recording_context['recording']['recording_scene_config_path']))
        input_files.update(Path(p) for p in recording_context['recording']['thermal_input_snapshot']['file_sha256'])
    elif config.task_path:
        input_files.add(config.task_path)
    profile = capture_profile(args.profile, recorded_task=task if recording_context else None)
    rgb_width, rgb_height = profile['rgb_resolution']
    thermal_width, thermal_height = profile['thermal_resolution']
    ev = profile['rgb_ev'] if args.rgb_ev is None else args.rgb_ev
    if not math.isfinite(ev) or abs(ev)>8:raise ValueError('RGB EV must be finite in [-8,8]')
    if recording_context is not None:
        recording_context['render_overrides'].update(rgb_ev=ev, rgb_resolution=profile['rgb_resolution'],
                                                     thermal_resolution=profile['thermal_resolution'])
    input_files.update(Path(p) for p in thermal_inputs['file_sha256'])
    from irsim.config.loader import load_sensor_config
    sensor_model=load_sensor_config(config.sensors.sensor_config)
    response_path=Path(sensor_model.sensor.band.spectral_response)
    if response_path.is_file():input_files.add(response_path)
    if config.sensors.display_config:input_files.add(config.sensors.display_config)
    if args.trajectory:input_files.add(args.trajectory.resolve())
    before={str(p):sha256(p) for p in input_files}
    from isaacsim import SimulationApp
    local_assets=ROOT/'validation/isaac_local_assets';(local_assets/'Isaac').mkdir(parents=True,exist_ok=True)
    app=SimulationApp(dict(headless=True,renderer='PathTracing',multi_gpu=False,
                          samples_per_pixel_per_frame=args.spp,
                          extra_args=['--/persistent/isaac/asset_root/default='+str(local_assets).replace('\\','/')]))
    preview=None
    args.out=args.out.resolve();args.out.mkdir(parents=True)
    try:
        import carb
        import omni.timeline
        from pxr import Gf,Usd,UsdGeom,UsdUtils
        from scripts.start_crazyflie_dual_camera_view import open_preview
        from scripts.experiment_conditions import snapshot_conditions
        try:  # 优先用本项目自己的副本（脱离作者的脚本目录）
            from scene_intake.fire_thermal_proxy import thermal_flame_surfaces
        except ImportError:
            from fire_thermal_proxy import thermal_flame_surfaces
        preview,_=open_preview(app,scene_config=config,width=rgb_width,height=rgb_height,
                              thermal_width=thermal_width,thermal_height=thermal_height,
                              fast=False,start_controller=False,show_ui=False,frozen_subframes=args.rt_subframes,
                              disable_external_viewport=target.get('verification_profile')=='forest_two_target_v1')
        stage=preview.stage
        if not stage.GetPrimAtPath(target['prim_path']):raise ValueError(f"selected stage has no target prim: {target['prim_path']}")
        stage.SetEditTarget(stage.GetSessionLayer())
        layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(config.scene.stage))
        if unresolved:raise RuntimeError(f'unresolved stage dependencies: {unresolved}')
        def _is_local_path(value):
            text=str(value)
            return bool(text) and not text.lower().startswith(('http://','https://','s3://')) and Path(text).is_file()
        # 只对本机文件算哈希：机体 payload 在 S3 上（https 路径），当本地路径读会 OSError。
        input_files.update(Path(str(layer.realPath)) for layer in layers if _is_local_path(layer.realPath))
        input_files.update(Path(str(asset)) for asset in assets if _is_local_path(asset))
        for p in input_files:before.setdefault(str(p),sha256(p))
        settings=carb.settings.get_settings()
        for k,v in {'/rtx/post/tonemap/op':6,'/rtx/post/tonemap/filmIso':100.*2**ev,
                    '/rtx/post/tonemap/cameraShutter':50.,'/rtx/post/tonemap/fNumber':5.,
                    '/rtx/post/histogram/enabled':False}.items():settings.set(k,v)
        conditions=snapshot_conditions(stage,config,rgb_width,rgb_height,
                                       thermal_width=thermal_width,thermal_height=thermal_height)
        # The shared snapshot helper's factory target is replaced only in this adapter.
        conditions['spot']=dict(prim_path=target['prim_path'],target=target['label'],
                                prescribed_temperature_c=target['prescribed_temperature_c'],
                                infrared_material=target['infrared_material'],
                                meaning=f"prescribed {target['label']} surface temperature")
        preview.metadata.update(experiment_conditions=conditions,source_stage_sha256=sha256(config.scene.stage),
                                thermal_input_snapshot=thermal_inputs)
        preview.metadata['thermal_environment_config_sha256']=sha256(config.scene.thermal_scene)
        timeline=omni.timeline.get_timeline_interface();timeline.pause();timeline.set_auto_update(False)
        episode_contract=(task.get('environment_evidence') or {}).get('collection_episode',{})
        body_restorer=None
        if args.trajectory and episode_contract.get('source_snapshot_version',0)>=1:
            from pxr import UsdPhysics
            for prim in Usd.PrimRange(stage.GetPrimAtPath(config.drone.prim_path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            from scripts.replay_pose import FrozenPoseRestorer
            body_restorer=FrozenPoseRestorer(stage,config)
        requests=[]
        trajectory_sha=None
        if args.trajectory:
            from scripts.trajectory_recorder import read_trajectory
            trajectory_sha=sha256(args.trajectory)
            requests_source=read_trajectory(args.trajectory)
            source_by_id={record['frame_id']:record for record in requests_source}
            for record in requests_source:
                world=record.get('rgb_camera_world_matrix')
                if world is None:raise ValueError('trajectory must contain recorded RGB camera world matrices')
                requests.append((float(record['simulation_time']),world,None,record.get('frame_id')))
        else:
            anchor=target['position_m']
            for distance in args.distances:
                eye=Gf.Vec3d(anchor[0],anchor[1]+distance,anchor[2]+1.5)
                world=Gf.Matrix4d().SetLookAt(eye,Gf.Vec3d(*anchor),Gf.Vec3d(0,0,1)).GetInverse()
                for seconds in args.times:requests.append((seconds,np.asarray(world),distance,None))
        if args.max_frames:requests=requests[:args.max_frames]
        if not requests:raise ValueError('no capture requests')
        (args.out/'truth').mkdir();(args.out/'quality').mkdir()
        rows=[]
        for i,(seconds,world,distance,trajectory_id) in enumerate(requests):
            tc=animation_time_code(seconds,stage.GetTimeCodesPerSecond(),stage.GetStartTimeCode(),stage.GetEndTimeCode())
            animation_policy=(task.get('environment_evidence') or {}).get('collection_episode',{}).get('animation_time_policy')
            if args.trajectory and animation_policy=='recorded_source_time':
                from scene_intake.forest_source_clock import recorded_animation_time_code
                tc=recorded_animation_time_code(seconds,stage.GetTimeCodesPerSecond())
            # Loop the scene explicitly; geometry and alpha use the same exact USD time.
            scene_seconds=tc/stage.GetTimeCodesPerSecond()
            timeline.set_current_time(scene_seconds);timeline.commit()
            if body_restorer is not None:body_restorer.restore(source_by_id[trajectory_id])
            for camera in (preview.mount.rgb_path,preview.mount.thermal_path):set_world_camera(stage,camera,world)
            app.update()
            from scene_intake.forest_source_clock import source_thermal_clock
            with source_thermal_clock(preview,seconds):
                rgb,thermal,meta,truth,legends=capture_dynamic_pair(preview,scene_seconds,tc,thermal_flame_surfaces)
            meta['thermal_time_source']='monotonic source request time, independent of looping USD animation'
            meta['animation_time_policy']=animation_policy or 'legacy_explicit_loop'
            if body_restorer is not None:
                from scripts.replay_pose import world_matrix
                from scene_intake.forest_replay_pose import check_body_replay
                meta['body_pose_replay']=check_body_replay(world_matrix(stage.GetPrimAtPath(config.drone.body_path)),source_by_id[trajectory_id])
            q,sat=quality_record(rgb,thermal.apparent_t,thermal.dn16,meta.get('display_profile') or {})
            mask=f'quality/frame_{i:06d}_saturation.png'
            from PIL import Image
            Image.fromarray(sat.astype('uint8')*255).save(args.out/mask);q['saturation_mask']=mask
            meta.update(frame=i,frame_id=i,requested_time_s=seconds,trajectory_frame_id=trajectory_id,
                        # 与回放脚本同名的两个字段：离线视频脚本按它们校验"图与轨迹同源"
                        replay_simulation_time=seconds,trajectory_sha256=trajectory_sha,
                        forest_target=dict(prim_path=target['prim_path'],position_m=target['position_m'],
                                          position_semantics=f"{target['label']} anchor",dataset_scene=target['dataset_scene']),
                        collection_profile=dict(name=profile['name'],rgb_ev=ev,sensor_range=args.sensor_range,
                                                range_meaning=('synthetic 700 C ceiling; not a measured Boson range' if args.sensor_range=='synthetic700' else 'original model transfer; hot targets may saturate')),
                        quality=q,requested_distance_m=distance,requested_render_subframes=args.rt_subframes,
                        task_id=task.get('task_id'),
                        # 说明：帧 JSON 由既有 preview 写入器落盘，中文会按本地代码页写坏。
                        # 这里只放 ASCII 安全的形式：指令本身以 JSON 转义串保存，消费方 json.loads 还原。
                        task_instruction_json=json.dumps(task.get('instruction'), ensure_ascii=True),
                        task_source=(recording_context['recording']['task_config_path'] if recording_context else
                                     (config.task_path.name if config.task_path else None)),
                        task_sha256=(recording_context['recording']['task_config_sha256'] if recording_context else config.task_sha256),
                        task_result_status=(recording_context['recording']['task_result']['status'] if recording_context else 'unjudged'))
            if recording_context:
                source=source_by_id[trajectory_id]
                meta['source_trajectory_pose']={key:deepcopy(source[key]) for key in ('position','orientation') if key in source}
            meta['truth_planes']={}
            for key,value in truth.items():
                filename=f'truth/frame_{i:06d}_{key}.npy';np.save(args.out/filename,value);meta['truth_planes'][key]=filename
            (args.out/f'truth/frame_{i:06d}_legends.json').write_text(json.dumps(legends,indent=2),encoding='utf-8')
            if target.get('verification_profile')=='forest_two_target_v1':
                from scripts.forest_target_evidence import target_surface_evidence
                meta['selected_target_evidence']=target_surface_evidence(truth['instance_id'],legends['instance_id'],
                    truth['temperature_k'],target['prim_path'],target['expected_temperatures_c'])
            preview.save_frame(args.out,i,rgb,thermal,meta)
            rows.append(dict(frame_id=i,time_code=tc,distance_m=distance,**q))
            print('[forest] '+json.dumps(rows[-1]),flush=True)
        for filename,digest in before.items():
            if sha256(filename)!=digest:raise RuntimeError(f'input changed during capture: {filename}')
        if recording_context:
            from scripts.trajectory_recorder import trajectory_provenance
            if trajectory_provenance(args.trajectory)!=recording_context['recording']:
                raise RuntimeError('recording metadata or trajectory changed during capture')
        report=dict(result='PASS',frames=len(rows),profile=profile['name'],rgb_ev=ev,sensor_range=args.sensor_range,
                    rgb_resolution=profile['rgb_resolution'],thermal_resolution=profile['thermal_resolution'],frame_quality=rows,
                    dynamic_proxy='current USD time per frame',strict_per_frame_pixel_sync_verified=False,
                    target=dict(prim_path=target['prim_path'],position_m=target['position_m'],
                                dataset_scene=target['dataset_scene'],config_path=target.get('config_path'),
                                verification_profile=target.get('verification_profile'),
                                expected_temperatures_c=target.get('expected_temperatures_c')),
                    task=dict(task_id=task.get('task_id'),instruction=task.get('instruction'),
                              source=(recording_context['recording']['task_config_path'] if recording_context else (config.task_path.name if config.task_path else None)),
                              sha256=(recording_context['recording']['task_config_sha256'] if recording_context else config.task_sha256),
                              start_position=task.get('start_position')),
                    task_result=(recording_context['recording']['task_result'] if recording_context else None),
                    recording_context=recording_context,
                    wall_time_end=time.time(),data_usage='synthetic development data; prescribed temperatures, uncalibrated sensor')
        (args.out/'collection_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if recording_context:
            from scene_intake.make_replay_metadata import build_replay_metadata
            replay=build_replay_metadata(args.out,args.trajectory,require_manifest=False)
            (args.out/'replay_metadata.json').write_text(json.dumps(replay,indent=2),encoding='utf-8')
        write_manifest(args.out,input_files)
        print('FOREST_COLLECTION_PASS '+str(args.out),flush=True)
        return 0
    except Exception:
        import traceback
        (args.out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        if preview:preview.close()
        app.close()

if __name__=='__main__':raise SystemExit(main())
