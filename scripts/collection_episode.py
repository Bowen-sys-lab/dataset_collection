"""Versioned episode selection. Inspection never implies verified launchability."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import base64
import datetime
import math
import subprocess
import ast
import yaml
from scripts.collection_capture_defaults import TRAJECTORY_PERIOD_S
from scripts.collection_shell import powershell_executable
from scripts.manual_collision_undo import collision_undo_enabled, RECOVERY_CONTRACT

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'configs/collection_episodes/pilot30.json'
READINESS = ROOT/'validation/diagnostics/pilot30-preparation-20261031/episode-readiness.json'


def load_episodes():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))['episodes']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _capture_dependency_paths():
    """Conservative static closure of project-local camera/pose/capture imports."""
    pending=[ROOT/'scripts'/name for name in ('start_crazyflie_dual_camera_view.py','capture_crazyflie_thermal_static.py',
        'replay_pose.py','rgb_scene_diagnostics.py','drone_keyboard_controller.py')]
    found=set()
    while pending:
        path=pending.pop().resolve()
        if path in found or not path.is_relative_to(ROOT) or not path.is_file():continue
        found.add(path);package=path.relative_to(ROOT).parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix='.'.join(package[:len(package)-node.level+1]) if node.level else ''
                module='.'.join(v for v in (prefix,node.module or '') if v)
                modules=[module]+[module+'.'+alias.name for alias in node.names]
            for module in modules:
                if module=='scripts' or module.startswith('scripts.'):
                    candidate=ROOT/Path(*module.split('.'))
                    if candidate.with_suffix('.py').is_file():pending.append(candidate.with_suffix('.py'))
                    elif (candidate/'__init__.py').is_file():pending.append(candidate/'__init__.py')
    return found


def required_input_paths(row):
    from scripts.scene_config import load_scene_config
    from scripts.runtime_parameter_provenance import thermal_input_snapshot
    from scripts.forest_collection_config import scene_config_path,target_config_path,source_asset_root,is_two_target,REGISTRY
    source=(ROOT/'configs/model02_v7m_bc_pilot_scene.yaml' if row['scene']=='factory'
            else scene_config_path(row))
    config=load_scene_config(source)
    paths={MANIFEST,source,config.scene.stage,ROOT/'scripts/record.ps1',
           ROOT/'scripts/collection_episode.py',ROOT/'scripts/trajectory_recorder.py',
           ROOT/'scripts/collection_mileage.py',ROOT/'scripts/on_demand_ir.py',
           ROOT/'scripts/recording_quality.py',
           ROOT/'scripts/start_crazyflie_dual_camera_view.py',ROOT/'scripts/run_isaac_with_clean_env.ps1',
           ROOT/'scripts/process.ps1',ROOT/'scripts/process_collection_episode.py',
           ROOT/'scripts/episode_replay_source.py',ROOT/'scripts/episode_dataset.py',ROOT/'scripts/collection_shell.py'}
    paths.add(ROOT/'lesson_crazyflie.usd')
    paths.update(_capture_dependency_paths())
    paths.update((ROOT/'third_party/isaac-thermal-camera-simulation/src').rglob('*.py'))
    paths.update(Path(p) for p in thermal_input_snapshot(config)['file_sha256'])
    if row['scene']=='factory':
        paths.add(ROOT/'scripts/factory_occlusion.py')
        paths.add(ROOT/'scripts/factory_geometry_provenance.py')
        if any(r.get('start_version')=='factory_interior_occluded_v1' for r in load_episodes() if r['scene']=='factory'):
            from scripts.factory_geometry_provenance import cache_sources_current
            metadata_path=ROOT/'validation/diagnostics/factory-interior-v1/geometry.json'
            metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
            if not cache_sources_current(metadata,metadata_path.with_suffix('.npz'),ROOT):
                raise ValueError('Factory geometry cache is missing or stale against composed scene inputs; regenerate geometry and route checks')
            paths.add(metadata_path)
            paths.update(ROOT/p for p in metadata['geometry_source_inputs'])
        from scripts.factory_collection_config import READINESS as target_proof, PLAN
        paths.update((target_proof,PLAN,ROOT/'configs/model02_v3_pipe_task.yaml',
                      ROOT/'configs/model02_v7m_bc_pilot_task.yaml'))
        proof=json.loads(target_proof.read_text(encoding='utf-8'))
        paths.update(ROOT/p for p in proof['inputs'])
        paths.update((ROOT/'scripts/replay_factory_inspection.py',ROOT/'scripts/process_factory_inspection.ps1',
                      ROOT/'scripts/verify_factory_replay_video.py',ROOT/'scripts/generate_replay_video.py'))
    else:
        paths.update((ROOT/'configs/forest_fire_task.yaml',target_config_path(row),
                      ROOT/'scene_intake/forest_collection.py',ROOT/'scene_intake/fire_thermal_proxy.py',
                      ROOT/'scene_intake/forest_source_clock.py'))
        paths.update((ROOT/'scripts/forest_collection_config.py',ROOT/'scripts/forest_target_evidence.py',ROOT/'scripts/forest_readiness.py'))
        if is_two_target(row):
            paths.add(REGISTRY)
            # Both goals share one current scene and evidence context.
            paths.update(target_config_path({**row,'target':target}) for target in ('barrel','seated_person'))
        paths.update((ROOT/'scene_intake/forest_replay_pose.py',ROOT/'scripts/replay_pose.py'))
        paths.update(ROOT/p for p in ('scene_intake/forest_provenance.py','scene_intake/capture_integrity.py',
            'scene_intake/make_replay_metadata.py','scene_intake/forest_dataset.py','scene_intake/verify_dataset.py',
            'scripts/generate_replay_video.py'))
    assets=config.scene.stage.parent if row['scene']=='factory' else source_asset_root(row)
    paths.update(p for p in assets.rglob('*') if p.is_file()
                 and not any(part in ('__pycache__','.git','.pytest_cache') for part in p.parts)
                 and p.suffix not in ('.pyc','.pyo'))
    return {p.resolve().relative_to(ROOT).as_posix() for p in paths}


def valid_readiness(row,evidence):
    try:
        inputs=evidence.get('input_sha256',{})
        native=evidence.get('native_evidence_sha256',{})
        if any(key in native and native[key]!=value for key,value in inputs.items()):return False
        report_path=evidence.get('native_report','')
        if not report_path or report_path not in native or not required_input_paths(row).issubset(inputs):return False
        for relative,digest in {**inputs,**native}.items():
            path=(ROOT/relative).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path)!=digest:return False
        report=json.loads((ROOT/report_path).read_text(encoding='utf-8'))
        if (report.get('schema_version')!=1 or report.get('result')!='PASS'
            or report.get('scene')!=row['scene'] or row['episode'] not in report.get('episodes',[])
            or report.get('source_manifest_sha256')!=row['manifest_sha256']
            or report.get('source_inputs')!=inputs):return False
        from scripts.forest_collection_config import is_two_target
        replay_gate=('keyframe_replay_passed' if row.get('start_version')=='factory_interior_occluded_v1'
                     else 'short_replay_passed' if is_two_target(row) else 'long_replay_passed')
        if not all(report.get(key) is True for key in ('spawn_passed','route_clearance_passed','view_passed',replay_gate)):return False
        position=report['position_m'];yaw=float(report['yaw_degrees'])
        if len(position)!=3 or not all(math.isfinite(float(v)) for v in position) or not math.isfinite(yaw):return False
        if math.dist(position,row['candidate_position_m'])>.25:return False
        delta=math.radians(yaw-row['candidate_yaw_degrees'])
        if abs(math.atan2(math.sin(delta),math.cos(delta)))>math.radians(1.):return False
        if row.get('start_version')=='factory_interior_occluded_v1':
            from scripts.factory_occlusion import validate_file_evidence,validate_route_file_evidence
            if report.get('initial_occlusion_passed') is not True:return False
            if not validate_file_evidence(report.get('occlusion_evidence_path'),row,ROOT,native,expected_source_inputs=inputs):return False
            inspection=task_snapshot(row)['environment_evidence']['factory_inspection']
            return validate_route_file_evidence(report.get('route_evidence_path'),row,ROOT,native,inspection['hotspot_prim_paths'],expected_source_inputs=inputs)
        if is_two_target(row):
            from scripts.forest_readiness import validate_forest_evidence
            return validate_forest_evidence(row,report,ROOT,native,inputs)
        return True
    except (KeyError,ValueError,TypeError,OSError):return False


def resolve_episode(scene, identifier, *, require_ready=False):
    if scene not in ('factory','forest') or not re.fullmatch(r'[FR][0-9]{2}',identifier):
        raise ValueError('Unknown scene or Episode; expected factory/F01..F10 or forest/R01..R10')
    rows=[row for row in load_episodes() if row['scene']==scene and row['episode']==identifier]
    if len(rows)!=1:
        raise ValueError('Episode does not belong to the selected scene')
    row=copy.deepcopy(rows[0]);row['enabled']=False
    row['manifest_sha256']=sha256(MANIFEST)
    proof = json.loads(READINESS.read_text(encoding='utf-8')) if READINESS.is_file() else {}
    evidence=proof.get('episodes',{}).get(identifier,{})
    if evidence.get('enabled') and evidence.get('manifest_sha256')==row['manifest_sha256']:
        row['enabled']=valid_readiness(row,evidence)
    if require_ready and not row['enabled']:
        detail=('; internal start requires current RGB/IR target masks and physical-blocker evidence'
                if row.get('start_version')=='factory_interior_occluded_v1' else
                '; forest requires current route, target-specific RGB/IR and replay evidence'
                if row.get('start_version')=='forest_two_target_v1' else '')
        raise ValueError(f'Episode {identifier} readiness incomplete or stale{detail}; new starts remain disabled')
    return row


def task_snapshot(row):
    if row['scene']=='factory':
        from scripts.factory_collection_config import resolve_selection
        task=resolve_selection(row['target'],1,'A',require_ready=False)['task']
        task['environment_evidence'].pop('bc_pilot', None)
        # Reuse checked target geometry, not the old start certification.
        from scripts.factory_collection_config import READINESS as TARGET_READINESS
        proof=json.loads(TARGET_READINESS.read_text(encoding='utf-8'))
        geometry=proof['targets'][row['target']]
        if sha256(ROOT/geometry['geometry_file'])!=geometry['geometry_sha256']:
            raise ValueError('Factory measurement geometry changed; regenerate target evidence')
        inspection=task['environment_evidence']['factory_inspection']
        inspection.update(geometry_file=geometry['geometry_file'],geometry_sha256=geometry['geometry_sha256'])
        inspection.pop('reference_waypoints',None)
        inspection['reference_route_status']='episode candidate; route evidence stored independently'
    else:
        from scripts.forest_collection_config import task_for
        task=task_for(row)
    task['task_id']=f"pilot30_{row['scene']}_{row['target']}"
    task['instruction']=row['instruction']
    task['start_position']=list(row['candidate_position_m'])
    task['notes']='Manual confirmation; start-relative bearing is fixed at initial spawn. No path length gate.'
    task.setdefault('environment_evidence',{})['collection_episode']={
        key:copy.deepcopy(row[key]) for key in ('episode','scene','target','start_version','start_group',
            'split','instruction_variant','language','candidate_yaw_degrees','length_policy','manifest_sha256')}
    if row.get('start_version')=='factory_interior_occluded_v1':
        task['environment_evidence']['collection_episode'].update(
            initial_visibility='both authored target source groups physically occluded in initial and target-aimed RGB/IR diagnostic views',
            start_region_description=row.get('start_region_description','internal open-air factory passage'))
    if row['scene']=='forest':
        from scripts.forest_collection_config import target_config_path
        task['environment_evidence']['collection_episode']['target_config_sha256']=sha256(target_config_path(row))
        task['environment_evidence']['collection_episode']['animation_time_policy']='recorded_source_time'
    return task


def runtime_snapshot(row, location):
    from scripts.forest_collection_config import scene_config_path
    source=(ROOT/'configs/model02_v7m_bc_pilot_scene.yaml' if row['scene']=='factory'
            else scene_config_path(row))
    config=yaml.safe_load(source.read_text(encoding='utf-8'))
    for section,keys in (('scene',('stage','thermal_scene')),
                         ('sensors',('sensor_config','thermal_config','display_config','material_directory'))):
        for key in keys:
            if config[section].get(key):config[section][key]=str((source.parent/config[section][key]).resolve())
    config['task']={'file':str(Path(location)/'task.yaml')}
    task=task_snapshot(row)
    assets=scene_asset_snapshot(row)
    task['environment_evidence']['collection_episode'].update(source_snapshot_version=1,
        scene_assets_path=str(Path(location)/'scene_assets.json'),scene_assets_sha256=hashlib.sha256(assets).hexdigest())
    if row['scene']=='forest':task['environment_evidence']['collection_episode']['target_config_path']=str(Path(location)/'target.yaml')
    return (yaml.safe_dump(config,allow_unicode=True,sort_keys=False),
            yaml.safe_dump(task,allow_unicode=True,sort_keys=False))


def scene_asset_snapshot(row):
    extensions={'.usd','.usda','.usdc','.png','.jpg','.jpeg','.tif','.tiff','.exr','.mdl','.mtlx',
                '.abc','.obj','.fbx','.bin','.gltf','.glb','.npz'}
    assets={str((ROOT/p).resolve()):sha256(ROOT/p) for p in required_input_paths(row)
            if Path(p).suffix.lower() in extensions}
    return json.dumps(assets,sort_keys=True,indent=2).encode('utf-8')


def recording_identity(row,stamp,smoke_frames=0):
    if type(smoke_frames) is not int or not 0<=smoke_frames<=120:
        raise ValueError('SmokeFrames must be an integer in 0..120')
    prefix='probe-' if smoke_frames else ''
    return {'run_id':f"{row['scene']}-pilot30-{prefix}{row['episode']}-{stamp}",
            'root':ROOT/'validation/diagnostics'/row['scene']/'recording_probes' if smoke_frames else ROOT/'validation/trajectory_runs',
            'trajectory_source':'synthetic_control_probe' if smoke_frames else 'manual_driving'}


def episode_preview_args(row,location,identity,smoke_frames=0,collision_undo=None):
    run_id=identity['run_id']
    location=Path(location)
    recovery=collision_undo_enabled(probe_control=bool(smoke_frames),requested=collision_undo)
    preview=ROOT/'validation/diagnostics'/row['scene']/run_id
    # Arguments are passed as literals in one encoded child command.
    python_args=[str(ROOT/'scripts/start_crazyflie_dual_camera_view.py'),'--','--config',str(location/'scene.yaml'),
        '--preview-mode','live','--record-trajectory','--run-id',run_id,'--no-save',
        '--trajectory-root',str(identity['root']),
        '--horizontal-speed','3','--vertical-speed','1','--trajectory-period',str(TRAJECTORY_PERIOD_S),
        '--preview-fps','15','--thermal-preview-fps','2','--live-refresh','on-demand','--out',str(preview),
        '--start-position',*[str(v) for v in row['candidate_position_m']],
        '--start-yaw-degrees',str(row['candidate_yaw_degrees'])]
    python_args+=['--collision-undo' if recovery else '--no-collision-undo']
    if row.get('start_version')=='forest_two_target_v1':python_args+=['--disable-external-viewport']
    if smoke_frames:python_args+=['--headless','--probe-control','--frames',str(smoke_frames)]
    return python_args


def record_episode(row,smoke_frames=0,collision_undo=None):
    recording_identity(row,'validation',smoke_frames)
    # Only explicitly synthetic, finite diagnostics may test unverified starts.
    row=resolve_episode(row['scene'],row['episode'],require_ready=not smoke_frames)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    identity=recording_identity(row,stamp,smoke_frames)
    recovery=collision_undo_enabled(probe_control=bool(smoke_frames),requested=collision_undo)
    run_id=identity['run_id'];run=identity['root']/run_id
    location=ROOT/'validation/collection_launches'/run_id
    location.mkdir(parents=True,exist_ok=False)
    config,task=runtime_snapshot(row,location)
    asset_bytes=scene_asset_snapshot(row)
    if hashlib.sha256(asset_bytes).hexdigest()!=yaml.safe_load(task)['environment_evidence']['collection_episode']['scene_assets_sha256']:
        raise ValueError('Scene assets changed while preparing the launch')
    (location/'scene_assets.json').write_bytes(asset_bytes)
    if row['scene']=='forest':
        from scripts.forest_collection_config import target_config_path
        target_bytes=target_config_path(row).read_bytes()
        if hashlib.sha256(target_bytes).hexdigest()!=yaml.safe_load(task)['environment_evidence']['collection_episode']['target_config_sha256']:
            raise ValueError('Forest target changed while preparing the launch')
        (location/'target.yaml').write_bytes(target_bytes)
    (location/'scene.yaml').write_text(config,encoding='utf-8')
    (location/'task.yaml').write_text(task,encoding='utf-8')
    (location/'episode.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    (location/'recovery.json').write_text(json.dumps({'collision_undo':recovery,
        'collision_contract':RECOVERY_CONTRACT if recovery else 'disabled','undo_seconds':3.},indent=2),encoding='utf-8')
    manifest_bytes=MANIFEST.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=row['manifest_sha256']:
        raise ValueError('Episode manifest changed while preparing the launch')
    (location/'manifest.json').write_bytes(manifest_bytes)
    python_args=episode_preview_args(row,location,identity,smoke_frames,collision_undo)
    literal=lambda value:"'"+str(value).replace("'","''")+"'"
    command=(f"& {literal(ROOT/'scripts/run_isaac_with_clean_env.ps1')} -ScriptPath "
             f"{literal(ROOT/'scene_intake/run_with_local_assets.py')} -ScriptArgs @("
             +','.join(map(literal,python_args))+')')
    encoded=base64.b64encode(command.encode('utf-16-le')).decode('ascii')
    print(f'[episode] Recording {run_id}; H checks IR, T confirms, then close preview.',flush=True)
    code=subprocess.run([str(powershell_executable()),'-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand',encoded],cwd=ROOT).returncode
    metadata_path=run/'run_metadata.json'
    if code or not metadata_path.is_file():
        raise ValueError(f'Recording incomplete; preserve logs and snapshot: {location}')
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    from scripts.episode_replay_source import validate_recorded_scene_assets
    validate_recorded_scene_assets(metadata)
    expected=yaml.safe_load(task)
    if metadata.get('task')!=expected or metadata.get('trajectory_source')!=identity['trajectory_source']:
        raise ValueError('Recorded episode task/source differs from launch contract')
    from scripts.trajectory_recorder import read_trajectory
    from scripts.factory_bc_pilot import recording_quality
    quality=recording_quality(read_trajectory(run/'trajectory.jsonl'),
                              sample_period_s=metadata.get('sample_period_s', .1),metadata=metadata)
    position=metadata.get('start_position')
    if position is None or math.dist(position,row['candidate_position_m'])>.25:
        raise ValueError('Actual spawn differs from episode start')
    yaw=metadata.get('start_yaw_radians')
    delta=None if yaw is None else yaw-math.radians(row['candidate_yaw_degrees'])
    if delta is None or abs(math.atan2(math.sin(delta),math.cos(delta)))>math.radians(1):
        raise ValueError('Actual spawn heading differs from episode start')
    if metadata.get('scene_config_sha256')!=sha256(location/'scene.yaml'):
        raise ValueError('Recorded configuration differs from snapshot')
    (run/'technical_quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
    if quality['result']!='PASS':raise ValueError('Recording technical quality failed; raw files retained')
    (run/'episode_runtime_check.json').write_text(json.dumps({'runtime_contract_verified':True,
        'scene':row['scene'],'episode':row['episode'],'trajectory_source':identity['trajectory_source'],
        'source_metadata_sha256':sha256(metadata_path),'source_trajectory_sha256':sha256(run/'trajectory.jsonl'),
        'actual_start_position_m':position,'actual_start_yaw_radians':yaw,'task_acceptance':'NOT_EVALUATED'},indent=2),encoding='utf-8')
    print(f'[episode] Recording saved: {run}; manual quality review is separate.',flush=True)
    print(f'Replay: powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{ROOT/"scripts/process.ps1"}" -Run "{run}"',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--episode',required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-frames',type=int,default=0)
    parser.add_argument('--collision-undo',action=argparse.BooleanOptionalAction,default=None)
    args=parser.parse_args()
    try:
        if not 0<=args.smoke_frames<=120:raise ValueError('SmokeFrames must be in 0..120')
        row=resolve_episode(args.scene,args.episode,require_ready=not args.check_only and not args.smoke_frames)
        if args.check_only:
            recovery=collision_undo_enabled(probe_control=bool(args.smoke_frames),requested=args.collision_undo)
            print(json.dumps(dict(row,task=task_snapshot(row),collision_undo=recovery,
                collision_contract=RECOVERY_CONTRACT if recovery else 'disabled'),ensure_ascii=False,indent=2))
        else:
            print(f"[episode] {row['scene']} / {row['episode']} / {row['split']}\nTask: {row['instruction']}",flush=True)
            record_episode(row,args.smoke_frames,args.collision_undo)
        return 0
    except (ValueError,KeyError,OSError) as error:
        print(f'[episode] Refused: {error}')
        return 2


if __name__=='__main__':raise SystemExit(main())
