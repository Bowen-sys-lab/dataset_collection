"""Factory-only target/start selection with persisted, replayable configuration."""
import copy
import hashlib
import json
from pathlib import Path
import yaml
from scripts.collection_capture_defaults import TRAJECTORY_PERIOD_S

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/model02_v7m_bc_two_target_plan.yaml'
READINESS = ROOT/'validation/factory-two-target/20261030/readiness.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_dependency_hashes():
    from scripts.scene_config import load_scene_config
    from scripts.runtime_parameter_provenance import thermal_input_snapshot
    config=load_scene_config(ROOT/'configs/model02_v7m_bc_pilot_scene.yaml')
    snapshot=thermal_input_snapshot(config)
    hashes={Path(path).relative_to(ROOT).as_posix():sha for path,sha in snapshot['file_sha256'].items()}
    # Pin vendor sensor spectra as well as the YAML that references them.
    for path in (ROOT/'third_party/isaac-thermal-camera-simulation/data').rglob('*.csv'):
        hashes[path.relative_to(ROOT).as_posix()]=digest(path)
    return hashes


def resolve_selection(target='tank', instruction_variant=1, start_group='A', *, require_ready=True):
    if target not in ('tank','pipe') or type(instruction_variant) is not int or instruction_variant not in (1,2,3):
        raise ValueError('Target must be tank/pipe and InstructionVariant must be 1..3')
    plan = yaml.safe_load(PLAN.read_text(encoding='utf-8'))
    group = next((g for g in plan['start_groups'] if g['group']==start_group),None)
    if group is None: raise ValueError('StartGroup must be A/B/D/E/F')
    item=plan['targets'][0 if target=='tank' else 1]
    task=yaml.safe_load((ROOT/item['source_task']).read_text(encoding='utf-8'))
    task['instruction']=item['instruction_variants'][instruction_variant-1]['text']
    task['task_id']='model02_v7m_bc_pilot' if target=='tank' else 'model02_v7m_bc_pipe_pilot'
    task['start_position']=copy.deepcopy(group['position_m'])
    task['notes']='人工确认并停止；不要求环绕、返航或累计里程。自动几何指标作为独立质检参考。'
    inspection=task['environment_evidence']['factory_inspection']
    inspection['reference_route_status']='historical source route, not a validated route for the selected start'
    req=inspection['inspection_requirements']
    req.update(hold_duration_s=3.,require_path_length=False)
    inspection['flight_settings'].update(horizontal_speed_m_s=3.,vertical_speed_m_s=1.,speed_scale=1.,yaw_speed_degrees_s=45.)
    task['environment_evidence']['bc_pilot']={
        'version':3,'target':target,'instruction_variant':instruction_variant,
        'start_group':start_group,'split':group['split'],'confirmation_mode':'manual_only',
        'requires_orbit':False,'requires_return':False,'record_period_s':TRAJECTORY_PERIOD_S,
        'automatic_task_acceptance':'NOT_EVALUATED'}
    if require_ready:
        if not READINESS.is_file(): raise ValueError('Factory readiness missing; run verify_factory_collection.py first')
        report=json.loads(READINESS.read_text(encoding='utf-8'))
        if runtime_dependency_hashes()!=report.get('runtime_dependencies'):
            raise ValueError('Sensor/thermal/weather/display/material dependencies changed; rerun readiness and native view checks')
        for path, expected in report['inputs'].items():
            if digest(ROOT/path)!=expected: raise ValueError(f'Readiness input changed: {path}; rerun geometry audit')
        if not report['combinations'].get(f'{target}:{start_group}',{}).get('enabled'):
            raise ValueError(f'Unverified combination {target}:{start_group}: '+str(report['combinations'].get(f'{target}:{start_group}')))
        geometry=report['targets'][target]
        inspection['geometry_file']=geometry['geometry_file']
        inspection['geometry_sha256']=geometry['geometry_sha256']
        inspection['geometry_source']='actual composed v7m equipment exported by verify_factory_collection.py'
        if digest(ROOT/inspection['geometry_file']) != inspection['geometry_sha256']:
            raise ValueError('Verified equipment geometry changed; rerun geometry audit')
    return {'task':task,'yaw_degrees':group['yaw_degrees'],'split':group['split'],
            'target':target,'instruction_variant':instruction_variant,'start_group':start_group}


def runtime_configuration(selection, runtime_dir):
    """Bytes and paths are planned without writing; CheckOnly stays non-mutating."""
    source=ROOT/'configs/model02_v7m_bc_pilot_scene.yaml'
    data=yaml.safe_load(source.read_text(encoding='utf-8'))
    for section, keys in (('scene',('stage','thermal_scene')),('sensors',('sensor_config','thermal_config','display_config','material_directory'))):
        for key in keys:
            if data[section].get(key): data[section][key]=str((source.parent/data[section][key]).resolve())
    task_path=Path(runtime_dir)/'task.yaml'
    config_path=Path(runtime_dir)/'scene.yaml'
    data['task']['file']=str(task_path)
    task_bytes=yaml.safe_dump(selection['task'],allow_unicode=True,sort_keys=False).encode('utf-8')
    config_bytes=yaml.safe_dump(data,allow_unicode=True,sort_keys=False).encode('utf-8')
    return config_path,config_bytes,task_path,task_bytes
