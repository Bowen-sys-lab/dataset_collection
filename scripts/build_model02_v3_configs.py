"""Bind actual v3 USD paths and prepare two tasks using one night environment."""
import copy,hashlib,json,math
from pathlib import Path
import yaml
from scripts.model02_migration import bind_surface_rows,validate_export_snapshot,night_weather,core_temperature_truth
from scripts.model02_authoring import compile_configuration
from scripts.scene_config import load_scene_config
from scripts.thermal_config import load_thermal_config

ROOT=Path(__file__).resolve().parents[1];ASSET=ROOT/'factory/model02_thermal_v3';CONFIGS=ROOT/'configs'


def dump(path,data):
    path.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')


def main():
    compile_configuration()
    manifest=json.loads((ASSET/'isaac_manifest.json').read_text(encoding='utf-8'))
    exported=json.loads((ASSET/'export_manifest.json').read_text(encoding='utf-8'))
    authoring=yaml.safe_load((ASSET/'configuration/thermal_authoring.yaml').read_text(encoding='utf-8'))
    validate_export_snapshot(exported,ASSET/'model02_thermal_v3_configured.blend',ASSET/'model02_v3_asset.usdc')
    thermal=yaml.safe_load((CONFIGS/'factory_inspection_thermal.yaml').read_text(encoding='utf-8'))
    thermal['bindings']=[b for b in thermal['bindings'] if not any(b['prim_path']==p or b['prim_path'].startswith(p+'/') for p in manifest['inactive_original_nodes'])]
    thermal['bindings'].append({'name':'preserved_background_coarse','prim_path':'/World/LimeExp_02',
        'include_children':True,'target':'factory_building','material':'concrete'})
    thermal['targets'].update(authoring['targets'])
    thermal['bindings'].extend(bind_surface_rows(manifest['surfaces'],authoring))
    background={'factory_building':20.,'factory_ground':19.5,'factory_equipment':26.,'factory_vegetation':19.,'factory_drone':35.}
    for name,temp in background.items():thermal['targets'][name]={'mode':'fixed','temperature_c':temp}
    dump(CONFIGS/'model02_v3_thermal.yaml',thermal)
    # Explicit synthetic fixture; shared by sky, atmosphere and all weather-driven nodes.
    weather=ASSET/'night_weather.csv'
    weather_text,start_utc=night_weather(authoring['condition'])
    weather.write_text(weather_text,encoding='utf-8')
    environment=yaml.safe_load((ROOT/'third_party/isaac-thermal-camera-simulation/configs/scenes/quad_flight_clear_noon.yaml').read_text(encoding='utf-8'))
    environment['scene'].update(name='model02_v3_operating_night',description='Estimated dry industrial operating night; authored constant air/wind,zero solar,not measured',
        weather_file=str(weather),site={'latitude_deg':31.2,'longitude_deg':121.5,'altitude_m':10.},start_utc=start_utc)
    # Site is a nominal solar/time reference only: the source model has no verified geolocation.
    dump(CONFIGS/'model02_v3_night_environment.yaml',environment)
    display={'schema_version':1,'name':'model02_v3_fixed_temperature','mode':'fixed_apparent_temperature',
        'temperature_min_c':0.,'temperature_max_c':85.,'dde_gain':0.,'polarity':'white_hot','palette':'gray'}
    dump(CONFIGS/'model02_v3_display.yaml',display)
    reports=[]
    for entry in manifest['tasks']:
        kind=entry['kind'];scene=yaml.safe_load((CONFIGS/'model02_refined_scene.yaml').read_text(encoding='utf-8'))
        scene['scene']['stage']='../factory/model02_thermal_v3/model02_v3_scene.usda'
        scene['scene']['thermal_scene']='model02_v3_night_environment.yaml'
        scene['task']['file']=f'model02_v3_{kind}_task.yaml'
        scene['sensors'].update(thermal_config='model02_v3_thermal.yaml',display_config='model02_v3_display.yaml',
            material_directory='../factory/model02_thermal_v3/configuration/materials')
        task=yaml.safe_load((CONFIGS/'model02_refined_task.yaml').read_text(encoding='utf-8'))
        task.update(task_id=entry['task_id'],instruction=entry['instruction'],target_prim_path=entry['target_prim_path'],
            target_position=entry['target_center_m'],start_position=entry['reference_waypoints_m'][0],
            target_position_semantics='Area-weighted center of all actual exported independently configured core facets',
            target_description='GC075储罐罐壁局部异常' if kind=='tank' else 'GC089热介质立管保温破损处钢管裸露热点',
            notes='人工键盘驾驶，按T只记录人工确认；完整热点包含多个独立温区，见证据清单。仿真验证截图不是人工驾驶记录。')
        evidence=task['environment_evidence']['factory_inspection'];eye=entry['observation_position_m'];center=entry['target_center_m']
        evidence.update({k:entry[k] for k in ['equipment_prim_path','normal_surface_prim_path','rgb_material_path','geometry_file','geometry_sha256']})
        evidence.update(hotspot_prim_path=entry['target_prim_path'],hotspot_prim_paths=entry['hotspot_prim_paths'],
            target_surface_ids=entry['target_surface_ids'],hotspot_position_m=center,observation_position=eye,
            observation_yaw_rad=math.atan2(center[1]-eye[1],center[0]-eye[0]),
            reference_waypoints=entry['reference_waypoints_m'],reference_waypoints_m=entry['reference_waypoints_m'],
            reference_path_length_m=entry['reference_path_length_m'],minimum_route_body_clearance_lower_bound_m=entry['minimum_route_body_clearance_lower_bound_m'],
            observation_equipment_clearance_m=entry['observation_equipment_clearance_m'],
            geometry_source='Actual composed v3 USD; equipment geometry restricted by original source ID; full core triangles included',
            thermal_truth={**core_temperature_truth(entry,authoring),'hotspot_area_m2':entry['hotspot_area_m2'],
                'source_mapping':'factory/model02_thermal_v3/configuration/thermal_authoring.yaml',
                'basis':'Estimated operating snapshot; not measured or conduction-solved'},
            evidence_images=[],inspection_requirements=copy.deepcopy(evidence['inspection_requirements']))
        evidence.pop('comparison_equipment_prim_path',None)
        # Old ground/route-height measurements refer to v1 and cannot certify v3.
        for key in ['ground_world_z_m','route_ground_height_range_m','route_ground_height_semantics']:evidence.pop(key,None)
        task['environment_evidence']['lighting']={'preset':'model02_v3_operating_night','fixtures':13,'manifest':'factory/model02_thermal_v3/isaac_manifest.json'}
        task['environment_evidence']['condition_assumptions']={'authoring_condition':authoring['condition'],
            'site':'Nominal31.2N121.5E10m solely for solar/time; source model location unknown',
            'background_temperatures_c':background,'humidity_fraction':.6,'cloud_fraction':.1,'measured':False}
        task['environment_landmarks']=['蓝色护栏储罐','较高设备组与架空管廊','沿线厂房','道路灯具']
        dump(CONFIGS/f'model02_v3_{kind}_task.yaml',task);dump(CONFIGS/f'model02_v3_{kind}_scene.yaml',scene)
        loaded=load_scene_config(CONFIGS/f'model02_v3_{kind}_scene.yaml')
        reports.append({'kind':kind,'task_id':loaded.task['task_id'],'scene_config':str(loaded.source_path)})
    loaded=load_thermal_config(CONFIGS/'model02_v3_thermal.yaml')
    report={'result':'PASS_BOUND_CONFIGURATION_SCHEMA','surface_bindings':8327,'total_bindings':len(loaded.bindings),
        'targets':len(loaded.targets),'tasks':reports,'manual_success':None,
        'authoring_sha256':hashlib.sha256((ASSET/'configuration/thermal_authoring.yaml').read_bytes()).hexdigest(),
        'material_library_sha256':json.loads((ASSET/'configuration/blender_parameters.json').read_text(encoding='utf-8'))['material_library_sha256'],
        'exported_geometry_sha256':exported['asset_sha256']}
    (ASSET/'binding_configuration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
