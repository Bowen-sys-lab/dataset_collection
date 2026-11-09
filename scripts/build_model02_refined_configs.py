"""Bind every exported refined surface using its actual USD path."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'configs'
ASSETS = ROOT/'factory/model02_thermal_v2'


def main():
    manifest = json.loads((ASSETS/'refined_asset_manifest.json').read_text(encoding='utf-8'))
    scene = yaml.safe_load((CONFIGS/'factory_inspection_scene.yaml').read_text(encoding='utf-8'))
    scene['scene']['stage'] = '../factory/model02_thermal_v2/model02_refined_scene.usda'
    scene['scene']['static_targets'].append('/World/IndustrialPark02')
    scene['task']['file'] = 'model02_refined_task.yaml'
    scene['sensors']['thermal_config'] = 'model02_refined_thermal.yaml'
    scene['sensors']['display_config'] = 'model02_refined_display.yaml'
    task = yaml.safe_load((CONFIGS/'factory_inspection_task.yaml').read_text(encoding='utf-8'))
    task['task_id'] = 'factory_tank_hotspot_refined_02'
    task['target_prim_path'] = manifest['hotspot_prim_path']
    task['target_position'] = manifest['hotspot_center_m']
    task['target_position_semantics'] = 'Area-weighted center of the independently configured local hotspot core on the source tank facets'
    task['environment_description'] += ' 夜间弱光工况，13盏常规灯具分布于道路、入口、管廊及设备区外围。'
    evidence = task['environment_evidence']['factory_inspection']
    for key in ['equipment_prim_path','hotspot_prim_path','normal_surface_prim_path',
                'comparison_equipment_prim_path','rgb_material_path','geometry_file','geometry_sha256']:
        evidence[key] = manifest[key]
    evidence['hotspot_position_m'] = manifest['hotspot_center_m']
    evidence['thermal_truth'] = {'hotspot_temperature_c':78.,'normal_equipment_temperature_c':30.,
        'hotspot_area_m2':manifest['hotspot_area_m2'],'surface_partition':'Local core plus eight transition bands on existing source facets',
        'source_mapping':'factory/model02_thermal_v2/surfaces.json',
        'transition_temperature_c':[72.,66.,60.,53.,46.,40.,35.,32.]}
    task['environment_evidence']['lighting'] = {'preset':'low_light_refined',
        'manifest':'factory/model02_thermal_v2/refined_asset_manifest.json','fixtures':13}
    task['notes'] += ' 本版重建门窗并细分四座储罐，需使用本版新记录，不复用旧场景来源哈希。'
    thermal = yaml.safe_load((CONFIGS/'factory_inspection_thermal.yaml').read_text(encoding='utf-8'))
    thermal['bindings'] = [binding for binding in thermal['bindings'] if not any(
        binding['prim_path'].startswith(prefix) for prefix in manifest['inactive_original_nodes'])]
    for surface in manifest['surfaces']:
        name = surface['temperature_target_id']
        thermal['targets'][name] = {'mode':'fixed','temperature_c':surface['temperature_c']}
        thermal['bindings'].append({'name':name,'prim_path':surface['usd_prim_path'],
            'include_children':False,'target':name,'material':surface['ir_material']})
    for name,data in [('scene',scene),('task',task),('thermal',thermal)]:
        path = CONFIGS/f'model02_refined_{name}.yaml'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')
    (CONFIGS/'model02_refined_display.yaml').write_bytes((CONFIGS/'factory_inspection_display.yaml').read_bytes())
    print(json.dumps({'explicit_new_surface_bindings':len(manifest['surfaces']),
                      'task_id':task['task_id'],'scene':'configs/model02_refined_scene.yaml'},ensure_ascii=False))


if __name__ == '__main__':
    main()
