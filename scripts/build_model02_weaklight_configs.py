"""Dedicated weak-light config; retain existing factory task/thermal semantics."""
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'configs'
ASSETS = ROOT/'factory/model02_thermal_v1'


def main():
    manifest = json.loads((ASSETS/'weaklight_asset_manifest.json').read_text(encoding='utf-8'))
    scene = yaml.safe_load((CONFIGS/'factory_inspection_scene.yaml').read_text(encoding='utf-8'))
    scene['scene']['stage'] = '../factory/model02_thermal_v1/model02_weaklight.usda'
    scene['scene']['static_targets'].append('/World/IndustrialPark02/Lighting')
    scene['task']['file'] = 'model02_weaklight_task.yaml'
    scene['sensors']['thermal_config'] = 'model02_weaklight_thermal.yaml'
    scene['sensors']['display_config'] = 'model02_weaklight_display.yaml'
    task = yaml.safe_load((CONFIGS/'factory_inspection_task.yaml').read_text(encoding='utf-8'))
    task['task_id'] = 'factory_tank_hotspot_weaklight_01'
    task['environment_description'] += ' 弱光工况：道路灯、入口壁灯与设备区灯局部工作，沿线存在照明区与暗区。'
    task['environment_evidence']['lighting'] = {
        'preset':'low_light', 'asset_manifest':'factory/model02_thermal_v1/weaklight_asset_manifest.json',
        'thermal_control':'Existing prescribed temperatures/environment retained to isolate the RGB illumination change'}
    thermal = yaml.safe_load((CONFIGS/'factory_inspection_thermal.yaml').read_text(encoding='utf-8'))
    for surface in manifest['surfaces']:
        name = surface['temperature_target_id']
        thermal['targets'][name] = {'mode':'fixed','temperature_c':surface['temperature_c']}
        thermal['bindings'].append({'name':name,'prim_path':surface['usd_prim_path'],
            'include_children':False,'target':name,'material':surface['ir_material']})
    for name, value in [('scene',scene),('task',task),('thermal',thermal)]:
        path = CONFIGS/f'model02_weaklight_{name}.yaml'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(value,allow_unicode=True,sort_keys=False),encoding='utf-8')
    (CONFIGS/'model02_weaklight_display.yaml').write_bytes((CONFIGS/'factory_inspection_display.yaml').read_bytes())
    print(json.dumps({'new_scene':'configs/model02_weaklight_scene.yaml',
                      'fixture_thermal_bindings':len(manifest['surfaces']),
                      'lighting_preset':'low_light'},ensure_ascii=False))


if __name__ == '__main__':
    main()
