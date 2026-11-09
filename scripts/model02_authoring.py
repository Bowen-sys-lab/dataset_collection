"""Editable v3 parameters before actual USD binding; no automatic heat solver."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml
from scripts.thermal_config import _UniqueKeyLoader

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'factory/model02_thermal_v3'
PACKAGE=ASSET/'configuration'
VENDOR=ROOT/'third_party/isaac-thermal-camera-simulation'


def finite(value,name):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError(name+': expected finite number')
    return float(value)


def resolve_temperature(spec,air):
    mode=spec.get('mode')
    expected={'fixed':{'mode','temperature_c'},'ambient':{'mode'},'offset':{'mode','offset_c'}}
    if mode not in expected or set(spec)!=expected[mode]:raise ValueError('Unsupported target fields/mode')
    temp=(finite(spec['temperature_c'],'temperature_c') if mode=='fixed' else
          air+finite(spec['offset_c'],'offset_c') if mode=='offset' else air)
    if temp<=-273.15:raise ValueError('Temperature must exceed absolute zero')
    return temp


def validate_authoring(data,surface_ids,material_names):
    if data.get('schema_version')!=1 or data.get('phase')!='prebinding':raise ValueError('Expected prebinding schema1')
    if set(data)-{'schema_version','phase','condition','targets','surfaces','source','basis'}:raise ValueError('Unknown authoring fields')
    condition=data['condition'];air=finite(condition['air_temperature_c'],'air temperature')
    if air<=-273.15 or finite(condition['indoor_temperature_c'],'indoor temperature')<=-273.15:raise ValueError('Air below absolute zero')
    if finite(condition['wind_m_s'],'wind')<0 or finite(condition['hours_after_sunset'],'night time')<0:raise ValueError('Negative wind/night duration')
    if condition['weather']!='dry' or condition['equipment_state']!='operating':raise ValueError('This approved snapshot is dry/operating; other scenarios need an explicit parameter review')
    if set(data['surfaces'])!=set(surface_ids):raise ValueError('Missing or unknown surface IDs')
    objects=set();used=set()
    resolved={name:resolve_temperature(spec,air) for name,spec in data['targets'].items()}
    for surface,item in data['surfaces'].items():
        if set(item)!={'object','target','material'}:raise ValueError('Invalid surface assignment fields: '+surface)
        if not isinstance(item['object'],str) or not item['object'] or item['object'] in objects:raise ValueError('Invalid/repeated Blender object')
        if item['target'] not in resolved or item['target'] in used:raise ValueError('Missing or shared temperature target')
        if item['material'] not in material_names:raise ValueError('Unknown material: '+str(item['material']))
        objects.add(item['object']);used.add(item['target'])
    if used!=set(resolved):raise ValueError('Unused targets')
    return resolved


def material_library():
    sys.path.insert(0,str(VENDOR/'src'))
    from irsim.materials.library import MaterialLibrary
    return MaterialLibrary.load(PACKAGE/'materials',VENDOR/'data')


def merged_material_library():
    """Keep existing background/drone materials when adding the three new IDs."""
    custom=material_library()
    from irsim.materials.library import MaterialLibrary
    base=MaterialLibrary.load(data_dir=VENDOR/'data')
    for name in set(base)&set(custom):
        if base[name].content_hash()!=custom[name].content_hash():raise ValueError('Unexpected override of existing material: '+name)
    return MaterialLibrary({**dict(base),**dict(custom)})


def create():
    PACKAGE.mkdir(exist_ok=False)
    (PACKAGE/'materials').mkdir()
    original=json.loads((ASSET/'surfaces.json').read_text(encoding='utf-8'))['surfaces']
    basis=json.loads((ASSET/'thermal_material_basis.json').read_text(encoding='utf-8'))
    sources={}
    for name in sorted({r['ir_material'] for r in original}):
        origin='aircraft_aluminium_painted' if name in ['industrial_painted_steel','painted_aluminium'] else name
        if name!='mineral_wool':
            path=VENDOR/'configs/materials'/f'{origin}.yaml'
            data=yaml.safe_load(path.read_text(encoding='utf-8'))
            sources[name]={'source_file':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            if name!=origin:
                data=copy.deepcopy(data);data['material']['name']=name
                data['material']['source']='estimated'
                data['material']['reference']='Project estimate; paint optical proxy inherited from '+origin+'; not calibrated'
                if name=='industrial_painted_steel':
                    data['material']['thermal'].update(density_kg_m3=7850,specific_heat_j_kgk=470,conductivity_w_mk=45,thickness_m=.004)
                    data['material']['description']='Painted steel: explicit steel substrate, paint optical proxy. 4mm nominal thickness is not source BIM; prescribed surface temperatures do not solve conduction.'
        else:
            sources[name]={'source_file':'thermal_material_basis.json','status':'estimated, dry mineral wool; no measurement'}
            data={'schema_version':2,'material':{'name':name,'source':'estimated','surface_treatment':'as_manufactured',
                'reference':'Project dry mineral-wool estimate; validate against selected product when available',
                'description':'Opaque dry insulation, nominal30mm. Static prescribed temperatures, not heat-transfer simulation.',
                'thermal':{'density_kg_m3':100,'specific_heat_j_kgk':840,'conductivity_w_mk':.04,'thickness_m':.03,'solar_absorptivity':.6},
                'optical':{'emissivity_per_band':{'nir':.6,'swir':.7,'mwir':.95,'lwir':.95},
                    'roughness_per_band':{'nir':.9,'swir':.9,'mwir':.9,'lwir':.9},'angular_model':{'type':'empirical','a':.05,'p':4.}}}}
        (PACKAGE/'materials'/f'{name}.yaml').write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')
    condition={k:basis['condition'][k] for k in ['air_temperature_c','indoor_temperature_c','wind_m_s','hours_after_sunset','weather','equipment_state']}
    config={'schema_version':1,'phase':'prebinding','source':{
        'blend_file':'../model02_thermal_v3.blend','blend_sha256':hashlib.sha256((ASSET/'model02_thermal_v3.blend').read_bytes()).hexdigest(),
        'surface_inventory':'../surfaces.json','surface_inventory_sha256':hashlib.sha256((ASSET/'surfaces.json').read_bytes()).hexdigest()},
        'basis':'Explicit dry low-wind night operating snapshot; estimates, not field measurements. No automatic temperature generator. No USD path is guessed.',
        'condition':condition,'targets':{},'surfaces':{}}
    for row in original:
        config['targets'][row['temperature_target_id']]={'mode':'fixed','temperature_c':row['temperature_c']}
        config['surfaces'][row['surface_id']]={'object':row['blender_object'],'target':row['temperature_target_id'],'material':row['ir_material']}
    (PACKAGE/'thermal_authoring.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False,width=120),encoding='utf-8')
    (PACKAGE/'material_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    return compile_configuration()


def compile_configuration():
    path=PACKAGE/'thermal_authoring.yaml'
    data=yaml.load(path.read_text(encoding='utf-8-sig'),Loader=_UniqueKeyLoader)
    raw=ASSET/'surfaces.json'
    if hashlib.sha256(raw.read_bytes()).hexdigest()!=data['source']['surface_inventory_sha256']:raise ValueError('Source inventory changed; do not silently reuse mappings')
    if hashlib.sha256((ASSET/'model02_thermal_v3.blend').read_bytes()).hexdigest()!=data['source']['blend_sha256']:raise ValueError('Original v3 blend changed')
    source=json.loads(raw.read_text(encoding='utf-8'))['surfaces']
    library=material_library();resolved=validate_authoring(data,{r['surface_id'] for r in source},set(library))
    by_id={r['surface_id']:r for r in source};objects={}
    for sid,item in data['surfaces'].items():
        if item['object']!=by_id[sid]['blender_object'] or item['target']!=by_id[sid]['temperature_target_id']:raise ValueError('Stable identity changed: '+sid)
        objects[item['object']]={'surface_id':sid,'target':item['target'],'material':item['material'],'temperature_c':resolved[item['target']]}
    optical={}
    for name in library:
        material=library[name];band=material.band_properties('lwir')
        eps,rho,tau=band.emissivity,band.reflectance,band.transmittance
        assert abs(eps+rho+tau-1.)<1e-6
        optical[name]={'lwir_emissivity':eps,'lwir_reflectance':rho,'lwir_transmittance':tau,'source':material.spec.source}
    merged=merged_material_library()
    from irsim.materials.table import MaterialTable
    table=MaterialTable.from_library(merged,'lwir')
    # Exercise the existing temperature schema, without inventing USD bindings.
    runtime={'schema_version':1,'defaults':{'target':next(iter(data['targets'])),'material':next(iter(library))},'targets':data['targets'],'bindings':[]}
    target_path=PACKAGE/'temperature_targets.yaml'
    target_path.write_text(yaml.safe_dump(runtime,allow_unicode=True,sort_keys=False),encoding='utf-8')
    from scripts.thermal_config import load_thermal_config
    assert len(load_thermal_config(target_path).targets)==len(objects)
    compiled={'phase':'prebinding','authoring_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'material_library_sha256':merged.content_hash(),'condition':data['condition'],'objects':objects}
    # Pin raw inputs too: Blender cannot import the YAML/material dependencies.
    directories=[PACKAGE/'materials',VENDOR/'configs/materials']
    inputs={path,raw,ASSET/'model02_thermal_v3.blend'}
    for directory in directories:inputs.update(directory.glob('*.yaml'))
    for material in merged.values():
        if material.spectrum is not None:inputs.add(material.spectrum.path)
        if material.n_k_path is not None:inputs.add(material.n_k_path)
    compiled['input_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    compiled['material_directories']=[p.relative_to(ROOT).as_posix() for p in directories]
    (PACKAGE/'blender_parameters.json').write_text(json.dumps(compiled,ensure_ascii=False,indent=2),encoding='utf-8')
    with (PACKAGE/'parameter_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['surface_id','component','region','object','target','material','temperature_c','temperature_basis'])
        writer.writeheader()
        for sid,item in data['surfaces'].items():
            original=by_id[sid];writer.writerow(dict(surface_id=sid,component=original['component'],region=original['region'],**item,temperature_c=resolved[item['target']],temperature_basis=original['temperature_basis']))
    report={'result':'PASS_PREBINDING_CONFIGURATION','surfaces':len(objects),'unique_targets':len(resolved),'materials':optical,
        'temperature_range_c':[min(resolved.values()),max(resolved.values())],
        'authoring_sha256':compiled['authoring_sha256'],'runtime_temperature_schema':'validated with existing load_thermal_config',
        'material_library_sha256':compiled['material_library_sha256'],'merged_material_count':len(merged),
        'material_table_slots_including_unmapped':len(table.names),
        'usd_bindings':0,'isaac_sensor_validation':False,
        'condition':data['condition'],'note':'Zero bindings intentional until actual export. This is complete parameter preparation, not a bound Isaac scene.'}
    (PACKAGE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--create',action='store_true');args=parser.parse_args()
    report=create() if args.create else compile_configuration()
    print(json.dumps({k:v for k,v in report.items() if k!='materials'},ensure_ascii=False))
