"""Editable v4 thermal snapshot; compile into existing target/material formats."""
import argparse,copy,csv,hashlib,json,shutil
from pathlib import Path
import yaml
from scripts import model02_authoring as base
from scripts.thermal_config import _UniqueKeyLoader,load_thermal_config

ROOT=Path(__file__).resolve().parents[1];ASSET=ROOT/'factory/model02_thermal_v4';PACKAGE=ASSET/'configuration'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_night_environment(template,condition,weather_file):
    """Reuse the established weather fixture so runtime and authoring agree."""
    from scripts.model02_migration import night_weather
    text,start=night_weather(condition)
    environment=copy.deepcopy(template)
    environment['scene'].update(name='model02_v4_operating_night',
        description='Estimated dry operating night; air/wind/time from v4 authoring, prescribed surfaces require separate review',
        weather_file=str(weather_file),start_utc=start)
    return environment,text


def create():
    PACKAGE.mkdir(exist_ok=False)
    shutil.copytree(ROOT/'factory/model02_thermal_v3/configuration/materials',PACKAGE/'materials')
    concrete=yaml.safe_load((PACKAGE/'materials/concrete.yaml').read_text(encoding='utf-8'))
    for name,description,density,capacity,k,thickness in [
        ('paving_concrete','Designed concrete paving units;600mm layout, no product measurements',2300,880,1.4,.06),
        ('paving_cement_joint','Estimated dry cement-sand filled paving/control joint; concrete optical proxy, no product measurements',2000,840,.9,.008)]:
        data=copy.deepcopy(concrete);m=data['material'];m.update(name=name,description=description,source='estimated',reference='Project design assumption; concrete spectral/angular proxy inherited, not calibrated')
        m['thermal'].update(density_kg_m3=density,specific_heat_j_kgk=capacity,conductivity_w_mk=k,thickness_m=thickness)
        (PACKAGE/'materials'/f'{name}.yaml').write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')
    rows=json.loads((ASSET/'surfaces.json').read_text(encoding='utf-8'))['surfaces']
    old=yaml.safe_load((ROOT/'factory/model02_thermal_v3/configuration/thermal_authoring.yaml').read_text(encoding='utf-8'))
    data={'schema_version':1,'phase':'prebinding','condition':old['condition'],
        'source':{'blend_file':'../model02_thermal_v4.blend','blend_sha256':sha(ASSET/'model02_thermal_v4.blend'),
                  'surface_inventory':'../surfaces.json','surface_inventory_sha256':sha(ASSET/'surfaces.json')},
        'basis':'Editable prescribed static dry night snapshot. All temperature estimates and design assumptions, no field calibration or heat-transfer solver.',
        'targets':{},'surfaces':{}}
    for r in rows:
        data['targets'][r['temperature_target_id']]={'mode':'fixed','temperature_c':r['temperature_c']}
        data['surfaces'][r['surface_id']]={'object':r['blender_object'],'target':r['temperature_target_id'],'material':r['ir_material']}
    (PACKAGE/'thermal_authoring.yaml').write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False,width=120),encoding='utf-8')
    return compile_configuration()


def compile_configuration():
    data=yaml.load((PACKAGE/'thermal_authoring.yaml').read_text(encoding='utf-8'),Loader=_UniqueKeyLoader)
    for relative,key in [('model02_thermal_v4.blend','blend_sha256'),('surfaces.json','surface_inventory_sha256')]:
        if sha(ASSET/relative)!=data['source'][key]:raise ValueError('Source changed: '+relative)
    rows=json.loads((ASSET/'surfaces.json').read_text(encoding='utf-8'))['surfaces'];by_id={r['surface_id']:r for r in rows}
    base.PACKAGE=PACKAGE
    library=base.merged_material_library()
    temps=base.validate_authoring(data,set(by_id),set(library));objects={}
    for sid,item in data['surfaces'].items():
        original=by_id[sid]
        if (item['object'],item['target'])!=(original['blender_object'],original['temperature_target_id']):raise ValueError('Identity changed: '+sid)
        objects[item['object']]={'surface_id':sid,'target':item['target'],'material':item['material'],'temperature_c':temps[item['target']]}
    targets={'schema_version':1,'defaults':{'target':next(iter(data['targets'])),'material':'concrete'},'targets':data['targets'],'bindings':[]}
    (PACKAGE/'temperature_targets.yaml').write_text(yaml.safe_dump(targets,allow_unicode=True,sort_keys=False),encoding='utf-8')
    assert len(load_thermal_config(PACKAGE/'temperature_targets.yaml').targets)==len(rows)
    compiled={'authoring_sha256':sha(PACKAGE/'thermal_authoring.yaml'),'material_library_sha256':library.content_hash(),
        'objects':objects,'condition':data['condition'],'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in
            [ASSET/'model02_thermal_v4.blend',ASSET/'surfaces.json',PACKAGE/'thermal_authoring.yaml',*sorted((PACKAGE/'materials').glob('*.yaml'))]},
        'material_directories':[(PACKAGE/'materials').relative_to(ROOT).as_posix()]}
    (PACKAGE/'blender_parameters.json').write_text(json.dumps(compiled,ensure_ascii=False,indent=2),encoding='utf-8')
    with (PACKAGE/'parameter_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['surface_id','component','region','object','target','material','temperature_c','estimate_range_c','temperature_basis','material_basis','condition','status'])
        writer.writeheader()
        for sid,item in data['surfaces'].items():
            r=by_id[sid];material=library[item['material']]
            writer.writerow(dict(surface_id=sid,component=r['component'],region=r['region'],**item,temperature_c=temps[item['target']],
                estimate_range_c=json.dumps(r.get('temperature_estimate_range_c',[temps[item['target']],temps[item['target']]])),
                temperature_basis=r['temperature_basis'],material_basis=material.spec.reference,condition=json.dumps(data['condition'],ensure_ascii=False),
                status='Estimated static simulation snapshot; interval is scenario assumption, not measurement uncertainty'))
    report={'result':'PASS_V4_PARAMETER_CONFIGURATION','surfaces':len(objects),'unique_targets':len(temps),'material_count':len(library),
        'temperature_range_c':[min(temps.values()),max(temps.values())],'condition':data['condition'],'authoring_sha256':compiled['authoring_sha256'],
        'material_library_sha256':compiled['material_library_sha256'],'measured':False,'heat_transfer_solved':False}
    material_parameters={}
    for name,material in library.items():
        band=material.band_properties('lwir')
        material_parameters[name]={'specification':material.spec.model_dump(mode='json'),
            'lwir_band':{'emissivity':band.emissivity,'reflectance':band.reflectance,'transmittance':band.transmittance},
            'usage':'Library/product proxy or explicit project estimate; not source-site calibration. Prescribed temperature does not solve conduction.'}
    (PACKAGE/'material_parameter_basis.json').write_text(json.dumps(material_parameters,ensure_ascii=False,indent=2),encoding='utf-8')
    (PACKAGE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--create',action='store_true');a=p.parse_args()
    create() if a.create else compile_configuration()
