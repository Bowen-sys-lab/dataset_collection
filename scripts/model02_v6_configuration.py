"""Explicit static v6 thermal authoring; RGB never determines temperatures."""
import copy,csv,hashlib,json,math,shutil
from pathlib import Path


def load_authoring(path):
    import yaml
    from scripts.thermal_config import _UniqueKeyLoader
    path=Path(path);data=yaml.load(path.read_text(encoding='utf-8-sig'),Loader=_UniqueKeyLoader)
    for relative in data.get('source_files',[]):
        part=yaml.load((path.parent/relative).read_text(encoding='utf-8-sig'),Loader=_UniqueKeyLoader)
        for field in ['targets','surfaces']:
            values=data.setdefault(field,{})
            if values.keys()&part[field].keys():raise ValueError('duplicate '+field+' across source files: '+relative)
            values.update(part[field])
    return data


def export_parts(batch,size=20000):
    faces={}
    for r in batch['surfaces']:faces.setdefault(r['source_part'],set()).update(r['source_face_ids'])
    total=(len(batch['surfaces'])+size-1)//size;parts=[]
    for index,start in enumerate(range(0,len(batch['surfaces']),size)):
        rows=batch['surfaces'][start:start+size];sources={r['source_part'] for r in rows}
        parts.append({'label':batch['label']+f'_p{index:03d}','batch_label':batch['label'],'part_index':index,'parts_in_batch':total,
            'surfaces':rows,'geometry_checks':[c for c in batch['geometry_checks'] if c['source_part'] in sources],
            'replaced_parts':batch['replaced_parts'],'source_face_coverage':{s:sorted(faces[s]) for s in sorted(sources)}})
    return parts


def retained_surfaces(rows,exports,replaced):
    geometry={r['surface_id']:r for r in exports};result=[]
    for r in rows:
        if r['surface_id'] in replaced:continue
        native=geometry[r['surface_id']]
        result.append(dict(r,object=r['blender_object'],target=r['temperature_target_id'],material=r['ir_material'],
            bounds_m=native['bounds'],area_m2=native['area_m2'],inherited=True,
            assumption=r.get('modeling_approximation','Preserved v4 geometry and prescribed temperature; source materials are uncalibrated proxies')))
    return result


def compile_configuration(authoring,surfaces):
    by_id={r['surface_id']:r for r in surfaces}
    if len(by_id)!=len(surfaces) or set(authoring['surfaces'])!=set(by_id):raise ValueError('Missing or duplicate surfaces')
    objects={};used=set()
    for sid,item in authoring['surfaces'].items():
        row=by_id[sid]
        if (item['object'],item['target'])!=(row['object'],row['target']):raise ValueError('Changed surface identity: '+sid)
        if item['target'] in used:raise ValueError('Shared temperature target: '+sid)
        used.add(item['target'])
        material=authoring['materials'].get(item['material'])
        if not material or not material.get('basis') or not row.get('assumption'):raise ValueError('Missing material proxy basis: '+sid)
        t=authoring['targets'].get(item['target'],{}).get('temperature_c')
        if type(t) not in (float,int) or not math.isfinite(t) or not -80.<t<400.:raise ValueError('Invalid temperature: '+sid)
        if item['object'] in objects:raise ValueError('Duplicate object')
        objects[item['object']]={'surface_id':sid,'target':item['target'],'material':item['material'],'temperature_c':float(t)}
    if used!=set(authoring['targets']):raise ValueError('Unused targets')
    return {'objects':objects}


def initial_temperature(row,source):
    """One-time documented night snapshot, editable afterwards; no random fields."""
    c=row['component'];x,y,z=row['center_m'];lo,hi=source['bounds_m'];height=max(.1,hi[2]-lo[2]);h=max(0.,z-lo[2])
    residual=.35*math.cos((x-700.)/70.)+.25*math.sin((y+700.)/90.)
    if c=='Landscape':t=19.4+.25*residual;basis='Estimated canopy radiative cooling under dry clear night; broad spatial residual only';interval=[17.,22.]
    elif c in ['PavingBrick','ConcreteRoadSlab','PavingJoint','RoadJoint','KerbAndRiser','KerbJoint']:
        from scripts.model02_paving import temperature
        joint=c in ['PavingJoint','RoadJoint','KerbAndRiser','KerbJoint'];t=temperature(x,y,'joint' if joint else 'tile')-(.3 if c=='KerbAndRiser' else .6 if c=='KerbJoint' else 0.)
        basis='Preserved v4 smooth residual-heat field and contact hierarchy; no random per-unit temperatures or new corridor seam';interval=[20.,25.]
    elif c in ['PavingAndKerb','GroundStructure','Foundation']:
        t=22.+residual+.35*math.exp(-h/2.);basis='Paving/concrete residual stored daytime heat, slow site variation and contact at base';interval=[20.,25.]
    elif c in ['PavingJoint','RoadJoint']:
        t=21.7+residual;basis='Shallow dry filled joint follows neighbouring paving with small cooling difference';interval=[20.,25.]
    elif source['role']=='ground':
        t=21.+.2*residual+.35*math.exp(-h/2.);basis='Passive ground/facility cover or ancillary metal; weak residual contact heat, no assumed process operation';interval=[19.,24.]
    elif c in ['Wall','Roof','Soffit','WindowGlass','WindowFrame','DoorFrame','DoorPanel','DoorGlass']:
        base={'Wall':22.,'Roof':20.,'Soffit':21.8,'WindowGlass':21.1,'WindowFrame':20.8,'DoorFrame':21.1,'DoorPanel':21.8,'DoorGlass':21.1}[c]
        t=base+.25*residual+.45*math.exp(-h/2.)-.2*h/height
        basis='Indoor24C/outdoor20C prescribed night: enclosure leakage, sky exposure and base contact; no solved heat transfer';interval=[18.,25.]
    elif source['role']=='structure':
        t=20.8+.2*residual+.4*math.exp(-h/1.5);basis='Passive structural metal approaches air20C with weak contact residual';interval=[19.,24.]
    elif source['role']=='building':
        t=21.5+.2*residual+.4*math.exp(-h/2.);basis='Building ancillary metal/tread components: modest indoor/contact residual';interval=[19.,25.]
    elif c=='EquipmentBase':
        t=23.5+.2*residual;basis='Equipment contact/support near concrete; weak operating heat conducted into base';interval=[21.,28.]
    elif source['role']=='pipework':
        t=29.+.45*math.exp(-h/4.)+.25*residual;basis='Unmeasured operating pipe exterior: modest warm-service scenario; not a claim about actual process fluid or insulation';interval=[23.,38.]
    else:
        t=27.5+1.2*math.sin(math.pi*min(1.,h/height))+.25*residual
        basis='Unmeasured normal equipment exterior: assumed moderately warm continuous operation; slow height/contact variation, no inferred service';interval=[23.,35.]
    return round(t,3),interval,basis


def create():
    import yaml
    root=Path(__file__).resolve().parents[1];out=root/'factory/model02_thermal_v6';folder=out/'configuration';folder.mkdir(exist_ok=True)
    if (folder/'thermal_authoring.yaml').exists():raise ValueError('Authoring already exists; compile saved settings without --create')
    registry=json.loads((out/'source_registry.json').read_text());sources={r['source_object']:r for r in registry['sources']}
    batches=[json.loads(p.read_text()) for p in sorted((out/'batches').glob('*.json'))]
    rows=[r for b in batches for r in b['surfaces']];replaced={n for b in batches for n in b['replaced_parts']}
    for r in rows:r['source_face_index_space']='raw_source_polygon' if r['source_part']==r['source_object'] else 'preserved_part_polygon'
    old=json.loads((root/'factory/model02_thermal_v4/surfaces.json').read_text(encoding='utf-8'))['surfaces']
    exports=json.loads((root/'factory/model02_thermal_v4/isaac_manifest.json').read_text(encoding='utf-8'))['surfaces']
    rows.extend(retained_surfaces(old,exports,replaced))
    from scripts.model02_full_factory_registry import validate_authoring_coverage
    validate_authoring_coverage(registry,batches,{r['source_object'] for r in rows if r.get('inherited')})
    material_dir=folder/'materials'
    if not material_dir.exists():shutil.copytree(root/'factory/model02_thermal_v4/configuration/materials',material_dir)
    materials={}
    for name in {r['material'] for r in rows}:
        path=material_dir/f'{name}.yaml'
        if not path.exists():raise ValueError('Material absent: '+name)
        materials[name]={'file':str(path),'basis':'Existing library spectral/angular proxy; source composition/finish not measured. Bare aluminium is an optical stand-in for unspecified bare metal, not an alloy identification.'}
    data={'schema_version':1,'condition':{'air_temperature_c':20.,'indoor_temperature_c':24.,'wind_m_s':1.,'hours_after_sunset':2.,'weather':'dry','equipment_state':'operating'},
        'basis':'Estimated static simulation snapshot; source process, composition and weather not measured; no automatic temperature solver.',
        'materials':materials,'targets':{},'surfaces':{}}
    for r in rows:
        if r.get('inherited'):
            t=r['temperature_c'];interval=r.get('temperature_estimate_range_c',[t,t]);basis=r.get('temperature_basis','Preserved explicit v4 estimate')
        else:t,interval,basis=initial_temperature(r,sources[r['source_object']])
        r.update(temperature_c=t,temperature_estimate_range_c=interval,temperature_basis=basis)
        data['targets'][r['target']]={'mode':'fixed','temperature_c':t}
        data['surfaces'][r['surface_id']]={k:r[k] for k in ['object','target','material']}
    # Existing grass20C is an authored boundary condition, not measured truth.
    # Join estimates gradually without changing any inherited target value.
    import numpy as np
    grass=[r for r in rows if r.get('inherited') and r['component']=='Landscape']
    if grass:
        bounds=np.array([r['bounds_m'] for r in grass]);assert all(r['temperature_c']==20. for r in grass)
        for r in rows:
            if r.get('inherited') or r['component']!='Landscape' or r['source_object']=='11_29_gc1750':continue
            box=np.array(r['bounds_m']);delta=np.maximum(np.maximum(bounds[:,0,:2]-box[1,:2],box[0,:2]-bounds[:,1,:2]),0.)
            distance=float(np.linalg.norm(delta,axis=1).min());v=min(1.,distance/160.);weight=v*v*(3.-2.*v)
            r['temperature_c']=round(20.*(1.-weight)+r['temperature_c']*weight,3)
            r['temperature_basis']+='; inherited20C boundary retained, smooth160m prescribed compatibility transition; not a conduction solution'
            data['targets'][r['target']]['temperature_c']=r['temperature_c']
    grouped={}
    for r in rows:grouped.setdefault(r['source_object'],[]).append(r)
    source_files=[];dumper=getattr(yaml,'CSafeDumper',yaml.SafeDumper)
    for source,items in sorted(grouped.items()):
        items.sort(key=lambda r:(r['component'],r['center_m'][2],r['center_m'][0],r['center_m'][1],r['surface_id']))
        for start in range(0,len(items),2048):
            part=items[start:start+2048];relative=f'sources/{source}/part_{start//2048:03d}.yaml';path=folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
            payload={'source_object':source,'surfaces':{r['surface_id']:data['surfaces'][r['surface_id']] for r in part},'targets':{r['target']:data['targets'][r['target']] for r in part}}
            path.write_text(yaml.dump(payload,Dumper=dumper,sort_keys=False,allow_unicode=True),encoding='utf-8');source_files.append(relative)
            for r in part:r['authoring_file']=relative
    index={k:data[k] for k in ['schema_version','condition','basis','materials']};index['source_files']=source_files
    (out/'surfaces.json').write_text(json.dumps({'version':6,'surfaces':rows},indent=2),encoding='utf-8')
    (folder/'thermal_authoring.yaml').write_text(yaml.dump(index,Dumper=dumper,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return compile_saved()


def compile_saved():
    import yaml
    from scripts.project_materials import load_project_material_library
    root=Path(__file__).resolve().parents[1];out=root/'factory/model02_thermal_v6';folder=out/'configuration'
    from scripts.thermal_config import _UniqueKeyLoader
    from scripts.model02_v4_configuration import compile_night_environment
    authoring=folder/'thermal_authoring.yaml';data=load_authoring(authoring);rows=json.loads((out/'surfaces.json').read_text())['surfaces']
    compiled=compile_configuration(data,rows);library=load_project_material_library(folder/'materials')
    if set(data['materials'])-set(library):raise ValueError('Unknown material in library')
    environment_path=root/'configs/model02_v6_night_environment.yaml';weather_path=folder/'night_weather.csv'
    template=yaml.safe_load((root/'configs/model02_v4_night_environment.yaml').read_text(encoding='utf-8'))
    environment,weather=compile_night_environment(template,data['condition'],weather_path)
    environment['scene']['name']='model02_v6_operating_night';weather_path.write_text(weather,encoding='utf-8')
    environment_path.write_text(yaml.safe_dump(environment,sort_keys=False),encoding='utf-8')
    authoring_files=[authoring,*[folder/p for p in data.get('source_files',[])]]
    files=[*authoring_files,environment_path,weather_path,out/'surfaces.json',root/'factory/model02_thermal_v5/model02_thermal_v5_configured.blend',*sorted((folder/'materials').glob('*.yaml')),*sorted((out/'batches').glob('*.json')),*sorted((out/'assets').glob('*.blend'))]
    def digest(p):
        with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    compiled.update(condition=data['condition'],authoring_sha256=digest(authoring),material_library_sha256=library.content_hash(),
        input_sha256={p.relative_to(root).as_posix():digest(p) for p in files},material_directories=[(folder/'materials').relative_to(root).as_posix()])
    compiled['authoring_input_sha256']={p.relative_to(root).as_posix():compiled['input_sha256'][p.relative_to(root).as_posix()] for p in authoring_files}
    (folder/'blender_parameters.json').write_text(json.dumps(compiled,indent=2),encoding='utf-8')
    batch_folder=folder/'batches';batch_folder.mkdir(exist_ok=True)
    jobs=[]
    for path in sorted((out/'batches').glob('*.json')):
        batch=json.loads(path.read_text());asset=out/batch['asset_file']
        asset_sha=compiled['input_sha256'][asset.relative_to(root).as_posix()]
        for part in export_parts(batch):
            params=dict(part,asset_file=batch['asset_file'],asset_sha256=asset_sha,authoring_sha256=compiled['authoring_sha256'],authoring_input_sha256=compiled['authoring_input_sha256'],
                objects={r['object']:compiled['objects'][r['object']] for r in part['surfaces']})
            name=part['label']+'.json';(batch_folder/name).write_text(json.dumps(params,indent=2),encoding='utf-8');jobs.append({'label':part['label'],'batch_label':batch['label'],'parameters':'batches/'+name})
    (folder/'export_jobs.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    basis={}
    for name,material in library.items():
        band=material.band_properties('lwir');basis[name]={'specification':material.spec.model_dump(mode='json'),
            'lwir_emissivity':band.emissivity,'lwir_reflectance':band.reflectance,'lwir_transmittance':band.transmittance,
            'usage':'Library/product optical proxy or project estimate; not measured source-site composition or finish'}
    (folder/'material_parameter_basis.json').write_text(json.dumps(basis,indent=2),encoding='utf-8')
    with (folder/'parameter_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        manifest_path=out/'isaac_manifest.json';paths={r['surface_id']:r['usd_prim_path'] for r in json.loads(manifest_path.read_text())['surfaces']} if manifest_path.exists() else {}
        writer=csv.writer(f);writer.writerow(['surface_id','source_object','source_part','component','region','center_m','bounds_m','authoring_file','usd_prim_path','target','ir_material','temperature_c','estimate_range_c','temperature_basis','material_assumption'])
        for r in rows:
            item=compiled['objects'][r['object']];writer.writerow([r['surface_id'],r.get('source_object'),r['source_part'],r['component'],r['region'],r['center_m'],r['bounds_m'],r.get('authoring_file','thermal_authoring.yaml'),paths.get(r['surface_id'],''),item['target'],item['material'],item['temperature_c'],r['temperature_estimate_range_c'],r['temperature_basis'],r['assumption']])
    manifest_path=out/'isaac_manifest.json'
    if manifest_path.exists():
        from scripts.model02_migration import bind_surface_rows
        from scripts.thermal_config import load_thermal_config
        manifest=json.loads(manifest_path.read_text());bindings=bind_surface_rows(manifest['surfaces'],data)
        runtime_path=root/'configs/model02_v6_thermal.yaml';runtime=yaml.load(runtime_path.read_text(encoding='utf-8'),Loader=_UniqueKeyLoader)
        paths={b['prim_path'] for b in bindings};runtime['bindings']=[b for b in runtime['bindings'] if b['prim_path'] not in paths]+bindings
        runtime['targets'].update(data['targets']);runtime_path.write_text(yaml.safe_dump(runtime,sort_keys=False),encoding='utf-8');load_thermal_config(runtime_path)
    print('V6 explicit configuration',len(rows),'independent targets');return compiled


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--create',action='store_true');args=parser.parse_args()
    create() if args.create else compile_saved()
