"""Compose actual v6 export paths over v5; preserve inherited rig and tasks."""
import copy,hashlib,json,os,re,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def dump(p,v):p.write_text(yaml.safe_dump(v,sort_keys=False,allow_unicode=True),encoding='utf-8')


def main():
    from pxr import Usd,UsdGeom
    from scripts.thermal_config import load_thermal_config
    from scripts.scene_config import load_scene_config
    from scripts.model02_migration import bind_surface_rows
    export=json.loads((OUT/'export_manifest.json').read_text());assert export['result']=='PASS_NATIVE_V6_EXPORT'
    newrows=json.loads((OUT/'surfaces.json').read_text())['surfaces'];lookup={r['surface_id']:r for r in newrows}
    rows=[dict(lookup[r['surface_id']],**r) for r in export['surfaces']]
    inherited=json.loads((ROOT/'factory/model02_thermal_v4/isaac_manifest.json').read_text(encoding='utf-8'))
    original=Usd.Stage.Open(str(ROOT/'factory/model02_thermal_v5/model02_v5_scene.usda'))
    registry=json.loads((OUT/'source_registry.json').read_text());sources={r['source_object']:r for r in registry['sources']}
    replaced=set(export['replaced_parts']);actual_original={}
    for p in original.Traverse():
        match=re.fullmatch(r'node_1_29_gc(\d+)',p.GetName())
        if match:
            source='11_29_gc'+match.group(1)
            if source in sources:actual_original[source]=str(p.GetPath())
    output=OUT/'model02_v6_scene.usda';assert not output.exists()
    stage=Usd.Stage.CreateNew(str(output));stage.GetRootLayer().subLayerPaths=['./model02_v6_surfaces.usdc','../model02_thermal_v5/model02_v5_scene.usda']
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetTimeCodesPerSecond(60);stage.SetStartTimeCode(0);stage.SetEndTimeCode(1000000)
    disabled=[]
    for name in sorted(replaced&set(sources)):
        path=actual_original[name];assert stage.GetPrimAtPath(path).IsActive();stage.OverridePrim(path).SetActive(False);disabled.append(path)
    kept=[]
    for row in inherited['surfaces']:
        if row['surface_id'] in replaced:stage.OverridePrim(row['usd_prim_path']).SetActive(False);disabled.append(row['usd_prim_path'])
        else:kept.append(dict(lookup[row['surface_id']],**row))
    for r in rows+kept:assert stage.GetPrimAtPath(r['usd_prim_path']).IsActive(),r['surface_id']
    profile=copy.deepcopy(dict(original.GetRootLayer().customLayerData['uavLighting']))
    oldfolder=ROOT/'factory/model02_thermal_v5'
    profile['dependencies']={os.path.relpath((oldfolder/p).resolve(),OUT).replace('\\','/'):h for p,h in profile['dependencies'].items()}
    for path in [oldfolder/'model02_v5_scene.usda',OUT/'model02_v6_surfaces.usdc']:
        profile['dependencies'][os.path.relpath(path,OUT).replace('\\','/')]=sha(path)
    for relative in export['batch_layers']:
        path=(OUT/relative).resolve();profile['dependencies'][os.path.relpath(path,OUT).replace('\\','/')]=sha(path)
    for batch in export['texture_dependencies'].values():
        for part in batch.values():
            for texture in part.values():
                path=Path(texture['resolved_path']);assert sha(path)==texture['sha256']
                profile['dependencies'][os.path.relpath(path,OUT).replace('\\','/')]=texture['sha256']
    stage.GetRootLayer().customLayerData={'uavLighting':profile,'v6Factory':{'configuration':'configuration/thermal_authoring.yaml','geometry_source_objects':865,'validation':'pending native whole-site views'}}
    stage.GetRootLayer().Save()
    from scripts.model02_v6_configuration import load_authoring
    authoring=load_authoring(OUT/'configuration/thermal_authoring.yaml')
    thermal=yaml.safe_load((ROOT/'configs/model02_v4_thermal.yaml').read_text(encoding='utf-8'))
    # Remove obsolete exact bindings only when their actual geometry is inactive.
    active={str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Gprim) and UsdGeom.Imageable(p).ComputeVisibility()!='invisible'}
    thermal['bindings']=[b for b in thermal['bindings'] if b['prim_path'] in active or (b['include_children'] and any(p.startswith(b['prim_path']+'/') for p in active))]
    allpaths={r['usd_prim_path'] for r in kept+rows}
    thermal['bindings']=[b for b in thermal['bindings'] if b['prim_path'] not in allpaths]
    thermal['targets'].update(authoring['targets']);thermal['bindings'].extend(bind_surface_rows(kept+rows,authoring))
    dump(ROOT/'configs/model02_v6_thermal.yaml',thermal);loaded=load_thermal_config(ROOT/'configs/model02_v6_thermal.yaml')
    used={loaded.binding_for(p).name for p in active if loaded.binding_for(p) is not None}
    thermal['bindings']=[b for b in thermal['bindings'] if b['name'] in used]
    dump(ROOT/'configs/model02_v6_thermal.yaml',thermal);loaded=load_thermal_config(ROOT/'configs/model02_v6_thermal.yaml')
    manifest={'result':'PASS_V6_COMPOSITION_PENDING_NATIVE_ACCEPTANCE','surfaces':kept+rows,'surface_count':len(kept)+len(rows),
        'original_tasks':inherited['original_tasks'],
        'new_surface_count':len(rows),'inherited_target_count':len(kept),'inactive_replaced_paths':disabled,'actual_original_source_paths':actual_original,
        'scene_sha256':sha(output),'configured_blend_sha256':export['configured_blend_sha256'],'units':'m','up_axis':'Z','manual_success':None}
    (OUT/'isaac_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for view in ['tank','pipe','detail','pipe_detail']:
        base=ROOT/f'configs/model02_v5_{view}_scene.yaml' if view in ['tank','detail'] else ROOT/'configs/model02_v3_pipe_scene.yaml'
        config=yaml.safe_load(base.read_text(encoding='utf-8'))
        if view=='pipe_detail':config['sensors']['display_config']='model02_v4_detail_display.yaml'
        config['scene']['stage']='../factory/model02_thermal_v6/model02_v6_scene.usda';config['sensors']['thermal_config']='model02_v6_thermal.yaml'
        config['scene']['thermal_scene']='model02_v6_night_environment.yaml'
        config['sensors']['material_directory']='../factory/model02_thermal_v6/configuration/materials'
        dump(ROOT/f'configs/model02_v6_{view}_scene.yaml',config);load_scene_config(ROOT/f'configs/model02_v6_{view}_scene.yaml')
    print('V6 composed',len(rows),'new surfaces;',len(kept),'inherited targets;',len(loaded.bindings),'bindings')


if __name__=='__main__':main()
