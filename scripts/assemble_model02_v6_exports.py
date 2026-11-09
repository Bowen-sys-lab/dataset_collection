"""Compose all actually exported batch paths without flattening independent surfaces."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    from pxr import Usd,UsdGeom
    from scripts.model02_configuration_freshness import verify_inputs
    config=json.loads((OUT/'configuration/blender_parameters.json').read_text());verify_inputs(ROOT,config)
    assembly=json.loads((OUT/'blender_assembly_manifest.json').read_text());assert assembly['result']=='PASS_FULL_V6_BLENDER_ASSEMBLY'
    configured=OUT/'model02_thermal_v6_configured.blend';assert sha(configured)==assembly['configured_blend_sha256']
    batches=[json.loads(p.read_text()) for p in sorted((OUT/'batches').glob('*.json'))];rows=[];layers=[];textures={}
    for batch in batches:
        manifest=json.loads((OUT/'usd'/batch['label']/'export_manifest.json').read_text());assert manifest['result']=='PASS_NATIVE_V6_BATCH_EXPORT'
        for item in manifest['usd_files']:
            assert sha(OUT/item['path'])==item['sha256'];layers.append('./'+item['path'])
        rows.extend(manifest['surfaces']);textures[batch['label']]=manifest['texture_dependencies']
    assert len({r['surface_id'] for r in rows})==len(rows)==assembly['new_surfaces']
    assert len({r['usd_prim_path'] for r in rows})==len(rows),'Exported batch path collision'
    output=OUT/'model02_v6_surfaces.usdc';assert not output.exists();stage=Usd.Stage.CreateNew(str(output));stage.GetRootLayer().subLayerPaths=layers
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    actual={str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)};assert actual=={r['usd_prim_path'] for r in rows}
    stage.GetRootLayer().Save()
    report={'result':'PASS_NATIVE_V6_EXPORT','surfaces':rows,'surface_count':len(rows),'source_objects_preserved':assembly['source_objects_preserved'],
        'inherited_targets_preserved':assembly['active_thermal_parts']-len(rows),'replaced_parts':sorted({n for b in batches for n in b['replaced_parts']}),
        'configured_blend_sha256':assembly['configured_blend_sha256'],'usd_sha256':sha(output),'texture_dependencies':textures,'batch_layers':layers}
    (OUT/'export_manifest.json').write_text(json.dumps(report,indent=2));print('V6 assembled actual USD paths',len(rows),flush=True)


if __name__=='__main__':main()
