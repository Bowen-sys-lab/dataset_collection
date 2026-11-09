"""Publish source completion only from actual export, binding and native view evidence."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'


def main():
    from scripts.model02_full_factory_registry import validate_registry
    registry_path=OUT/'source_registry.json';registry=json.loads(registry_path.read_text());inventory=json.loads((ROOT/'validation/full-factory/inventory.json').read_text())
    native=json.loads((ROOT/'validation/full-factory/final/summary.json').read_text());assert native['result']=='PASS_V6_NATIVE_MULTISCALE_VIEWS' and native['coverage']['complete']
    exports=json.loads((OUT/'export_manifest.json').read_text());assert exports['result']=='PASS_NATIVE_V6_EXPORT' and exports['source_objects_preserved']==865
    manifest=json.loads((OUT/'isaac_manifest.json').read_text());assert manifest['surface_count']==len(manifest['surfaces'])
    grouped=collections.defaultdict(list);batches=collections.defaultdict(list);audit=collections.defaultdict(list);replaced=[]
    for path in sorted((OUT/'batches').glob('*.json')):
        batch=json.loads(path.read_text());proof=json.loads((OUT/'usd'/batch['label']/'export_manifest.json').read_text());assert proof['result']=='PASS_NATIVE_V6_BATCH_EXPORT'
        replaced.extend(batch['replaced_parts']);sources={c['source_object'] for c in batch['geometry_checks']}
        for source in sources:batches[source].append(batch['label'])
        for item in proof['source_face_audit']:audit[item['source_part']].append({'manifest':'usd/'+batch['label']+'/export_manifest.json','audit':item})
    assert len(replaced)==len(set(replaced)),'A source part was replaced in multiple batches'
    for row in manifest['surfaces']:grouped[row['source_object']].append(row)
    for source in registry['sources']:
        name=source['source_object'];rows=grouped[name];assert rows,('No actual source output',name)
        if source['render_visible_original']:assert name in replaced,('Raw source not processed',name)
        source['outputs']=[{'surface_id':r['surface_id'],'target':r['target'],'component':r['component'],'usd_prim_path':r['usd_prim_path']} for r in rows]
        source['status']='validated';source['batches']=batches[name]
        source['preservation_evidence']={'raw_geometry_hash_verified_in':'blender_assembly_manifest.json','source_face_audit_index':'source_face_audits.json',
            'inherited_provenance':'Existing v4/v5 component chain retained; preserved-part local polygon indices are not relabelled as raw source polygon indices',
            'view_scope':'Native per-zone role overall/mid/near coverage; individual source objects are not each claimed photographed'}
    result=validate_registry(inventory,registry);assert result['complete']
    (OUT/'source_face_audits.json').write_text(json.dumps(dict(audit),indent=2),encoding='utf-8')
    registry_path.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    (OUT/'registry_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
