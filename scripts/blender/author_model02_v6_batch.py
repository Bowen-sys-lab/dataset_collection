"""Author one registered zone from immutable v5 source data, without a full-scene graph."""
import argparse,bpy,hashlib,importlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['buildings','equipment','ground'],required=True);parser.add_argument('--zone',required=True);parser.add_argument('--review',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);label=args.family+'_'+args.zone
    path=OUT/'batches'/(label+'.json')
    if path.exists() and not args.review:raise ValueError('Batch already saved: '+label)
    roles={'buildings':{'building'},'equipment':{'equipment','pipework','structure'},'ground':{'ground'}}[args.family]
    registry=json.loads((OUT/'source_registry.json').read_text());sources={r['source_object']:r for r in registry['sources']}
    selected={name:r for name,r in sources.items() if r['role'] in roles and r['zone_id']==args.zone and r['render_visible_original']}
    old=json.loads((ROOT/'factory/model02_thermal_v4/surfaces.json').read_text(encoding='utf-8'))['surfaces']
    for row in old:
        if row['source_object'] not in sources:continue
        source=sources[row['source_object']]
        if source['role'] in roles and source['zone_id']==args.zone and (row['thermal_scope']=='background_retained' or row['component']=='PreservedRoadBackground'):selected[row['blender_object']]=source
    assert selected,('No sources in registered batch',label)
    baseline=ROOT/'factory/model02_thermal_v5/model02_thermal_v5_configured.blend'
    with baseline.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='d4e2dc48f85af649c6a1aa5a8b556b38c88cb69ff5a2aa37315df7eb500761f6'
    bpy.data.batch_remove([*bpy.data.objects,*bpy.data.meshes,*bpy.data.materials,*bpy.data.collections])
    with bpy.data.libraries.load(str(baseline),link=False) as (source,destination):
        assert set(selected)|{'InspectionZone01'}<=set(source.objects)
        destination.objects=sorted(set(selected)|{'InspectionZone01'})
        if args.family=='ground':destination.materials=[n for n in source.materials if n.startswith('V5_')]
    objects={o.name:o for o in destination.objects}
    for name in selected:bpy.context.scene.collection.objects.link(objects[name])
    bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1.
    from scripts.blender import model02_full_factory_authoring as authoring
    from scripts.blender.model02_inventory import snapshot_scene
    actual={r['name']:r['geometry_uv_material_normal_sha256'] for r in snapshot_scene()['objects']}
    assert all(actual[n]==r['source_geometry_sha256'] for n,r in selected.items() if n in sources)
    bpy.context.scene['v6_batch_asset_mode']=True
    author=importlib.import_module('scripts.blender.refine_model02_v6_'+args.family).author
    rows=[];checks=[]
    for name,r in sorted(selected.items()):
        print('[v6] start',label,name,flush=True);added,check=author(objects[name],r);rows.extend(added);checks.append(check)
        print('[v6] authored',name,len(added),'regions',flush=True)
    if path.exists():
        history=OUT/'batch_history';history.mkdir(exist_ok=True);shutil.copy2(path,history/(path.stem+'_'+str(time.time_ns())+'.json'))
    authoring.save_batch(label,rows,sorted(selected),checks)


if __name__=='__main__':main()
