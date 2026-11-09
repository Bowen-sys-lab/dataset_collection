"""Preserve metal decks/covers and masonry instead of converting them to paving."""
import bpy,json,sys,shutil,numpy as np
from pathlib import Path
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.refine_model02_v6_ground import author
from scripts.blender.model02_full_factory_authoring import OUT,save_batch,refresh_index,authored_world
from scripts.blender.model02_full_factory_common import material_texture
from scripts.blender.model02_route_refinement import BUILDING_BASIS


def main():
    bpy.context.scene['v6_checkpoint_name']='model02_v6_ground_working.blend'
    registry={r['source_object']:r for r in json.loads((OUT/'source_registry.json').read_text())['sources']}
    history=OUT/'batch_history';history.mkdir(exist_ok=True)
    for path in sorted((OUT/'batches').glob('ground_*.json')):
        batch=json.loads(path.read_text())
        if batch.get('reviewed_material_boundaries'):continue
        selected=set()
        for c in batch['geometry_checks']:
            o=bpy.data.objects[c['source_part']];textures={material_texture(m) for m in o.data.materials}
            if o.get('component') in ['PavingAndKerb','PlatformDeck','RailAndSupport','RailAndAccess'] or c['source_object']=='11_29_gc052' or textures&{'11-29-gc013.jpg','11-29-gc016.jpg','11-29-gc030.jpg','11-29-gc012.jpg'}:selected.add(o.name)
        removed={r['object'] for r in batch['surfaces'] if r['source_part'] in selected};objects=[bpy.data.objects[n] for n in removed]
        bpy.data.batch_remove([*objects,*(o.data for o in objects)]);refresh_index()
        rows=[r for r in batch['surfaces'] if r['source_part'] not in selected];checks=[c for c in batch['geometry_checks'] if c['source_part'] not in selected]
        for name in sorted(selected):
            source=bpy.data.objects[name];r=registry[source.get('source_object',name)]
            print('[v6] ground material review',name,flush=True);added,check=author(source,r);rows.extend(added);checks.append(check)
        if not (history/path.name).exists():shutil.copy2(path,history/path.name)
        parameters=json.loads((ROOT/'factory/model02_thermal_v5/texture_parameters.json').read_text())
        uv_fixed=0
        for r in rows:
            o=bpy.data.objects[r['object']]
            if r['source_object']=='11_29_gc1750' or not any(m and m.name.startswith('V5_') for m in o.data.materials):continue
            c=r['component'];kind='grass' if c=='Landscape' else 'joint' if c in ['RoadJoint','PavingJoint','KerbJoint'] else 'brick' if c=='PavingBrick' else 'concrete'
            points=np.array([v.co[:] for v in o.data.vertices]);matrix=np.array(authored_world(o));world=(np.c_[points,np.ones(len(points))]@matrix.T)[:,:3]
            uv=((world-np.array([700.,-845.,0.]))@BUILDING_BASIS.T)[:,:2]/parameters[kind]['texture_span_m']
            indices=np.empty(len(o.data.loops),dtype='int32');o.data.loops.foreach_get('vertex_index',indices)
            o.data.uv_layers.active.data.foreach_set('uv',uv[indices].astype('float32').reshape(-1));uv_fixed+=1
        save_batch(batch['label'],rows,batch['replaced_parts'],checks)
        updated=dict(batch,surfaces=rows,geometry_checks=checks,reviewed_material_boundaries=True);path.write_text(json.dumps(updated,indent=2))
        print('[v6] UV spans match inherited v5',uv_fixed,'meshes',flush=True)


if __name__=='__main__':main()
