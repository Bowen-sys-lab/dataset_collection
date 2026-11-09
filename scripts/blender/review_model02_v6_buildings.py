"""Apply measured opening review to own unreleased batches; preserve revisions."""
import bpy,json,sys,shutil
from pathlib import Path
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.refine_model02_v6_buildings import author
from scripts.blender.model02_full_factory_authoring import OUT,save_batch,refresh_index


def main():
    registry={r['source_object']:r for r in json.loads((OUT/'source_registry.json').read_text())['sources']}
    history=OUT/'batch_history';history.mkdir(exist_ok=True)
    for path in sorted((OUT/'batches').glob('buildings_*.json')):
        batch=json.loads(path.read_text())
        if batch.get('reviewed_opening_connectivity'):continue
        selected={c['source_part'] for c in batch['geometry_checks'] if c['openings']['openings'] or 'submicron_clipping_area_loss_m2' not in c}
        removed={r['object'] for r in batch['surfaces'] if r['source_part'] in selected}
        objects=[bpy.data.objects[n] for n in removed];meshes=[o.data for o in objects]
        bpy.data.batch_remove([*objects,*meshes])
        refresh_index()
        rows=[r for r in batch['surfaces'] if r['source_part'] not in selected];checks=[c for c in batch['geometry_checks'] if c['source_part'] not in selected]
        for name in sorted(selected):
            source=bpy.data.objects[name];r=registry[source.get('source_object',name)]
            print('[v6] review building',name,flush=True);added,check=author(source,r);rows.extend(added);checks.append(check)
        shutil.copy2(path,history/path.name)
        save_batch(batch['label'],rows,batch['replaced_parts'],checks)
        updated=json.loads(path.read_text());updated['reviewed_opening_connectivity']=True;path.write_text(json.dumps(updated,indent=2))


if __name__=='__main__':main()
