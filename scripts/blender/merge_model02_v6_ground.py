"""Merge independently authored ground objects by explicit batch identity only."""
import bpy,json,sys
from pathlib import Path
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.model02_full_factory_authoring import OUT,parents_for,refresh_index


def main():
    assert Path(bpy.data.filepath)==OUT/'model02_v6_working.blend'
    registry={r['source_object']:r for r in json.loads((OUT/'source_registry.json').read_text())['sources']}
    batches=[json.loads(p.read_text()) for p in (OUT/'batches').glob('ground_*.json')];assert batches
    names={r['object'] for b in batches for r in b['surfaces']};assert not names&set(bpy.data.objects.keys())
    before=set(bpy.data.objects.keys())
    with bpy.data.libraries.load(str(OUT/'model02_v6_ground_working.blend'),link=False) as (source,destination):
        assert names<=set(source.objects);destination.objects=sorted(names)
    refresh_index()
    for o in destination.objects:
        collection,parent=parents_for(registry[o['source_object']]);o.parent=parent;collection.objects.link(o)
    extras=[o for o in bpy.data.objects if o.name not in before and o.name not in names and not o.users_collection]
    bpy.data.batch_remove(extras)
    for b in batches:
        for name in b['replaced_parts']:bpy.data.objects[name].hide_render=True;bpy.data.objects[name].hide_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'model02_v6_working.blend'))
    print('[v6] merged independent ground branch',len(names),'surfaces',flush=True)


if __name__=='__main__':main()
