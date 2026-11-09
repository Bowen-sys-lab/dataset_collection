"""Load authored scene data before enabling a large viewport dependency graph."""
import argparse,bpy,json,runpy,sys
from pathlib import Path
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--script',type=Path);parser.add_argument('--probe',type=Path)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
startup=list(bpy.data.scenes)
bpy.data.batch_remove([*bpy.data.objects,*bpy.data.meshes,*bpy.data.materials,*bpy.data.cameras,*bpy.data.lights,*bpy.data.worlds,*bpy.data.collections])
with bpy.data.libraries.load(str(args.input.resolve()),link=False) as (source,destination):
    assert 'Scene' in source.scenes;destination.scenes=['Scene']
scene=destination.scenes[0];collection=bpy.data.collections.get('Model02_FullFactoryV6')
if collection:collection.hide_viewport=True
if bpy.context.window:bpy.context.window.scene=scene
for old in startup:bpy.data.scenes.remove(old)
scene.name='Scene';scene['v6_loaded_source']=str(args.input.resolve())
with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
    if args.probe:
        report={'result':'PASS_SCENE_DATA_LOAD','objects':len(scene.objects),'meshes':sum(o.type=='MESH' for o in scene.objects),
            'new_surfaces':sum(bool(o.get('thermal_scope')=='full_factory_v6') for o in scene.objects),'units':scene.unit_settings.system,
            'viewport_full_factory_deferred':bool(collection and collection.hide_viewport),'source':str(args.input.resolve())}
        args.probe.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if args.script:
        sys.argv=[str(args.script)];runpy.run_path(str(args.script.resolve()),run_name='__main__')
