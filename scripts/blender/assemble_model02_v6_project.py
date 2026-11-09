"""Preserve original source Scene and link configured zone libraries into a full project."""
import bpy,hashlib,json,sys
from pathlib import Path
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'
from scripts.blender.model02_inventory import snapshot_scene
from scripts.blender.model02_route_refinement import scene_parts


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    assert Path(bpy.data.filepath)==ROOT/'factory/model02_thermal_v5/model02_thermal_v5_configured.blend'
    from scripts.model02_configuration_freshness import verify_inputs
    config=json.loads((OUT/'configuration/blender_parameters.json').read_text());verify_inputs(ROOT,config)
    registry=json.loads((OUT/'source_registry.json').read_text());expected={r['source_object']:r['source_geometry_sha256'] for r in registry['sources']}
    actual={r['name']:r['geometry_uv_material_normal_sha256'] for r in snapshot_scene()['objects']};assert expected==actual
    batches=[]
    for path in sorted((OUT/'batches').glob('*.json')):
        batch=json.loads(path.read_text());pairs={(r['source_part'],r['source_object']) for r in batch['surfaces']}
        batches.append({'label':batch['label'],'replaced_parts':batch['replaced_parts'],'surface_count':len(batch['surfaces']),
            'surfaces':[{'source_part':a,'source_object':b} for a,b in sorted(pairs)]})
    replaced={n for b in batches for n in b['replaced_parts']}
    from scripts.model02_full_factory_registry import validate_authoring_coverage
    inherited=json.loads((ROOT/'factory/model02_thermal_v4/surfaces.json').read_text(encoding='utf-8'))['surfaces']
    validate_authoring_coverage(registry,batches,{r['source_object'] for r in inherited if r['blender_object'] not in replaced})
    overview=bpy.data.collections.new('V6_Overview_ViewportOnly');overview.hide_render=True;bpy.context.scene.collection.children.link(overview)
    for name in replaced:
        o=bpy.data.objects[name]
        proxy=o.copy();proxy.name='V6Overview_'+name;proxy.hide_render=True;proxy.hide_viewport=False
        proxy['viewport_proxy_only']=True;proxy['fine_geometry_location']='V6Asset_* linked collections; full detail in Isaac USD'
        overview.objects.link(proxy);o.hide_render=True;o.hide_viewport=True
    added=0;dependencies={}
    for batch in batches:
        path=OUT/'usd'/batch['label']/'export_manifest.json';report=json.loads(path.read_text());assert report['result']=='PASS_NATIVE_V6_BATCH_EXPORT'
        for item in report['configured_assets']:
            asset=OUT/item['path'];assert sha(asset)==item['sha256'];name=item['collection']
            with bpy.data.libraries.load(str(asset),link=True,relative=True) as (source,destination):
                assert name in source.collections;destination.collections=[name]
            collection=destination.collections[0];bpy.context.scene.collection.children.link(collection)
            bpy.context.view_layer.layer_collection.children[collection.name].exclude=True;dependencies[item['path']]=item['sha256']
        added+=batch['surface_count'];print('[v6] linked',batch['label'],batch['surface_count'],flush=True)
    parts=scene_parts();assert {o.name for o in parts}==set(config['objects'])
    assert all(o['temperature_target_id']==config['objects'][o.name]['target'] and abs(o['temperature_c']-config['objects'][o.name]['temperature_c'])<1e-6 and o['ir_material']==config['objects'][o.name]['material'] for o in parts)
    viewport_layer=bpy.context.view_layer
    for layer in bpy.context.scene.view_layers:layer.use=False
    render_layer=bpy.context.scene.view_layers.new('V6_AllFine_Render');render_layer.use=True
    if bpy.context.window:bpy.context.window.view_layer=viewport_layer
    bpy.context.scene['full_factory_v6_phase']='Full factory configured; native acceptance status: validation/full-factory/final/summary.json'
    bpy.context.scene['thermal_configuration_file']='//configuration/thermal_authoring.yaml'
    bpy.context.scene['v6_viewport_policy']='Overview copies are viewport-only and excluded from render/export; enable V6Asset_* collection and hide corresponding overview objects for detailed Blender inspection. Isaac USD includes all fine surfaces.'
    bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1.
    output=OUT/'model02_thermal_v6_configured.blend';assert not output.exists()
    bpy.data.libraries.write(str(output),set(bpy.data.scenes)|set(bpy.data.workspaces),path_remap='RELATIVE_ALL')
    report={'result':'PASS_FULL_V6_BLENDER_ASSEMBLY','source_objects_preserved':len(actual),'new_surfaces':added,'active_thermal_parts':len(parts),'dependencies':dependencies,'configured_blend_sha256':sha(output),'source_v5_sha256':sha(Path(bpy.data.filepath)),
        'viewport_policy':'Viewport-only source overview, fine collection evaluation deferred; full fine geometry remains in linked libraries and all-fine Isaac USD','render_view_layer':render_layer.name,'viewport_proxies':len(replaced)}
    (OUT/'blender_assembly_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
