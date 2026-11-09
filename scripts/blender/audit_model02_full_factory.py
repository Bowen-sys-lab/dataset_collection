"""Native read-only source registry and shape contact sheets; no blend saved."""
import bpy,json,sys,math
from pathlib import Path
from collections import defaultdict
from mathutils import Matrix,Vector
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.model02_inventory import snapshot_scene
from scripts.blender.model02_route_refinement import authored_world,scene_parts
from scripts.model02_full_factory_registry import validate_registry


def main():
    folder=ROOT/'factory/model02_thermal_v6';folder.mkdir(exist_ok=True)
    out=ROOT/'validation/full-factory/classification';out.mkdir(parents=True,exist_ok=True)
    inventory=json.loads((ROOT/'validation/full-factory/inventory.json').read_text())
    snapshot={r['name']:r for r in snapshot_scene()['objects']}
    parts=defaultdict(list)
    for o in scene_parts():parts[o.get('source_object')].append(o)
    rows=[];groups=defaultdict(list)
    visible_collections=set()
    def visit(collection,enabled):
        enabled=enabled and not collection.hide_render
        if enabled:visible_collections.add(collection.name)
        for child in collection.children:visit(child,enabled)
    visit(bpy.context.scene.collection,True)
    for r in inventory['objects']:
        name=r['name'];o=bpy.data.objects[name]
        effective=not o.hide_render and any(c.name in visible_collections for c in o.users_collection)
        components={p.get('component') for p in parts[name]}
        role='unclassified'
        if components:
            role='building' if 'Wall' in components else 'equipment' if 'TankShell' in components else 'pipework' if 'PipeWall' in components else 'ground'
        center=[(a+b)/2 for a,b in zip(*r['bounds_m'])]
        zone=f'Z{min(3,max(0,int((center[0]-400)/200)))}{min(3,max(0,int((center[1]+950)/200)))}'
        rows.append({'source_object':name,'source_geometry_sha256':snapshot[name]['geometry_uv_material_normal_sha256'],
            'zone_id':zone,'role':role,'classification_evidence':['v5 existing authored components: '+','.join(sorted(components))] if components else [],
            'component_rules':[],'face_accounting':[{'start':0,'stop':r['faces'],'action':'retained'}],
            'status':'classified' if components else 'unclassified','render_visible_original':effective,
            'source_faces':r['faces'],'bounds_m':r['bounds_m'],'materials':r['rgb_materials'],'image_references':r['images'],
            'repeated_geometry_key':r['repeated_geometry_key']})
        groups[r['repeated_geometry_key']].append(name)
    registry={'version':6,'source_blend':'factory/model02_thermal_v5/model02_thermal_v5_configured.blend','sources':rows}
    report=validate_registry(inventory,registry)
    (folder/'source_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    (out/'registry_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    prototypes=[names[0] for names in groups.values()]
    scene=bpy.data.scenes.new('FullFactory_ReadOnly_ContactSheets');scene.render.engine='BLENDER_WORKBENCH'
    scene.render.resolution_x=1200;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.display.shading.light='STUDIO'
    scene.display.shading.color_type='SINGLE';scene.display.shading.single_color=(.62,.67,.72)
    scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True
    camera=bpy.data.objects.new('AuditCamera',bpy.data.cameras.new('AuditCamera'));scene.collection.objects.link(camera)
    camera.location=(3.,2.5,15.);camera.data.type='ORTHO';camera.data.ortho_scale=6.1;scene.camera=camera
    rotation=Matrix.Rotation(math.radians(60),4,'X')@Matrix.Rotation(math.radians(-35),4,'Z')
    sheets=[]
    for start in range(0,len(prototypes),30):
        copies=[];cells=[]
        for index,name in enumerate(prototypes[start:start+30]):
            source=bpy.data.objects[name];world=authored_world(source)
            points=[rotation@(world@Vector(v)) for v in source.bound_box]
            lo=Vector([min(p[i] for p in points) for i in range(3)]);hi=Vector([max(p[i] for p in points) for i in range(3)])
            center=(lo+hi)/2;scale=.82/max(hi.x-lo.x,hi.y-lo.y,.001)
            location=Vector([index%6+.5,4.5-index//6,0.])
            obj=bpy.data.objects.new('Audit_'+name,source.data);scene.collection.objects.link(obj)
            obj.matrix_world=Matrix.Translation(location)@Matrix.Scale(scale,4)@Matrix.Translation(-center)@rotation@world
            copies.append(obj);cells.append({'cell':index,'source_object':name,'members':groups[next(k for k,v in groups.items() if v[0]==name)]})
        path=out/f'shapes_{start//30:02d}.png';scene.render.filepath=str(path)
        bpy.ops.render.render(write_still=True,scene=scene.name)
        sheets.append({'image':str(path.relative_to(ROOT)),'cells':cells})
        for obj in copies:bpy.data.objects.remove(obj,do_unlink=True)
        print('[full audit] sheet',len(sheets),flush=True)
    (out/'shape_sheets.json').write_text(json.dumps(sheets,indent=2),encoding='utf-8')
    print('[full audit] complete',len(rows),'sources',len(prototypes),'shape prototypes',flush=True)


if __name__=='__main__':main()
