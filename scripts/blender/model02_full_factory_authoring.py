"""Native source extraction and identity-preserving component mesh creation."""
import bpy,hashlib,json,math,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import os
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
OUT=ROOT/'factory/model02_thermal_v6'
from scripts.blender.model02_route_refinement import authored_world,scene_parts
from scripts.blender.model02_geometry import polygon_area
OBJECT_INDEX={o.name:o for o in bpy.data.objects}
COLLECTION_INDEX={c.name:c for c in bpy.data.collections}


def refresh_index():
    """Call after the explicit library merge or bulk deletion review pass."""
    OBJECT_INDEX.clear();OBJECT_INDEX.update((o.name,o) for o in bpy.data.objects)
    COLLECTION_INDEX.clear();COLLECTION_INDEX.update((c.name,c) for c in bpy.data.collections)


def inputs(roles):
    registry=json.loads((OUT/'source_registry.json').read_text())
    by_source={r['source_object']:r for r in registry['sources']}
    selected=[]
    for o in bpy.context.scene.objects:
        if o.type!='MESH' or o.hide_render:continue
        raw=o.name in by_source and by_source[o.name]['render_visible_original']
        coarse=o.get('thermal_scope')=='background_retained' or o.get('component')=='PreservedRoadBackground'
        source=o.name if raw else o.get('source_object')
        if (raw or coarse) and source in by_source and by_source[source]['role'] in roles:
            selected.append((o,by_source[source]))
    return sorted(selected,key=lambda item:item[0].name)


def parents_for(registry):
    from mathutils import Matrix
    collection=COLLECTION_INDEX.get('Model02_FullFactoryV6')
    if collection is None:
        collection=bpy.data.collections.new('Model02_FullFactoryV6')
        if bpy.context.scene.get('v6_batch_asset_mode'):collection.hide_viewport=True
        else:bpy.context.scene.collection.children.link(collection)
        COLLECTION_INDEX[collection.name]=collection
    root=OBJECT_INDEX.get('FullFactoryV6')
    if root is None:
        root=bpy.data.objects.new('FullFactoryV6',None);collection.objects.link(root)
        OBJECT_INDEX[root.name]=root
        root.parent=bpy.data.objects['InspectionZone01'].parent;root.matrix_basis=authored_world(root.parent).inverted()
    key='V6_'+registry['zone_id'];zone=OBJECT_INDEX.get(key)
    if zone is None:
        zone=bpy.data.objects.new(key,None);collection.objects.link(zone);zone.parent=root;zone.matrix_world=Matrix.Identity(4)
        OBJECT_INDEX[zone.name]=zone
    name='V6_'+registry['source_object'];parent=OBJECT_INDEX.get(name)
    if parent is None:
        parent=bpy.data.objects.new(name,None);collection.objects.link(parent);parent.parent=zone;parent.matrix_world=Matrix.Identity(4)
        OBJECT_INDEX[parent.name]=parent
    return collection,parent


def source_records(o):
    mesh=o.data;mesh.calc_loop_triangles();matrix=np.asarray(authored_world(o));points=np.asarray([v.co[:] for v in mesh.vertices])
    world=(np.c_[points,np.ones(len(points))]@matrix.T)[:,:3]
    _,weld=np.unique(np.round(world,5),axis=0,return_inverse=True)
    roots=list(range(len(mesh.polygons)));edges={}
    def find(i):
        while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
        return i
    for face in mesh.polygons:
        ids=[int(weld[v]) for v in face.vertices]
        for a,b in zip(ids,ids[1:]+ids[:1]):
            key=(face.material_index,min(a,b),max(a,b))
            old=edges.get(key)
            if old is None:edges[key]=face.index
            else:
                x,y=find(face.index),find(old)
                if x!=y:roots[max(x,y)]=min(x,y)
    islands={i:find(i) for i in range(len(roots))};uv=mesh.uv_layers.active.data if mesh.uv_layers.active else None
    normals=np.asarray([n.vector[:] for n in mesh.corner_normals]);items=[];zero=0
    for triangle in mesh.loop_triangles:
        face=mesh.polygons[triangle.polygon_index];poly=[]
        for li in triangle.loops:
            vertex=mesh.loops[li].vertex_index
            poly.append(np.r_[world[vertex],uv[li].uv[:] if uv else (0.,0.),normals[li]])
        if polygon_area(poly)<1e-12:zero+=1;continue
        items.append({'poly':poly,'face':face.index,'material':face.material_index,'island':islands[face.index],'smooth':face.use_smooth})
    return items,world,zero


def create_surface(source,registry,key,items,component,ir_material,region,assumption='Original source surface, split continuously into independent configurable regions',materials=None):
    name='F6_'+registry['source_object'].replace('11_29_','').upper()+'_'+component+'_'+hashlib.sha256(str((source.name,key)).encode()).hexdigest()[:16]
    assert len(name)<64 and name not in OBJECT_INDEX,name
    collection,parent=parents_for(registry);matrix=np.asarray(authored_world(source));inv=np.linalg.inv(matrix)
    mesh=bpy.data.meshes.new(name+'_Mesh');vertices=[];faces=[];corners=[];lookup={};used=sorted({r['material'] for r in items})
    for r in items:
        face=[];data=[];seen=set()
        for p in r['poly']:
            vkey=tuple(np.round(p[:3],8))
            if vkey in seen:continue
            seen.add(vkey)
            if vkey not in lookup:lookup[vkey]=len(vertices);vertices.append((inv@np.r_[p[:3],1.])[:3])
            face.append(lookup[vkey]);data.append(p)
        if len(face)>=3:faces.append(face);corners.append((r,data))
    if not faces:
        (OUT/'failed_surface.json').write_text(json.dumps({'name':name,'polys':[np.asarray(r['poly']).tolist() for r in items]},indent=2))
        raise ValueError('Surface collapsed during vertex welding: '+name)
    mesh.from_pydata(vertices,[],faces);mesh.update()
    for i in used:mesh.materials.append((source.data.materials if materials is None else materials)[i])
    uv=mesh.uv_layers.new(name='UVMap');normals=[]
    for face,(r,data) in zip(mesh.polygons,corners):
        face.material_index=used.index(r['material'])
        face.use_smooth=r.get('smooth',False)
        for li,p in zip(face.loop_indices,data):uv.data[li].uv=p[3:5];normals.append(p[5:8])
    mesh.normals_split_custom_set(normals)
    obj=bpy.data.objects.new(name,mesh);collection.objects.link(obj);obj.parent=parent
    OBJECT_INDEX[name]=obj
    obj.matrix_basis=authored_world(parent).inverted()@authored_world(source)
    for k,v in {'surface_id':name,'temperature_target_id':name.lower(),'component':component,'region':str(region),
        'ir_material':ir_material,'physical_material':ir_material,'thermal_scope':'full_factory_v6',
        'source_object':registry['source_object'],'source_part':source.name,'temperature_configuration':'pending_explicit_v6_configuration',
        'modeling_approximation':assumption}.items():obj[k]=v
    obj['source_face_ids']=sorted({r['face'] for r in items})
    points=np.concatenate([np.asarray(r['poly'])[:,:3] for r in items]);center=points.mean(0)
    return {'surface_id':name,'object':name,'source_object':registry['source_object'],'source_part':source.name,
        'component':component,'region':str(region),'target':name.lower(),'material':ir_material,'center_m':center.tolist(),
        'source_face_ids':list(obj['source_face_ids']),'bounds_m':[points.min(0).tolist(),points.max(0).tolist()],
        'area_m2':sum(polygon_area(r['poly']) for r in items),'assumption':assumption}


def save_batch(label,rows,replaced,checks):
    import time
    started=time.perf_counter();print('[v6] checkpoint writing',label,len(rows),'regions',flush=True)
    if bpy.context.scene.get('v6_batch_asset_mode'):
        assets=OUT/'assets';assets.mkdir(exist_ok=True);asset=assets/(label+'.blend');temporary=assets/(label+'_saving.blend')
        bpy.data.libraries.write(str(temporary),{OBJECT_INDEX[r['object']] for r in rows},path_remap='NONE')
        if asset.exists():os.replace(asset,asset.with_suffix('.blend1'))
        os.replace(temporary,asset)
        folder=OUT/'batches';folder.mkdir(exist_ok=True)
        report={'label':label,'surfaces':rows,'replaced_parts':replaced,'geometry_checks':checks,'asset_file':asset.relative_to(OUT).as_posix(),'asset_mode':'native_object_library','reviewed_opening_connectivity':label.startswith('buildings_'),'reviewed_material_boundaries':label.startswith('ground_')}
        (folder/(label+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('[v6] saved object library',label,len(rows),'surfaces; seconds',round(time.perf_counter()-started,2),flush=True)
        return
    name=bpy.context.scene.get('v6_checkpoint_name','model02_v6_working.blend')
    assert name in ['model02_v6_working.blend','model02_v6_ground_working.blend']
    checkpoint=OUT/name
    bpy.context.scene['full_factory_v6_phase']='Geometry authoring; not configured or accepted'
    # The operator first evaluates/copies the entire dependency graph. This
    # checkpoint needs authored datablocks only; native reload was verified.
    temporary=checkpoint.with_name(checkpoint.stem+'_saving.blend')
    assert temporary.parent.resolve()==OUT.resolve()
    bpy.data.libraries.write(str(temporary),set(bpy.data.scenes)|set(bpy.data.workspaces),path_remap='NONE')
    if checkpoint.exists():os.replace(checkpoint,checkpoint.with_suffix('.blend1'))
    os.replace(temporary,checkpoint)
    folder=OUT/'batches';folder.mkdir(exist_ok=True)
    (folder/(label+'.json')).write_text(json.dumps({'label':label,'surfaces':rows,'replaced_parts':replaced,'geometry_checks':checks},indent=2),encoding='utf-8')
    print('[v6] saved',label,len(rows),'new independent regions; checkpoint seconds',round(time.perf_counter()-started,2),flush=True)
