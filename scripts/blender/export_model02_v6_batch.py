"""Configure and export one verified object library, preserving measured world bounds."""
import argparse,bpy,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    from pxr import Usd,UsdGeom,UsdShade,Sdf,Ar
    from scripts.blender.model02_route_refinement import authored_world
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    params=json.loads((OUT/'configuration/batches'/(args.label+'.json')).read_text());assert args.label==params['label']
    asset=OUT/params['asset_file'];assert sha(asset)==params['asset_sha256']
    assert sha(OUT/'configuration/thermal_authoring.yaml')==params['authoring_sha256']
    for relative,digest in params.get('authoring_input_sha256',{}).items():assert sha(ROOT/relative)==digest,('Changed source configuration; recompile',relative)
    rows=params['surfaces'];names={r['object'] for r in rows};expected={r['object']:r for r in rows}
    bpy.data.batch_remove([*bpy.data.objects,*bpy.data.meshes,*bpy.data.materials,*bpy.data.collections])
    with bpy.data.libraries.load(str(asset),link=False) as (source,destination):
        assert names<=set(source.objects);destination.objects=sorted(names)
    parts=destination.objects
    from scripts.blender.model02_full_factory_authoring import source_records
    from scripts.blender.model02_geometry import polygon_area
    from scripts.blender.model02_full_factory_common import native_area_rounding_bound
    source_names={r['source_part'] for r in rows}
    with bpy.data.libraries.load(str(ROOT/'factory/model02_thermal_v5/model02_thermal_v5_configured.blend'),link=True) as (source,destination):destination.objects=sorted(source_names)
    originals={o.name:o for o in destination.objects};face_audit=[];checks={r['source_part']:r for r in params['geometry_checks']}
    for name,original in sorted(originals.items()):
        records,_,_=source_records(original);nonzero={r['face'] for r in records};selected=[r for r in rows if r['source_part']==name]
        used=set(params.get('source_face_coverage',{}).get(name,[i for r in selected for i in r['source_face_ids']]));carrier=set(checks[name].get('openings',{}).get('opaque_carrier_source_face_ids',[]))
        assert used|carrier<=set(range(len(original.data.polygons))),(name,'Invalid source face index')
        missing=nonzero-used-carrier;missing_area=sum(polygon_area(r['poly']) for r in records if r['face'] in missing);area=sum(polygon_area(r['poly']) for r in records)
        assert missing_area<=max(1e-8,area*1e-7),(name,'Unaccounted source geometry',len(missing),missing_area)
        face_audit.append({'source_part':name,'index_space':'raw_source_polygon' if name.startswith('11_29_gc') else 'preserved_part_polygon',
            'source_faces':len(original.data.polygons),'nonzero_faces':len(nonzero),'represented_face_ids':len(used),'carrier_cut_face_ids':sorted(carrier),'chunk_surfaces':len(selected),
            'subresolution_face_ids':sorted(missing),'subresolution_area_m2':missing_area})
    collection=bpy.data.collections.new('V6Asset_'+args.label);bpy.context.scene.collection.children.link(collection)
    linked=set();max_area_error=0.;max_rounding_bound=0.
    for o in parts:
        r=params['objects'][o.name];assert o['surface_id']==r['surface_id'] and o['temperature_target_id']==r['target']
        o['temperature_c']=r['temperature_c'];o['ir_material']=r['material'];o['physical_material']=r['material'];o['temperature_configuration']='explicit_v6_static_snapshot'
        o['source_face_index_space']='raw_source_polygon' if expected[o.name]['source_part']==expected[o.name]['source_object'] else 'preserved_part_polygon'
        local=np.array([v.co[:] for v in o.data.vertices]);m=np.array(authored_world(o));p=(np.c_[local,np.ones(len(local))]@m.T)[:,:3]
        assert np.allclose([p.min(0),p.max(0)],expected[o.name]['bounds_m'],atol=.001,rtol=0.),o.name
        o.data.calc_loop_triangles();indices=np.array([t.vertices[:] for t in o.data.loop_triangles]);triangles=p[indices]
        areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2.;area=float(areas.sum())
        area_error=abs(area-expected[o.name]['area_m2']);rounding_bound=native_area_rounding_bound(local,m,indices)
        assert area>0. and area_error<max(1e-6,expected[o.name]['area_m2']*1e-5)+rounding_bound,(o.name,'Native mesh area changed beyond float32 precision',area,expected[o.name]['area_m2'],rounding_bound)
        max_area_error=max(max_area_error,area_error);max_rounding_bound=max(max_rounding_bound,rounding_bound)
        uv=np.array([d.uv[:] for d in o.data.uv_layers.active.data]);normals=np.array([n.vector[:] for n in o.data.corner_normals]);assert np.isfinite(p).all() and np.isfinite(uv).all() and np.isfinite(normals).all(),o.name
        parent=o
        while parent:
            if parent.name not in linked:collection.objects.link(parent);linked.add(parent.name)
            if parent is not o:assert parent.type=='EMPTY',('Unexpected geometry ancestor',parent.name)
            parent=parent.parent
    bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1.
    configured=OUT/'assets/configured';configured.mkdir(parents=True,exist_ok=True);blend=configured/(args.label+'.blend');assert not blend.exists()
    bpy.data.libraries.write(str(blend),{collection},path_remap='NONE')
    # Blender's IDPropertyArray is not assigned a USD array type by this exporter.
    # Keep the native array in the saved Blender asset; encode the USD value explicitly.
    for o in parts:
        o['source_face_ids']=json.dumps(list(o['source_face_ids']),separators=(',',':'));o['source_face_ids_encoding']='json'
    bpy.context.view_layer.update()
    for o in bpy.context.scene.objects:o.select_set(True)
    folder=OUT/'usd'/args.label;folder.mkdir(parents=True,exist_ok=False);output=folder/'geometry.usdc'
    bpy.ops.wm.usd_export(filepath=str(output),selected_objects_only=True,root_prim_path='/World',export_animation=False,
        export_custom_properties=True,author_blender_name=True,export_cameras=False,convert_world_material=False,triangulate_meshes=True,
        export_textures_mode='NEW',overwrite_textures=False,relative_paths=True,export_uvmaps=True,export_normals=True,generate_preview_surface=True)
    stage=Usd.Stage.Open(str(output));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');exported=[]
    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Mesh):continue
        parent=p
        while parent and not parent.GetAttribute('userProperties:surface_id').HasAuthoredValueOpinion():parent=parent.GetParent()
        assert parent;sid=parent.GetAttribute('userProperties:surface_id').Get();mesh=UsdGeom.Mesh(p);row=expected[sid]
        assert json.loads(parent.GetAttribute('userProperties:source_face_ids').Get())==row['source_face_ids']
        points=np.asarray(mesh.GetPointsAttr().Get());m=np.asarray(mesh.ComputeLocalToWorldTransform(0));q=(np.c_[points,np.ones(len(points))]@m)[:,:3]
        bounds=[q.min(0).tolist(),q.max(0).tolist()];assert np.allclose(bounds,row['bounds_m'],atol=.001,rtol=0),(sid,bounds,row['bounds_m'])
        assert mesh.GetNormalsAttr().HasAuthoredValueOpinion() and UsdGeom.PrimvarsAPI(p).HasPrimvar('st') and UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
        exported.append({'surface_id':sid,'object':sid,'usd_prim_path':str(p.GetPath()),'bounds_m':bounds,'triangles':len(mesh.GetFaceVertexCountsAttr().Get())})
    assert {r['surface_id'] for r in exported}==names
    textures={}
    for p in stage.Traverse():
        if not p.IsA(UsdShade.Shader):continue
        for attr in p.GetAttributes():
            value=attr.Get()
            if not isinstance(value,Sdf.AssetPath) or not value.path:continue
            resolved=value.resolvedPath or str(Ar.GetResolver().Resolve(Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(),value.path)));path=Path(resolved)
            assert resolved and path.is_file(),('Missing shader texture',str(attr.GetPath()),value.path)
            textures[value.path]={'resolved_path':str(path),'sha256':sha(path)}
    stage.GetRootLayer().Save()
    report={'result':'PASS_NATIVE_V6_CHUNK_EXPORT','label':args.label,'batch_label':params.get('batch_label',args.label),'part_index':params.get('part_index',0),'parts_in_batch':params.get('parts_in_batch',1),'surfaces':exported,'replaced_parts':params['replaced_parts'],
        'configured_asset':blend.relative_to(OUT).as_posix(),'configured_asset_sha256':sha(blend),'usd_file':output.relative_to(OUT).as_posix(),'usd_sha256':sha(output),'texture_dependencies':textures,'source_face_audit':face_audit,
        'native_mesh_precision':{'maximum_area_error_m2':max_area_error,'maximum_float32_area_bound_m2':max_rounding_bound,'world_bounds_tolerance_m':.001}}
    (folder/'export_manifest.json').write_text(json.dumps(report,indent=2));print('[v6] batch exported',args.label,len(exported),'surfaces',flush=True)


if __name__=='__main__':main()
