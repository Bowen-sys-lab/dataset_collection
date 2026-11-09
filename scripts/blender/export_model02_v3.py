"""Export configured visible thermal parts, then audit actual USD identities/geometry."""
import bpy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from pxr import Usd,UsdGeom,UsdShade

ROOT=Path('D:/IsaacSim_Demo');ASSET=ROOT/'factory/model02_thermal_v3'
sys.path.insert(0,str(ROOT))
from scripts.blender.model02_route_refinement import scene_parts
from scripts.model02_configuration_freshness import verify_inputs


def area(points,faces):
    t=points[faces]
    return float(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum()/2)


def main():
    source=ASSET/'model02_thermal_v3_configured.blend'
    assert Path(bpy.data.filepath)==source
    compiled=json.loads((ASSET/'configuration/blender_parameters.json').read_text(encoding='utf-8'))
    verify_inputs(ROOT,compiled)
    assert bpy.context.scene['thermal_configuration_sha256']==compiled['authoring_sha256']
    parts=scene_parts();assert len(parts)==len(compiled['objects'])==8327
    assert {o.name for o in parts}==set(compiled['objects'])
    lights=[o for o in bpy.context.scene.objects if o.type=='LIGHT' and o.name!='MoonLight']
    assert len(lights)==13
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    output=ASSET/'model02_v3_asset.usdc';assert not output.exists()
    bpy.ops.object.select_all(action='DESELECT')
    expected={};depsgraph=bpy.context.evaluated_depsgraph_get()
    for obj in parts:
        values=compiled['objects'][obj.name]
        assert obj['temperature_c']==values['temperature_c'] and obj['ir_material']==values['material']
        # Compact only the in-memory export copy. Unused source slots otherwise
        # receive incorrect fallback RGB bindings from the USD exporter.
        used=sorted({p.material_index for p in obj.data.polygons})
        mats=[obj.data.materials[i] for i in used];indices={old:new for new,old in enumerate(used)}
        assignments=[indices[p.material_index] for p in obj.data.polygons]
        obj.data.materials.clear()
        for material in mats:obj.data.materials.append(material)
        for polygon,index in zip(obj.data.polygons,assignments):polygon.material_index=index
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
        local=np.array([v.co[:] for v in mesh.vertices],dtype=float)
        matrix=np.asarray(evaluated.matrix_world,dtype=float)
        points=(np.c_[local,np.ones(len(local))]@matrix.T)[:,:3]
        faces=np.array([tuple(t.vertices) for t in mesh.loop_triangles])
        expected[obj['surface_id']]={'object':obj.name,'bounds':[points.min(0).tolist(),points.max(0).tolist()],
            'area_m2':area(points,faces),'triangles':len(faces),'has_uv':bool(mesh.uv_layers),
            'rgb_materials':[m.name for m in mats],
            'source_object':obj.get('source_object'),'component':obj.get('component'),'region':obj.get('region')}
        evaluated.to_mesh_clear()
    for obj in parts+lights:
        obj.select_set(True)
        parent=obj.parent
        while parent is not None:
            parent.select_set(True);parent=parent.parent
    bpy.context.view_layer.update()
    bpy.ops.wm.usd_export(filepath=str(output),selected_objects_only=True,root_prim_path='/World',
        export_animation=False,export_custom_properties=True,author_blender_name=True,
        export_cameras=False,convert_world_material=False,triangulate_meshes=True,
        export_textures_mode='NEW',overwrite_textures=False,relative_paths=True,
        export_uvmaps=True,export_normals=True,generate_preview_surface=True)
    stage=Usd.Stage.Open(str(output));rows=[];seen=set();light_paths=[]
    for prim in stage.Traverse():
        if prim.GetTypeName().endswith('Light'):light_paths.append(str(prim.GetPath()))
        if not prim.IsA(UsdGeom.Mesh):continue
        parent=prim
        while parent and not parent.GetAttribute('userProperties:surface_id').HasAuthoredValueOpinion():parent=parent.GetParent()
        assert parent,'Mesh without stable surface ID: '+str(prim.GetPath())
        sid=parent.GetAttribute('userProperties:surface_id').Get()
        assert sid in expected and sid not in seen,sid;seen.add(sid)
        obj=expected[sid];mesh=UsdGeom.Mesh(prim)
        assert np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get())==3)
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0),dtype=float)
        world=(np.c_[points,np.ones(len(points))]@matrix)[:,:3]
        faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        assert len(faces)==obj['triangles'],sid
        assert np.allclose([world.min(0),world.max(0)],obj['bounds'],atol=.001,rtol=0),sid
        actual_area=area(world,faces)
        assert abs(actual_area-obj['area_m2'])<max(.00001,obj['area_m2']*.0001),sid
        assert mesh.GetNormalsAttr().HasAuthoredValueOpinion(),sid
        if obj['has_uv']:assert UsdGeom.PrimvarsAPI(prim).HasPrimvar('st'),sid
        if len(obj['rgb_materials'])==1:
            material=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            assert material and material.GetPrim().GetAttribute('userProperties:blender:data_name').Get()==obj['rgb_materials'][0],sid
        rows.append(dict(obj,surface_id=sid,usd_prim_path=str(prim.GetPath()),area_m2=actual_area))
    assert seen==set(expected) and len(light_paths)==13
    assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
    report={'result':'PASS_ACTUAL_USD_EXPORT','configured_blend_sha256':original_hash,
        'authoring_sha256':compiled['authoring_sha256'],'material_library_sha256':compiled['material_library_sha256'],
        'asset_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'surfaces':rows,'lights':light_paths,
        'replaced_blender_sources':sorted({r['source_object'] for r in rows if r['source_object'] and r['source_object'].startswith('11_29_gc')}),
        'geometry_audit':'Every exported mesh triangle count, world bounds/area, normals, UV and single-material RGB binding checked against evaluated Blender source'}
    (ASSET/'export_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['surfaces','lights']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
