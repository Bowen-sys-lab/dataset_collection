"""Apply explicit v6 parameters and enumerate native exported surface paths."""
import bpy,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT));OUT=ROOT/'factory/model02_thermal_v6'
from scripts.blender.model02_route_refinement import authored_world,scene_parts
from scripts.blender.model02_inventory import snapshot_scene


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    from pxr import Usd,UsdGeom,UsdShade,Sdf,Ar
    assert Path(bpy.data.filepath)==OUT/'model02_v6_working.blend'
    from scripts.model02_configuration_freshness import verify_inputs
    config=json.loads((OUT/'configuration/blender_parameters.json').read_text());verify_inputs(ROOT,config)
    allparts=scene_parts();parts=[o for o in allparts if o.get('thermal_scope')=='full_factory_v6']
    assert {o.name for o in allparts}==set(config['objects'])
    registry=json.loads((OUT/'source_registry.json').read_text());before={r['source_object']:r['source_geometry_sha256'] for r in registry['sources']}
    actual={r['name']:r['geometry_uv_material_normal_sha256'] for r in snapshot_scene()['objects']};assert before==actual,'Original source geometry changed'
    old=json.loads((ROOT/'factory/model02_thermal_v4/configuration/blender_parameters.json').read_text())['objects']
    replaced={n for p in (OUT/'batches').glob('*.json') for n in json.loads(p.read_text())['replaced_parts']}
    inherited=[o for o in scene_parts() if o.name in old]
    for o in inherited:
        r=old[o.name];assert [o['surface_id'],o['temperature_target_id'],o['temperature_c'],o['ir_material']]==[r['surface_id'],r['target'],r['temperature_c'],r['material']]
    assert len(inherited)==len(old)-len(replaced&set(old))
    expected={}
    for o in allparts:
        r=config['objects'][o.name];o['temperature_c']=r['temperature_c'];o['ir_material']=r['material'];o['physical_material']=r['material'];o['temperature_configuration']='explicit_v6_static_snapshot'
    for o in parts:
        m=np.asarray(authored_world(o));p=np.asarray([v.co[:] for v in o.data.vertices]);p=(np.c_[p,np.ones(len(p))]@m.T)[:,:3]
        expected[o.name]={'bounds_m':[p.min(0).tolist(),p.max(0).tolist()],'blender_faces':len(o.data.polygons)}
    bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1.
    bpy.context.scene['full_factory_v6_phase']='Explicit thermal configuration; export pending native acceptance'
    bpy.context.scene['thermal_configuration_file']='//configuration/thermal_authoring.yaml'
    destination=OUT/'model02_thermal_v6_configured.blend';assert not destination.exists()
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    bpy.ops.object.select_all(action='DESELECT')
    for o in parts:
        o.select_set(True);p=o.parent
        while p:p.select_set(True);p=p.parent
    output=OUT/'model02_v6_surfaces.usdc';assert not output.exists()
    bpy.ops.wm.usd_export(filepath=str(output),selected_objects_only=True,root_prim_path='/World',export_animation=False,
        export_custom_properties=True,author_blender_name=True,export_cameras=False,convert_world_material=False,triangulate_meshes=True,
        export_textures_mode='NEW',overwrite_textures=False,relative_paths=True,export_uvmaps=True,export_normals=True,generate_preview_surface=True)
    stage=Usd.Stage.Open(str(output));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');rows=[]
    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Mesh):continue
        parent=p
        while parent and not parent.GetAttribute('userProperties:surface_id').HasAuthoredValueOpinion():parent=parent.GetParent()
        assert parent;sid=parent.GetAttribute('userProperties:surface_id').Get();mesh=UsdGeom.Mesh(p)
        q=np.asarray(mesh.GetPointsAttr().Get());matrix=np.asarray(mesh.ComputeLocalToWorldTransform(0));q=(np.c_[q,np.ones(len(q))]@matrix)[:,:3]
        bounds=[q.min(0).tolist(),q.max(0).tolist()];assert np.allclose(bounds,expected[sid]['bounds_m'],atol=.001,rtol=0),(sid,bounds)
        assert mesh.GetNormalsAttr().HasAuthoredValueOpinion() and UsdGeom.PrimvarsAPI(p).HasPrimvar('st') and UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
        rows.append({'surface_id':sid,'object':sid,'usd_prim_path':str(p.GetPath()),'bounds_m':bounds,'triangles':len(mesh.GetFaceVertexCountsAttr().Get())})
    assert {r['surface_id'] for r in rows}==set(expected);stage.GetRootLayer().Save()
    textures={}
    for p in stage.Traverse():
        if not p.IsA(UsdShade.Shader):continue
        for attr in p.GetAttributes():
            value=attr.Get()
            if not isinstance(value,Sdf.AssetPath) or not value.path:continue
            resolved=value.resolvedPath or str(Ar.GetResolver().Resolve(Sdf.ComputeAssetPathRelativeToLayer(stage.GetRootLayer(),value.path)))
            asset=Path(resolved);assert resolved and asset.is_file(),('Missing exported shader asset',str(attr.GetPath()),value.path)
            textures[value.path]={'resolved_path':str(asset),'sha256':sha(asset)}
    assert textures,'No exported RGB texture dependencies'
    report={'result':'PASS_NATIVE_V6_EXPORT','surfaces':rows,'surface_count':len(rows),'source_objects_preserved':len(before),
        'inherited_targets_preserved':len(inherited),'replaced_parts':sorted(replaced),'configured_blend_sha256':sha(destination),
        'usd_sha256':sha(output),'texture_dependencies':textures}
    (OUT/'export_manifest.json').write_text(json.dumps(report,indent=2));print('[v6] exported',len(rows),'actual surface paths',flush=True)


if __name__=='__main__':main()
