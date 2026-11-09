"""Apply formal parameters, save configured Blender and export only v4 replacement surfaces."""
import bpy,hashlib,json,sys
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdShade
ROOT=Path('D:/IsaacSim_Demo');OUT=ROOT/'factory/model02_thermal_v4';sys.path.insert(0,str(ROOT))
from scripts.blender.model02_route_refinement import scene_parts
from scripts.model02_configuration_freshness import verify_inputs
from scripts.blender.export_model02_v3 import area


def main():
    assert Path(bpy.data.filepath)==OUT/'model02_thermal_v4.blend'
    config=json.loads((OUT/'configuration/blender_parameters.json').read_text(encoding='utf-8'));verify_inputs(ROOT,config)
    parts=scene_parts();assert {o.name for o in parts}==set(config['objects'])
    for o in parts:
        v=config['objects'][o.name];assert o['surface_id']==v['surface_id']
        o['temperature_c']=v['temperature_c'];o['ir_material']=v['material'];o['physical_material']=v['material']
    scene=bpy.context.scene
    scene['thermal_configuration_file']='//configuration/thermal_authoring.yaml';scene['thermal_configuration_sha256']=config['authoring_sha256']
    scene['thermal_material_library_sha256']=config['material_library_sha256'];scene['thermal_condition_json']=json.dumps(config['condition'])
    destination=OUT/'model02_thermal_v4_configured.blend';assert not destination.exists()
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    bpy.ops.object.select_all(action='DESELECT')
    expected={}
    for o in parts:
        if not o.name.startswith('PAV4_'):continue
        o.select_set(True);parent=o.parent
        while parent:parent.select_set(True);parent=parent.parent
        o.data.calc_loop_triangles();p=np.array([v.co[:] for v in o.data.vertices]);m=np.array(o.matrix_world)
        p=(np.c_[p,np.ones(len(p))]@m.T)[:,:3];faces=np.array([t.vertices[:] for t in o.data.loop_triangles])
        expected[o['surface_id']]={'bounds':[p.min(0).tolist(),p.max(0).tolist()],'area':area(p,faces),'triangles':len(faces),'object':o.name}
    output=OUT/'model02_v4_paving.usdc';assert not output.exists()
    bpy.ops.wm.usd_export(filepath=str(output),selected_objects_only=True,root_prim_path='/World',export_animation=False,
        export_custom_properties=True,author_blender_name=True,export_cameras=False,convert_world_material=False,triangulate_meshes=True,
        export_textures_mode='NEW',overwrite_textures=False,relative_paths=True,export_uvmaps=True,export_normals=True,generate_preview_surface=True)
    stage=Usd.Stage.Open(str(output));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    rows=[]
    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Mesh):continue
        parent=p
        while parent and not parent.GetAttribute('userProperties:surface_id').HasAuthoredValueOpinion():parent=parent.GetParent()
        assert parent
        sid=parent.GetAttribute('userProperties:surface_id').Get();e=expected[sid];mesh=UsdGeom.Mesh(p)
        local=np.asarray(mesh.GetPointsAttr().Get(),dtype=float);m=np.asarray(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0))
        points=(np.c_[local,np.ones(len(local))]@m)[:,:3];faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        assert len(faces)==e['triangles'] and np.allclose([points.min(0),points.max(0)],e['bounds'],atol=.001,rtol=0)
        assert abs(area(points,faces)-e['area'])<max(1e-5,e['area']*.001)
        assert UsdGeom.PrimvarsAPI(p).HasPrimvar('st') and mesh.GetNormalsAttr().HasAuthoredValueOpinion()
        assert UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
        rows.append({'surface_id':sid,'object':e['object'],'usd_prim_path':str(p.GetPath()),'bounds':e['bounds'],'area_m2':area(points,faces),'triangles':len(faces)})
    assert len(rows)==len(expected)
    stage.GetRootLayer().Save()
    report={'result':'PASS_V4_ACTUAL_USD_EXPORT','surfaces':rows,'surface_count':len(rows),'asset_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'configured_blend_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'authoring_sha256':config['authoring_sha256']}
    (OUT/'export_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[v4] actual exported surfaces '+str(len(rows)),flush=True)


if __name__=='__main__':main()
