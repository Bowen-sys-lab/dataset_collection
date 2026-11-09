"""Blender-authored shallow ground relief; preserve all independent thermal IDs."""
import bpy,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
OUT=ROOT/'factory/model02_thermal_v5';OLD=ROOT/'factory/model02_thermal_v4'
from scripts.blender.model02_route_refinement import records,scene_parts,BUILDING_BASIS
from scripts.model02_ground_relief import relief_pieces
from scripts.blender.model02_geometry import polygon_area


def signature(o):
    m=o.data;h=hashlib.sha256()
    for a in [np.array([v.co[:] for v in m.vertices],dtype='float32'),
              np.array([l.vertex_index for l in m.loops],dtype='int32'),np.array(o.matrix_world)]:h.update(a.tobytes())
    return h.hexdigest()


def thermal_state(parts):
    return {o.name:[o.get(k) for k in ('surface_id','temperature_target_id','temperature_c','ir_material')]
            for o in parts}


def material(kind):
    m=bpy.data.materials.new('V5_'+kind);m.use_nodes=True
    nodes=m.node_tree.nodes;links=m.node_tree.links
    bsdf=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    for suffix,socket in [('basecolor','Base Color'),('roughness','Roughness'),('normal','Normal')]:
        n=nodes.new('ShaderNodeTexImage');n.image=bpy.data.images.load(str(OUT/'textures'/(kind+'_'+suffix+'.png')),check_existing=True)
        n.image.colorspace_settings.name='sRGB' if suffix=='basecolor' else 'Non-Color'
        n.image.pack()
        if suffix=='normal':
            normal=nodes.new('ShaderNodeNormalMap');normal.inputs['Strength'].default_value=1.
            links.new(n.outputs['Color'],normal.inputs['Color']);links.new(normal.outputs['Normal'],bsdf.inputs[socket])
        else:links.new(n.outputs['Color'],bsdf.inputs[socket])
    return m


def write_mesh(o,polygons,material):
    inv=np.linalg.inv(np.asarray(o.matrix_world));verts=[];faces=[];lookup={};uvs=[]
    for poly in polygons:
        face=[];seen=set()
        for p in poly:
            key=tuple(np.round(p[:3],8))
            if key in seen:continue
            seen.add(key)
            if key not in lookup:
                lookup[key]=len(verts);verts.append((inv@np.r_[p[:3],1])[:3])
            face.append(lookup[key]);uvs.append(p[3:5])
        if len(face)>=3:faces.append(face)
        else:raise ValueError('Degenerate relief polygon: '+o.name)
    old=o.data;old_name=old.name
    assert old.users==1,('Ground mesh is shared',o.name)
    old.name='V5_Replaced_'+old_name
    mesh=bpy.data.meshes.new(old_name);mesh.from_pydata(verts,[],faces);mesh.update()
    mesh.materials.append(material);uv=mesh.uv_layers.new(name='UVMap')
    for index,value in enumerate(uvs):uv.data[index].uv=value
    o.data=mesh


def main():
    source=OLD/'model02_thermal_v4_configured.blend'
    assert Path(bpy.data.filepath)==source
    destination=OUT/'model02_thermal_v5_configured.blend';assert not destination.exists()
    parts=scene_parts();before_thermal=thermal_state(parts)
    before={o.name:signature(o) for o in bpy.context.scene.objects if o.type=='MESH'}
    mats={k:material(k) for k in ['brick','concrete','joint','grass']}
    parameters=json.loads((OUT/'texture_parameters.json').read_text())
    anchor=np.array([700.,-845.,0.]);basis=BUILDING_BASIS
    modified=[];counts={};area_before=area_after=0.;degenerate_source_triangles=0
    for o in parts:
        component=o.get('component');grass=component=='Landscape' and all(m and m.name.startswith('11-29-gc008_4') for m in o.data.materials)
        if component not in ['PavingBrick','PavingJoint','ConcreteRoadSlab','RoadJoint','PreservedRoadBackground','KerbAndRiser'] and not grass:continue
        kind='grass' if grass else 'brick' if component=='PavingBrick' else 'joint' if component in ['PavingJoint','RoadJoint'] else 'concrete'
        span=parameters[kind]['texture_span_m'];polygons=[]
        for r in records(o):
            p=np.asarray(r['poly']).copy();original=p.copy()
            if polygon_area(list(p))<=1e-12:
                degenerate_source_triangles+=1;continue
            # Shared factory-local coordinates keep UVs continuous across thermal quadrants.
            p[:,:3]=(p[:,:3]-anchor)@basis.T
            if component in ['PavingBrick','ConcreteRoadSlab']:
                pieces=relief_pieces(list(p),'paving' if component=='PavingBrick' else 'road')
            else:
                if component in ['PavingJoint','RoadJoint']:p[:,2]-=.002 if component=='PavingJoint' else .001
                pieces=[p]
            for piece in pieces:
                piece=np.asarray(piece);piece[:,3:5]=piece[:,:2]/span
                piece[:,:3]=piece[:,:3]@basis+anchor
                if component in ['PavingBrick','PavingJoint','ConcreteRoadSlab','RoadJoint']:
                    flat=piece.copy();flat[:,2]=0.;area_after+=polygon_area(list(flat))
                polygons.append(piece)
            if component in ['PavingBrick','PavingJoint','ConcreteRoadSlab','RoadJoint']:
                original[:,2]=0.;area_before+=polygon_area(list(original))
        write_mesh(o,polygons,mats[kind])
        o['v5_surface_detail']='2mm filled brick seam/3mm edge chamfer;1mm concrete control joint; soft canopy normal only for grass; no solved heat transfer'
        modified.append(o);counts[component]=counts.get(component,0)+1
        if len(modified)%700==0:print('[v5] ground parts '+str(len(modified)),flush=True)
    assert thermal_state(parts)==before_thermal
    names={o.name for o in modified}
    unchanged=[o.name for o in bpy.context.scene.objects if o.type=='MESH' and o.name not in names and signature(o)==before[o.name]]
    assert len(unchanged)+len(names)==len(before)
    assert abs(area_before-area_after)<1e-6*max(area_before,1.)
    bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1.
    bpy.context.scene['v5_ground_assumptions']='Dry night: shallow filled seams and small chamfers; dimension assumptions, not surveyed. Existing thermal targets retained.'
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    bpy.ops.object.select_all(action='DESELECT')
    for o in modified:
        o.select_set(True);p=o.parent
        while p:p.select_set(True);p=p.parent
    output=OUT/'model02_v5_ground.usdc';assert not output.exists()
    bpy.ops.wm.usd_export(filepath=str(output),selected_objects_only=True,root_prim_path='/World',export_animation=False,
        export_custom_properties=True,author_blender_name=True,export_cameras=False,convert_world_material=False,triangulate_meshes=True,
        export_textures_mode='NEW',overwrite_textures=False,relative_paths=True,export_uvmaps=True,export_normals=True,generate_preview_surface=True)
    report={'result':'PASS_BLENDER_GROUND_RELIEF','modified_parts':len(modified),'component_counts':counts,
        'unchanged_other_meshes':len(unchanged),'thermal_targets_unchanged':len(before_thermal),
        'projected_area_before_m2':area_before,'projected_area_after_m2':area_after,
        'zero_area_source_triangles_omitted':degenerate_source_triangles,
        'modified_names':sorted(names),'source_blend_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'configured_blend_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'export_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'note':'RGB normal maps express subgeometry grass canopy only; brick edges and joints are actual mesh relief. Thermal targets remain existing static night estimates.'}
    (OUT/'blender_ground_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('[v5] '+json.dumps({k:v for k,v in report.items() if k!='modified_names'}),flush=True)


if __name__=='__main__':main()
