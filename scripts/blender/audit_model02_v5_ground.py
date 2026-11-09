"""Reload both Blender files to verify original geometry and independent thermal data."""
import bpy,hashlib,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.model02_route_refinement import authored_world,scene_parts
OUT=ROOT/'factory/model02_thermal_v5'


def geometry(o):
    h=hashlib.sha256()
    for a in [np.array([v.co[:] for v in o.data.vertices],dtype='float32'),
              np.array([l.vertex_index for l in o.data.loops],dtype='int32'),np.array(authored_world(o)),
              np.array([p.material_index for p in o.data.polygons],dtype='int32'),
              np.array([n.vector[:] for n in o.data.corner_normals],dtype='float32')]:h.update(a.tobytes())
    for uv in o.data.uv_layers:h.update(np.array([v.uv[:] for v in uv.data],dtype='float32').tobytes())
    h.update(json.dumps([m.name if m else None for m in o.data.materials]).encode())
    return h.hexdigest()


def state():return {o.name:tuple(o.get(k) for k in ['surface_id','temperature_target_id','temperature_c','ir_material']) for o in scene_parts()}


def main():
    report=json.loads((OUT/'blender_ground_report.json').read_text())
    names=set(report['modified_names']);newstate=state()
    unchanged={o.name:geometry(o) for o in bpy.context.scene.objects if o.type=='MESH' and o.name not in names}
    endpoints=defaultdict(list)
    for o in scene_parts():
        if o.get('component') not in ['PavingBrick','PavingJoint']:continue
        for v in o.data.vertices:
            p=o.matrix_world@v.co
            endpoints[(round(p.x,4),round(p.y,4))].append((o.name,p.z))
    jumps=[max(v for _,v in values)-min(v for _,v in values) for values in endpoints.values() if len({name for name,_ in values})>1]
    assert jumps and max(jumps)<.0001,('Brick joins exceed100um source/export tolerance',max(jumps))
    assert len([m for m in bpy.data.materials if m.name.startswith('V5_')])==4
    maps=[i for i in bpy.data.images if i.name.split('.')[0].endswith(('_basecolor','_normal','_roughness'))]
    assert len(maps)==12 and all(i.packed_file for i in maps)
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'factory/model02_thermal_v4/model02_thermal_v4_configured.blend'))
    assert state()==newstate
    differences=[name for name,digest in unchanged.items() if geometry(bpy.data.objects[name])!=digest]
    assert not differences,differences[:10]
    result={'result':'PASS_V5_RELOADED_BLENDER','independent_thermal_targets_unchanged':len(newstate),
        'other_meshes_unchanged':len(unchanged),'unchanged_scope':'positions, topology, authored transforms, normals, UVs, RGB material assignments','packed_pbr_maps':len(maps),
        'shared_paving_vertices_checked':len(jumps),'maximum_shared_vertex_height_difference_m':max(jumps),
        'join_tolerance_m':.0001,'note':'Join tolerance accounts for source/export float coordinates; nominal joints are continuous and filled.'}
    (OUT/'blender_reload_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
