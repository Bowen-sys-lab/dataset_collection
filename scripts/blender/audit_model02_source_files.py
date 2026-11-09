"""Compare serialized source geometry on equal Blender cache/evaluation footing."""
import bpy,sys,json,hashlib
import numpy as np
from pathlib import Path
sys.path.insert(0,'D:/IsaacSim_Demo')
from scripts.blender.model02_inventory import snapshot_scene
from scripts.blender.model02_route_refinement import authored_world
ROOT=Path('D:/IsaacSim_Demo/factory')

def capture():
    result={}
    for o in bpy.context.scene.objects:
        if o.type!='MESH' or not o.name.startswith('11_29_gc'):continue
        m=o.data;m.calc_loop_triangles()
        fields={'positions':np.array([v.co[:] for v in m.vertices],dtype='float32'),
                'indices':np.array([l.vertex_index for l in m.loops],dtype='int32'),
                'material_indices':np.array([p.material_index for p in m.polygons],dtype='int32'),
                'matrix':np.array(authored_world(o),dtype=float),
                'normals':np.array([n.vector[:] for n in m.corner_normals],dtype='float32')}
        for i,uv in enumerate(m.uv_layers):fields['uv'+str(i)]=np.array([x.uv[:] for x in uv.data],dtype='float32')
        result[o.name]=fields
    return result

before=capture()
snapshot=snapshot_scene()
for record in snapshot['objects']:record['matrix_world']=[list(row) for row in authored_world(bpy.data.objects[record['name']])]
(ROOT/'model02_thermal_v3/source_snapshot_serialized_baseline.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'model02_thermal_v3_build/model02_thermal_v3.blend'))
after=capture();differences=[]
for name,fields in before.items():
    for field,value in fields.items():
        other=after[name][field]
        if not np.array_equal(value,other):
            differences.append({'object':name,'field':field,'shape_before':value.shape,'shape_after':other.shape,'max_error':float(np.max(np.abs(value-other))) if value.shape==other.shape else None})
report={'result':'PASS' if not differences else 'DIFFERENCES','source_meshes':len(before),'differences':differences}
(ROOT/'model02_thermal_v3/source_file_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
