"""Reload both releases on equal footing and audit source/retained geometry."""
import bpy,hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path('D:/IsaacSim_Demo');sys.path.insert(0,str(ROOT))
from scripts.blender.model02_route_refinement import scene_parts,authored_world


def fingerprint(o):
    h=hashlib.sha256();m=o.data
    for values in [np.array([v.co[:] for v in m.vertices],dtype='float32'),np.array([l.vertex_index for l in m.loops],dtype='int32'),
                   np.array([p.material_index for p in m.polygons],dtype='int32'),np.array(authored_world(o),dtype=float),
                   np.array([v.vector[:] for v in m.corner_normals],dtype='float32')]:h.update(values.tobytes())
    for uv in m.uv_layers:h.update(np.array([v.uv[:] for v in uv.data],dtype='float32').tobytes())
    h.update(json.dumps([m.name if m else None for m in o.data.materials]).encode())
    return h.hexdigest()


def main():
    out=ROOT/'factory/model02_thermal_v4'
    assert Path(bpy.data.filepath)==out/'model02_thermal_v4_configured.blend'
    config=json.loads((out/'configuration/blender_parameters.json').read_text(encoding='utf-8'))
    parts=scene_parts();assert {o.name for o in parts}==set(config['objects'])
    for o in parts:
        p=config['objects'][o.name]
        assert (o['temperature_c'],o['ir_material'],o['temperature_target_id'])==(p['temperature_c'],p['material'],p['target'])
    fingerprints={o.name:fingerprint(o) for o in bpy.context.scene.objects if o.type=='MESH' and (o.name.startswith('11_29_gc') or (o.get('surface_id') and not o.name.startswith('PAV4_')))}
    independence=[]
    for component in ['PavingBrick','PavingJoint','Wall','TankShell']:
        chosen=next(o for o in parts if o.get('component')==component)
        old={o['temperature_target_id']:o['temperature_c'] for o in parts};chosen['temperature_c']+=1.
        changed=[o['temperature_target_id'] for o in parts if o['temperature_c']!=old[o['temperature_target_id']]]
        assert changed==[chosen['temperature_target_id']]
        chosen['temperature_c']-=1.;independence.append({'component':component,'only_changed_target':changed[0],'restored':True})
    counts=dict(Counter(o['component'] for o in parts))
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'factory/model02_thermal_v3/model02_thermal_v3_configured.blend'))
    differences=[]
    for name,digest in fingerprints.items():
        if fingerprint(bpy.data.objects[name])!=digest:differences.append(name)
    assert not differences,differences
    report={'result':'PASS_RELOADED_V4_BLENDER','visible_surfaces':len(parts),'original_and_retained_meshes_identical':len(fingerprints),
        'changed_original_or_retained_meshes':differences,'component_counts':counts,'temperature_independence':independence,
        'note':'Original meshes and unchanged refined parts compared across reloaded v3/v4:positions,indices,RGB assignments,UV,normals,authored transforms. v4 ground intentionally replaces old regions.'}
    (out/'blender_reload_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
