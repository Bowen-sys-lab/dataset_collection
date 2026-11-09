"""Apply validated parameter values only, save a separate configured .blend."""
import bpy
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path('D:/IsaacSim_Demo')
ASSET=ROOT/'factory/model02_thermal_v3'
PACKAGE=ASSET/'configuration'
sys.path.insert(0,str(ROOT))
from scripts.blender.model02_route_refinement import scene_parts,authored_world
from scripts.model02_configuration_freshness import verify_inputs


def geometry_hash():
    digest=hashlib.sha256()
    for o in sorted(scene_parts(),key=lambda x:x.name):
        digest.update(o.name.encode());digest.update(np.asarray(authored_world(o),dtype='float64').tobytes())
        mesh=o.data
        for items,field,width,dtype in [(mesh.vertices,'co',3,'float32'),(mesh.loops,'vertex_index',1,'int32'),(mesh.polygons,'material_index',1,'int32')]:
            values=np.empty(len(items)*width,dtype=dtype);items.foreach_get(field,values);digest.update(values.tobytes())
        for uv in mesh.uv_layers:
            values=np.empty(len(uv.data)*2,dtype='float32');uv.data.foreach_get('uv',values);digest.update(values.tobytes())
        digest.update(np.array([n.vector[:] for n in mesh.corner_normals],dtype='float32').tobytes())
        digest.update(json.dumps([m.name if m else None for m in mesh.materials]).encode())
    return digest.hexdigest()


data=json.loads((PACKAGE/'blender_parameters.json').read_text(encoding='utf-8'))
assert data['phase']=='prebinding'
author_hash=hashlib.sha256((PACKAGE/'thermal_authoring.yaml').read_bytes()).hexdigest()
assert data['authoring_sha256']==author_hash,'Recompile edited YAML before applying'
verify_inputs(ROOT,data)
source=ASSET/'model02_thermal_v3.blend'
assert Path(bpy.data.filepath)==source
destination=ASSET/'model02_thermal_v3_configured.blend'
assert not destination.exists(),'Refusing to overwrite a previously configured version'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
parts={o.name:o for o in scene_parts()}
assert set(parts)==set(data['objects'])
before=geometry_hash()
for name,values in data['objects'].items():
    o=parts[name]
    assert o['surface_id']==values['surface_id'] and o['temperature_target_id']==values['target']
    assert math.isfinite(values['temperature_c']) and values['temperature_c']>-273.15
    assert isinstance(values['material'],str) and values['material']
# Commit only after every assignment is checked.
for name,values in data['objects'].items():
    o=parts[name];o['temperature_c']=values['temperature_c'];o['ir_material']=values['material'];o['physical_material']=values['material']
bpy.context.scene['thermal_configuration_file']='//configuration/thermal_authoring.yaml'
bpy.context.scene['thermal_configuration_sha256']=author_hash
bpy.context.scene['thermal_material_library_sha256']=data['material_library_sha256']
bpy.context.scene['thermal_condition_json']=json.dumps(data['condition'],ensure_ascii=False)
bpy.context.scene['thermal_configuration_phase']='prebinding; actual USD bindings and infrared validation pending'
bpy.ops.wm.save_as_mainfile(filepath=str(destination))
bpy.ops.wm.open_mainfile(filepath=str(destination))
after=geometry_hash();assert before==after,'Saved geometry/RGB/UV/normals changed'
for o in scene_parts():
    values=data['objects'][o.name]
    assert o['temperature_c']==values['temperature_c'] and o['ir_material']==values['material']
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
report={'result':'PASS_SAVED_BLENDER_CONFIGURATION','applied_surfaces':len(parts),'source_blend_sha256':source_hash,
        'configured_blend':str(destination),'configured_blend_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'geometry_rgb_uv_normals_sha256':before,'geometry_preserved_after_reload':True,
        'authoring_sha256':author_hash,'material_library_sha256':data['material_library_sha256'],
        'input_freshness_verified':True,'isaac_binding_or_validation_performed':False}
(PACKAGE/'blender_application.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
