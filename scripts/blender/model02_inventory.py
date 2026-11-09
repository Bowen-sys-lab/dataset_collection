"""Source fingerprints and exact face-accounting checks for model02 refinement."""
from collections import Counter
import hashlib


def validate_partition_plan(snapshot, plan):
    sources = {o['name']:o['faces'] for o in snapshot['objects']}
    used = {name:Counter() for name in sources}
    unknown = set()
    missing_evidence = []
    invalid_faces = []
    for part in plan['parts']:
        source = part['source_object']
        if source not in sources:
            unknown.add(source)
            continue
        if part['action'] == 'replace' and not part.get('reconstruction'):
            missing_evidence.append(source)
        for face in part['source_face_ids']:
            if not 0 <= face < sources[source]:
                invalid_faces.append([source,face])
            else:
                used[source][face] += 1
    missing = [[source,i] for source,count in sources.items() for i in range(count) if not used[source][i]]
    duplicate = [[source,i] for source,counts in used.items() for i,n in counts.items() if n > 1]
    return {'complete':not(missing or duplicate or unknown or missing_evidence or invalid_faces),
            'missing_faces':missing,'duplicate_faces':duplicate,'unknown_sources':sorted(unknown),
            'replacement_without_evidence':missing_evidence,'invalid_faces':invalid_faces}


def snapshot_scene():
    import bpy
    import numpy as np
    objects = []
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not obj.name.startswith('11_29_gc'):
            continue
        mesh = obj.data
        digest = hashlib.sha256()
        for collection, field, width, dtype in [(mesh.vertices,'co',3,'float32'),
                (mesh.loops,'vertex_index',1,'int32'),(mesh.polygons,'material_index',1,'int32')]:
            values = np.empty(len(collection)*width,dtype=dtype)
            collection.foreach_get(field,values)
            digest.update(values.tobytes())
        for uv in mesh.uv_layers:
            values = np.empty(len(uv.data)*2,dtype='float32')
            uv.data.foreach_get('uv',values)
            digest.update(values.tobytes())
        normals = np.asarray([tuple(n.vector) for n in mesh.corner_normals],dtype='float32')
        digest.update(normals.tobytes())
        objects.append({'name':obj.name,'mesh':mesh.name,'vertices':len(mesh.vertices),
            'faces':len(mesh.polygons),'matrix_world':[list(row) for row in obj.matrix_world],
            'geometry_uv_material_normal_sha256':digest.hexdigest(),
            'materials':[m.name if m else None for m in mesh.materials]})
    return {'blender_file':bpy.data.filepath,'objects':objects}
