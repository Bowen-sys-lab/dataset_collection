"""Extend the approved seven fixtures to thirteen without altering v1 assets."""
import json
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path('D:/IsaacSim_Demo/factory/model02_thermal_v2')


def extend_lights():
    assert Path(bpy.data.filepath) == ROOT/'model02_thermal_v2.blend'
    assert bpy.data.objects.get('ROAD_005') is None
    collection = bpy.data.collections['Model02_WeakLight']
    dg = bpy.context.evaluated_depsgraph_get()
    originals = json.loads((ROOT.parent/'model02_thermal_v1/lights.json').read_text(encoding='utf-8'))
    by_id = {item['light_id']:item for item in originals['lights']}
    additions = [('ROAD_005','ROAD_001',(690.,-825.)),
                 ('ROAD_006','ROAD_004',(733.,-873.)),
                 ('AREA_002','AREA_001',(701.,-778.)),
                 ('AREA_003','AREA_001',(673.,-802.)),
                 ('ENTRY_003','ENTRY_001',None),('WALL_001','ENTRY_001',None)]
    for name,source_id,xy in additions:
        source = bpy.data.objects[source_id]
        root = source.copy()
        root.name = name
        collection.objects.link(root)
        if xy:
            hit,location,normal,face,ground,matrix = bpy.context.scene.ray_cast(
                dg,Vector((*xy,35)),Vector((0,0,-1)),distance=40)
            assert hit and abs(location.z)<.5
            destination = Vector((*xy,float(location.z)))
            root['ground_z_m'] = float(location.z)
            root['ground_source_object'] = ground.name
        elif name == 'ENTRY_003':
            destination = source.matrix_world.translation+Vector((.743,.669,0))*8.
        else:
            destination = Vector((713.8532,-816.0596,4.45))+Vector((-.669,.743,0))*.08
            root['fixture_role'] = 'GC021 rear service facade wall light'
        delta = destination-source.matrix_world.translation
        root.location = destination
        bpy.context.view_layer.update()
        for original in list(source.children):
            obj = original.copy()
            obj.data = original.data.copy()
            obj.name = original.name.replace(source_id,name,1)
            obj.data.name = original.data.name.replace(source_id,name,1)
            collection.objects.link(obj)
            world = original.matrix_world.copy()
            world.translation += delta
            obj.parent = root
            obj.matrix_world = world
            if obj.type == 'MESH':
                obj['surface_id'] = original['surface_id'].replace(source_id,name)
                obj['temperature_target_id'] = obj.name.lower()
            elif obj.type == 'LIGHT':
                obj['light_id'] = name
        bpy.context.view_layer.update()
        light = bpy.data.objects[name+'_Light']
        origin = light.matrix_world.translation.copy()
        target = Vector(by_id[source_id]['target_m'])+delta
        if name == 'AREA_002':
            target = Vector((702.,-766.,.2))
        elif name == 'AREA_003':
            target = Vector((674.,-793.,.2))
        elif name == 'WALL_001':
            origin = destination+Vector((-.669,.743,0))*.12-Vector((0,0,.027))
            light.location = origin-root.matrix_world.translation
            target = destination+Vector((-.669,.743,0))*2.-Vector((0,0,4.1))
        direction = (target-origin).normalized()
        rotation = direction.to_track_quat('-Z','Y')
        light.rotation_mode = 'QUATERNION'
        light.rotation_quaternion = rotation
        for suffix in ['Housing','Diffuser']:
            part = bpy.data.objects[name+'_'+suffix]
            part.rotation_mode = 'QUATERNION'
            part.rotation_quaternion = rotation
            world = part.matrix_world.copy()
            world.translation = origin-direction*(.105 if suffix=='Housing' else .018)
            part.location = world.translation-root.matrix_world.translation
        if name == 'WALL_001':
            mount = bpy.data.objects[name+'_MountPlate']
            mount.rotation_mode = 'QUATERNION'
            mount.rotation_quaternion = Vector((-.669,.743,0)).to_track_quat('Z','Y')
            mount.location = -Vector((-.669,.743,0))*.045
        record = dict(by_id[source_id])
        record.update(light_id=name,blender_object=light.name,position_m=list(origin),target_m=list(target))
        originals['lights'].append(record)
    for item in originals['lights']:
        light = bpy.data.objects[item['blender_object']]
        power = 180. if item['light_id'].startswith('ROAD') else (240. if item['light_id'].startswith('AREA') else 40.)
        light.data.energy = power
        item['blender_radiant_power_w'] = power
    bpy.context.view_layer.update()
    originals['remaining_validation'] = 'Geometry refinement, export and native Isaac validation pending'
    (ROOT/'lights.json').write_text(json.dumps(originals,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'fixtures':len(originals['lights']),'road':6,'equipment':3,'wall':4}
